import torch
import torch.nn as nn
from torch.utils._pytree import tree_map, tree_flatten, tree_unflatten
from typing import List, Any, Dict, Optional, Union, NamedTuple
from collections import defaultdict
from torch.utils._python_dispatch import TorchDispatchMode
from torch.utils.hooks import RemovableHandle
from torch._decomp import register_decomposition
from math import prod
from functools import wraps
from functools import partial


__all__ = ["FlopCounterMode", "register_flop_formula"]

aten = torch.ops.aten

def get_shape(i):
    if isinstance(i, torch.Tensor):
        return i.shape
    return i

flop_registry: Dict[Any, Any] = {}

def shape_wrapper(f):
    @wraps(f)
    def nf(*args, out=None, **kwargs):
        args, kwargs, out_shape = tree_map(get_shape, (args, kwargs, out))
        return f(*args, out_shape=out_shape, **kwargs)
    return nf

def register_flop_formula(targets, get_raw=False):
    def register_fun(flop_formula):
        if not get_raw:
            flop_formula = shape_wrapper(flop_formula)
        register_decomposition(targets, registry=flop_registry, unsafe=True)(flop_formula)
        return flop_formula

    return register_fun

@register_flop_formula(aten.mm)
def mm_flop(a_shape, b_shape, *args, out_shape=None, **kwargs) -> int:
    """Count flops for matmul."""
    # Inputs should be a list of length 2.
    # Inputs contains the shapes of two matrices.
    m, k = a_shape
    k2, n = b_shape
    assert k == k2
    # NB(chilli): Should be 2 * k - 1 technically for FLOPs.
    return m * n * 2 * k

@register_flop_formula(aten.addmm)
def addmm_flop(self_shape, a_shape, b_shape, out_shape=None, **kwargs) -> int:
    """Count flops for addmm."""
    return mm_flop(a_shape, b_shape)

@register_flop_formula(aten.bmm)
def bmm_flop(a_shape, b_shape, out_shape=None, **kwargs) -> int:
    """Count flops for the bmm operation."""
    # Inputs should be a list of length 2.
    # Inputs contains the shapes of two tensor.
    b, m, k = a_shape
    b2, k2, n = b_shape
    assert b == b2
    assert k == k2
    # NB(chilli): Should be 2 * k - 1 technically for FLOPs.
    flop = b * m * n * 2 * k
    return flop

@register_flop_formula(aten.baddbmm)
def baddbmm_flop(self_shape, a_shape, b_shape, out_shape=None, **kwargs) -> int:
    """Count flops for the baddbmm operation."""
    # Inputs should be a list of length 3.
    # Inputs contains the shapes of three tensors.
    return bmm_flop(a_shape, b_shape)


def conv_flop_count(
    x_shape: List[int],
    w_shape: List[int],
    out_shape: List[int],
    transposed: bool = False,
) -> int:
    """Count flops for convolution.

    Note only multiplication is
    counted. Computation for bias are ignored.
    Flops for a transposed convolution are calculated as
    flops = (x_shape[2:] * prod(w_shape) * batch_size).
    Args:
        x_shape (list(int)): The input shape before convolution.
        w_shape (list(int)): The filter shape.
        out_shape (list(int)): The output shape after convolution.
        transposed (bool): is the convolution transposed
    Returns:
        int: the number of flops
    """

    batch_size = x_shape[0]
    conv_shape = (x_shape if transposed else out_shape)[2:]
    c_out, c_in, *filter_size = w_shape

    """
    General idea here is that for a regular conv, for each point in the output
    spatial dimension we convolve the filter with something (hence
    `prod(conv_shape) * prod(filter_size)` ops). Then, this gets multiplied by
    1. batch_size, 2. the cross product of input and weight channels.

    For the transpose, it's not each point in the *output* spatial dimension but
    each point in the *input* spatial dimension.
    """
    # NB(chilli): I don't think this properly accounts for padding :think:
    # NB(chilli): Should be 2 * c_in - 1 technically for FLOPs.
    flop = prod(conv_shape) * prod(filter_size) * batch_size * c_out * c_in * 2
    return flop

@register_flop_formula([aten.convolution, aten._convolution])
def conv_flop(x_shape, w_shape, _bias, _stride, _padding, _dilation, transposed, *args, out_shape=None, **kwargs) -> int:
    """Count flops for convolution."""
    return conv_flop_count(x_shape, w_shape, out_shape, transposed=transposed)


@register_flop_formula(aten.convolution_backward)
def conv_backward_flop(
        grad_out_shape,
        x_shape,
        w_shape,
        _bias,
        _stride,
        _padding,
        _dilation,
        transposed,
        _output_padding,
        _groups,
        output_mask,
        out_shape) -> int:

    def t(shape):
        return [shape[1], shape[0]] + list(shape[2:])
    flop_count = 0

    """
    Let's say we have a regular 1D conv
    {A, B, C} [inp]
    {i, j} [weight]
    => (conv)
    {Ai + Bj, Bi + Cj} [out]

    And as a reminder, the transposed conv of the above is
    => {Ai, Aj + Bi, Bj + Ci, Cj} [transposed conv out]

    For the backwards of conv, we now have
    {D, E} [grad_out]
    {A, B, C} [inp]
    {i, j} [weight]

    # grad_inp as conv_transpose(grad_out, weight)
    Let's first compute grad_inp. To do so, we can simply look at all the
    multiplications that each element of inp is involved in. For example, A is
    only involved in the first element of the output (and thus only depends upon
    D in grad_out), and C is only involved in the last element of the output
    (and thus only depends upon E in grad_out)

    {Di, Dj + Ei, Ej} [grad_inp]

    Note that this corresponds to the below conv_transpose. This gives us the
    output_mask[0] branch, which is grad_inp.

    {D, E} [inp (grad_out)]
    {i, j} [weight]
    => (conv_transpose)
    {Di, Dj + Ei, Ej} [out (grad_inp)]

    I leave the fact that grad_inp for a transposed conv is just conv(grad_out,
    weight) as an exercise for the reader.

    # grad_weight as conv(inp, grad_out)
    To compute grad_weight, we again look at the terms in the output, which as
    a reminder is:
    => {Ai + Bj, Bi + Cj} [out]
    => {D, E} [grad_out]
    If we manually compute the gradient for the weights, we see it's
    {AD + BE, BD + CE} [grad_weight]

    This corresponds to the below conv
    {A, B, C} [inp]
    {D, E} [weight (grad_out)]
    => (conv)
    {AD + BE, BD + CE} [out (grad_weight)]

    # grad_weight of transposed conv as conv(grad_out, inp)
    As a reminder, the terms of the output of a transposed conv are:
    => {Ai, Aj + Bi, Bj + Ci, Cj} [transposed conv out]
    => {D, E, F, G} [grad_out]

    Manually computing the gradient for the weights, we see it's
    {AD + BE + CF, AE + BF + CG} [grad_weight]

    This corresponds to the below conv
    {D, E, F, G} [inp (grad_out)]
    {A, B, C} [weight (inp)]
    => (conv)
    {AD + BE + CF, AE + BF + CG} [out (grad_weight)]

    For the full backwards formula, there are also some details involving
    transpose of the batch/channel dimensions and groups, but I skip those for
    the sake of brevity (and they're pretty similar to matmul backwards)

    Check [conv backwards decomposition as conv forwards]
    """
    # grad_inp as conv_transpose(grad_out, weight)
    if output_mask[0]:
        grad_input_shape = get_shape(out_shape[0])
        flop_count += conv_flop_count(grad_out_shape, w_shape, grad_input_shape, not transposed)

    if output_mask[1]:
        grad_weight_shape = get_shape(out_shape[1])
        if transposed:
            # grad_weight of transposed conv as conv(grad_out, inp)
            flop_count += conv_flop_count(t(grad_out_shape), t(x_shape), t(grad_weight_shape), transposed=False)
        else:
            # grad_weight as conv(inp, grad_out)
            flop_count += conv_flop_count(t(x_shape), t(grad_out_shape), t(grad_weight_shape), transposed=False)

    return flop_count

def sdpa_flop_count(query_shape, key_shape, value_shape):
    """
    Count flops for self-attention.

    NB: We can assume that value_shape == key_shape
    """
    b, h, s_q, d_q = query_shape
    _b2, _h2, s_k, _d2 = key_shape
    _b3, _h3, _s3, d_v = value_shape
    assert b == _b2 == _b3 and h == _h2 == _h3 and d_q == _d2 and s_k == _s3 and d_q == _d2
    total_flops = 0
    # q: [b, h, s_q, d_q] @ k: [b, h, d_q, s_k] -> scores: [b, h, s_q, s_k]
    total_flops += bmm_flop((b * h, s_q, d_q), (b * h, d_q, s_k))
    # scores: [b, h, s_q, s_k] @ v: [b, h, s_k, d_v] -> out: [b, h, s_q, d_v]
    total_flops += bmm_flop((b * h, s_q, s_k), (b * h, s_k, d_v))
    return total_flops


@register_flop_formula([aten._scaled_dot_product_efficient_attention, aten._scaled_dot_product_flash_attention])
def sdpa_flop(query_shape, key_shape, value_shape, *args, out_shape=None, **kwargs) -> int:
    """Count flops for self-attention."""
    # NB: We aren't accounting for causal attention here
    return sdpa_flop_count(query_shape, key_shape, value_shape)


def sdpa_backward_flop_count(grad_out_shape, query_shape, key_shape, value_shape):
    total_flops = 0
    b, h, s_q, d_q = query_shape
    _b2, _h2, s_k, _d2 = key_shape
    _b3, _h3, _s3, d_v = value_shape
    _b4, _h4, _s4, _d4 = grad_out_shape
    assert b == _b2 == _b3 == _b4 and h == _h2 == _h3 == _h4 and d_q == _d2
    assert d_v == _d4 and s_k == _s3 and s_q == _s4
    total_flops = 0
    # Step 1: We recompute the scores matrix.
    # q: [b, h, s_q, d_q] @ k: [b, h, d_q, s_k] -> scores: [b, h, s_q, s_k]
    total_flops += bmm_flop((b * h, s_q, d_q), (b * h, d_q, s_k))

    # Step 2: We propagate the gradients through the score @ v operation.
    # gradOut: [b, h, s_q, d_v] @ v: [b, h, d_v, s_k] -> gradScores: [b, h, s_q, s_k]
    total_flops += bmm_flop((b * h, s_q, d_v), (b * h, d_v, s_k))
    # scores: [b, h, s_k, s_q] @ gradOut: [b, h, s_q, d_v] -> gradV: [b, h, s_k, d_v]
    total_flops += bmm_flop((b * h, s_k, s_q), (b * h, s_q, d_v))

    # Step 3: We propagate th gradients through the k @ v operation
    # gradScores: [b, h, s_q, s_k] @ k: [b, h, s_k, d_q] -> gradQ: [b, h, s_q, d_q]
    total_flops += bmm_flop((b * h, s_q, s_k), (b * h, s_k, d_q))
    # q: [b, h, d_q, s_q] @ gradScores: [b, h, s_q, s_k] -> gradK: [b, h, d_q, s_k]
    total_flops += bmm_flop((b * h, d_q, s_q), (b * h, s_q, s_k))
    return total_flops


@register_flop_formula([aten._scaled_dot_product_efficient_attention_backward, aten._scaled_dot_product_flash_attention_backward])
def sdpa_backward_flop(grad_out_shape, query_shape, key_shape, value_shape, *args, out_shape=None, **kwargs) -> int:
    """Count flops for self-attention backward."""
    return sdpa_backward_flop_count(grad_out_shape, query_shape, key_shape, value_shape)

flop_registry = {
    aten.mm: mm_flop,
    aten.addmm: addmm_flop,
    aten.bmm: bmm_flop,
    aten.baddbmm: baddbmm_flop,
    aten.convolution: conv_flop,
    aten._convolution: conv_flop,
    aten.convolution_backward: conv_backward_flop,
    aten._scaled_dot_product_efficient_attention: sdpa_flop,
    aten._scaled_dot_product_flash_attention: sdpa_flop,
    aten._scaled_dot_product_efficient_attention_backward: sdpa_backward_flop,
    aten._scaled_dot_product_flash_attention_backward: sdpa_backward_flop,
}

def normalize_tuple(x):
    if not isinstance(x, tuple):
        return (x,)
    return x


# Define the suffixes for different orders of magnitude
suffixes = ["", "K", "M", "B", "T"]
# Thanks BingChat!
def get_suffix_str(number):
    # Find the index of the appropriate suffix based on the number of digits
    # with some additional overflow.
    # i.e. 1.01B should be displayed as 1001M, not 1.001B
    index = max(0, min(len(suffixes) - 1, (len(str(number)) - 2) // 3))
    return suffixes[index]

def convert_num_with_suffix(number, suffix):
    index = suffixes.index(suffix)
    # Divide the number by 1000^index and format it to two decimal places
    value = f"{number / 1000 ** index:.3f}"
    # Return the value and the suffix as a string
    return value + suffixes[index]

def convert_to_percent_str(num, denom):
    if denom == 0:
        return "0%"
    return f"{num / denom:.2%}"

def _pytreeify_preserve_structure(f):
    @wraps(f)
    def nf(args):
        flat_args, spec = tree_flatten(args)
        out = f(*flat_args)
        return tree_unflatten(out, spec)

    return nf



# =============================================================================
# Mem tracker
# =============================================================================

import math
from collections import defaultdict
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, NamedTuple, Optional, Union
from torch._subclasses.fake_tensor import FakeTensorMode
import torch
import torch.nn as nn
from torch.utils._python_dispatch import TorchDispatchMode
from torch.utils._pytree import tree_map_only
from torch.utils.hooks import RemovableHandle
from torch.utils.weak import WeakIdKeyDictionary

_PYTORCH_MIN_ALLOCATE = 2**9

__all__ = ["MemoryTrackingMode"]


class _RefType(str, Enum):
    parameter = "parameter"
    buffer = "buffer"
    gradient = "gradient"
    activation = "activation"
    optstate = "optstate"

@dataclass
class _WeakRefInfo:
    def __init__(self, size: int, element_size: int, reftype: _RefType) -> None:
        self.size = size
        self.element_size = element_size
        self.reftype = reftype
        self.mem_consumed = self._calculate_mem_consumed()

    def _calculate_mem_consumed(self) -> int:
        return (math.ceil((self.size * self.element_size) / _PYTORCH_MIN_ALLOCATE)
                * _PYTORCH_MIN_ALLOCATE)

    def get_mem_consumed(self, st: torch.UntypedStorage) -> int:
        if st.size() != self.size:
            self.size = st.size()
            self.mem_consumed = self._calculate_mem_consumed()
        return self.mem_consumed
    
    def __repr__(self) -> str:
        return f"_WeakRefInfo: (size: {self.size}, element_size: {self.element_size}, reftype: {self.reftype}, mem_consumed: {self.mem_consumed})"


from collections.abc import MutableMapping
from collections import defaultdict

class FlopCounterMode(TorchDispatchMode):
    """
    ``FlopCounterMode`` is a context manager that counts the number of flops within its context.

    It does this using a ``TorchDispatchMode``.

    It also supports hierarchical output by passing a module (or list of
    modules) to FlopCounterMode on construction. If you do not need hierarchical
    output, you do not need to use it with a module.

    Example usage

    .. code-block:: python

        mod = ...
        flop_counter = FlopCounterMode(mod)
        with flop_counter:
            mod.sum().backward()

    """

    def __init__(
            self,
            mods: Optional[Union[torch.nn.Module, List[torch.nn.Module]]] = None,
            depth: int = 2,
            display: bool = True,
            custom_mapping: Optional[Dict[Any, Any]] = None):
        self.flop_counts: Dict[str, Dict[Any, int]] = defaultdict(lambda: defaultdict(int))
        self.depth = depth
        self.parents = ["Global"]
        self.in_backward = False
        self.display = display
        if custom_mapping is None:
            custom_mapping = {}
        if isinstance(mods, torch.nn.Module):
            mods = [mods]
        self.mods = mods
        # Keys will include the modules in `mods` and their submodules
        self._module_to_forward_hook_handles: Dict[nn.Module, _ForwardHookHandles] = {}
        self.flop_registry = {
            **flop_registry,
            **{k: v if getattr(v, "_get_raw", False) else shape_wrapper(v) for k, v in custom_mapping.items()}
        }

    
        # Mem tracker
        self.units = "B"
        self.depth = depth
        self.memory_tracking: Dict[str, Dict[str, Dict[str, int]]] = defaultdict(
            lambda: defaultdict(defaultdict)
        )
        self.parents: List[str] = []
        self._MEMORY_MAX: int = 0
        self.FIRST_OPT_ITER: bool = True
        # self._module_to_hook_handles: Dict[nn.Module, _ModuleHookHandles] = {}
        self._param_to_grad_hook_handles: Dict[nn.Parameter, RemovableHandle] = {}
        self._optimizer_hook_handle: Union[RemovableHandle, None] = None
        # key: UntypedStorage, value: _WeakRefInfo
        self.WINFO = WeakIdKeyDictionary()
        
        
        # Mem tracker extend
        # Key: module name
        # Value: UntypedStorage or _WeakRefInfo ?
        # self.memory_dict: Dict[str, Any] = defaultdict(set)
        self.memory_dict = defaultdict(set)
        
        # Collect all mem_states for a given module
        # Key: module name
    
    def _register_weak_info(self, module, name: str):
        # Can we move this into _enter_module used by `pre_forward_hook`?
        print("Registering weak info for", name)
        def _grad_hook(param: nn.Parameter, name):
            if param.grad is not None:
                print(f"call Grad hook for {name}")
                # TODO: looks like the grad already has weak info
                st = param.grad.untyped_storage()
                winfo = self.WINFO.get(st, None)
                if winfo is not None:
                    for st_name, st_lst in self.memory_dict.items():
                        if st in st_lst:
                            print(f"The grad(type: {winfo.reftype}) of {st_name} already added into memory dict, force it type into `grad` but from where ?")
                            winfo.reftype = _RefType.gradient
                            
                    self.memory_dict[name].add(st)
                    return
                print(f"Adding to gradient for {name}")
                winfo = _WeakRefInfo(st.size(), st.element_size(), _RefType.gradient)
                self.memory_dict[name].add(st)
                
        for p_name, param in module.named_parameters():
            st = param.untyped_storage()
            if self.WINFO.get(st, None) is not None:
                self.memory_dict[name].add(st)
                print(f"register post accumulate grad hook for {name}")
                _grad_hook = partial(_grad_hook, name=name)
                param.register_post_accumulate_grad_hook(_grad_hook)
                continue
            winfo = _WeakRefInfo(st.size(), st.element_size(), _RefType.parameter)
            self.WINFO[st] = winfo
            # TODO: Should use the WeakIdKeyDictionary to aviod messing up the garbage collection?
            print("Adding to memory dict", name)
            self.memory_dict[name].add(st)
            _grad_hook = partial(_grad_hook, name=name)
            grad_hook_handle = param.register_post_accumulate_grad_hook(_grad_hook)


        for _name, buffer in module.named_buffers():
            st = buffer.untyped_storage()
            if self.WINFO.get(st, None) is not None:
                continue
            winfo = _WeakRefInfo(st.size(), st.element_size(), _RefType.buffer)
            self.WINFO[st] = winfo
            self.memory_dict[name].add(winfo)

    def _register_forward_hooks(self):
        if self.mods is None:
            return                
        for mod in self.mods:
            prefix = type(mod).__name__
            for name, module in dict(mod.named_modules()).items():
                if name == "":
                    name = prefix
                else:
                    name = ".".join([prefix, name])
                self._register_weak_info(module, name)
                forward_pre_hook_handle = module.register_forward_pre_hook(self._enter_module(name))
                _ = module.register_forward_pre_hook(self._enter_module_mem(name, "Forward pass"))
                forward_hook_handle = module.register_forward_hook(self._exit_module(name))
                self._module_to_forward_hook_handles[module] = _ForwardHookHandles(
                    forward_pre_hook_handle, forward_hook_handle
                )

    def _update_stats(self):
        curr_use: int = 0
        for st, winfo in self.WINFO.items():
            curr_use += winfo.get_mem_consumed(st)

        if self._MEMORY_MAX < curr_use:
            self._MEMORY_MAX = curr_use

    def _track(self, t: torch.Tensor, parents_set: set[str], func_name:str):
        st = t.untyped_storage()
        if self.WINFO.get(st, None) is not None:
            return
        winfo = _WeakRefInfo(st.size(), st.element_size(), _RefType.activation)
        self.WINFO[st] = winfo
        for parent in parents_set:
            self.memory_dict[parent].add(st)
            # self.memory_dict[(parent, func_name)].add(st)
        self._update_stats()

    def _get_current_memory_allocated(self) -> Dict[str, int]:
        mem_stats = defaultdict(int)
        mem_stats[_RefType.parameter.name] = 0
        mem_stats[_RefType.gradient.name] = 0
        mem_stats[_RefType.optstate.name] = 0
        mem_stats[_RefType.activation.name] = 0
        mem_stats[_RefType.buffer.name] = 0
        for st, winfo in self.WINFO.items():
            mem_stats[winfo.reftype.name] += winfo.get_mem_consumed(st)
        mem_stats["TRACKER_total"] = sum([m for m in mem_stats.values()])
        if torch.cuda.is_available():
            mem_stats["CUDA_total"] = torch.cuda.memory_allocated()
        return mem_stats


    def print_mem_stats(self, depth=3):
        print(f"======================================")
        if depth is None:
            depth = self.depth
        for mod in self.memory_dict.keys():
            mod_depth = mod.count(".") + 1
            if mod_depth > depth:
                continue
            print(f"Module:  {mod}")
            mem_stats = defaultdict(int)
            # mem_stats[_RefType.parameter.name] = 0
            mem_stats[_RefType.gradient.name] = 0
            # mem_stats[_RefType.optstate.name] = 0
            # mem_stats[_RefType.activation.name] = 0
            # mem_stats[_RefType.buffer.name] = 0
            st_lst = self.memory_dict.get(mod, [])
            for target_type, stats in mem_stats.items():
                target_winfo_lst = _collect_winfo_by_type(st_lst, target_type, self.WINFO)
                print(f"collected {len(target_winfo_lst)} {target_type} winfos")
                mem_stats[target_type] = sum([winfo.get_mem_consumed(st) for winfo, st in target_winfo_lst])
            self._print_mem_stats(mem_stats)

    def print_mem_tracking(self, depth=3):
        print(f"======================================")
        if depth is None:
            depth = self.depth
        for mod in self.memory_tracking.keys():
            mod_depth = mod.count(".") + 1
            if mod_depth > depth:
                continue
            print(f"Module:  {mod}")
            for state, stats in self.memory_tracking[mod].items():
                print(f"{state}")
                self._print_mem_stats(stats)
            print()

    def _print_mem_stats(self, stats: Optional[Dict[str, int]] = None):
        if stats is None:
            stats = self._get_current_memory_allocated()

        def rounding_fn(value, divisor, precision) -> Union[float, int]:
            if divisor == 1:
                return value
            return round(value / divisor, precision)
        divisor = 1
        if self.units == "GB":
            divisor = 2**30
        elif self.units == "MB":
            divisor = 2**20
        elif self.units == "KB":
            divisor = 2**10

        for mem_type, mem_val in stats.items():
            print(f"\t{mem_type}: {rounding_fn(mem_val, divisor, 2)} {self.units}")

    def get_max_memory(self) -> int:
        return self._MEMORY_MAX

    def _deregister_forward_hooks(self):
        for forward_hook_handles in self._module_to_forward_hook_handles.values():
            forward_hook_handles[0].remove()
            forward_hook_handles[1].remove()
        self._module_to_forward_hook_handles.clear()

    def _enter_module(self, name):
        def f(module, inputs):
            out = _pytreeify_preserve_structure(self._create_pre_module(name))(inputs)
            return out

        return f

    def _enter_module_mem(self, name: str, state: str):
        def f(module: nn.Module, args: Any):
            # self.parents.append(name)
            self.memory_tracking[name][state] = self._get_current_memory_allocated()

        return f

    def _exit_module(self, name):
        def f(module, inputs, outputs):
            outputs = _pytreeify_preserve_structure(self._create_post_module(name))(outputs)
            return outputs
        return f

    def _create_post_module(self, name):
        class PushState(torch.autograd.Function):
            @staticmethod
            def forward(ctx, *args):
                assert self.parents[-1] == name, f"{self.parents[-1]} is not {name}"
                self.parents.pop()
                args = tree_map(lambda x: x.clone() if isinstance(x, torch.Tensor) else x, args)
                return args

            @staticmethod
            def backward(ctx, *grad_outs):
                self.in_backward = True
                self.parents.append(name)
                return grad_outs

        return PushState.apply

    def _create_pre_module(self, name):
        class PopState(torch.autograd.Function):
            @staticmethod
            def forward(ctx, *args):
                if self.in_backward:
                    self.parents = ["Global"]
                    self.in_backward = True
                self.parents.append(name)
                args = tree_map(lambda x: x.clone() if isinstance(x, torch.Tensor) else x, args)
                return args

            @staticmethod
            def backward(ctx, *grad_outs):
                assert self.parents[-1] == name
                self.parents.pop()
                return grad_outs

        return PopState.apply

    def get_total_flops(self) -> int:
        return sum(self.flop_counts['Global'].values())

    def get_flop_counts(self) -> Dict[str, Dict[Any, int]]:
        """Return the flop counts as a dictionary of dictionaries.

        The outer
        dictionary is keyed by module name, and the inner dictionary is keyed by
        operation name.

        Returns:
            Dict[str, Dict[Any, int]]: The flop counts as a dictionary.
        """
        return {k: dict(v) for k, v in self.flop_counts.items()}

    def get_table(self, depth=None):
        if depth is None:
            depth = self.depth
        if depth is None:
            depth = 999999

        import tabulate
        tabulate.PRESERVE_WHITESPACE = True
        header = ["Module", "FLOP", "% Total"]
        values = []
        global_flops = self.get_total_flops()
        global_suffix = get_suffix_str(global_flops)
        is_global_subsumed = False

        def process_mod(mod_name, depth):
            nonlocal is_global_subsumed

            total_flops = sum(self.flop_counts[mod_name].values())

            is_global_subsumed |= total_flops >= global_flops

            padding = " " * depth
            values = []
            values.append([
                padding + mod_name,
                convert_num_with_suffix(total_flops, global_suffix),
                convert_to_percent_str(total_flops, global_flops)
            ])
            for k, v in self.flop_counts[mod_name].items():
                values.append([
                    padding + " - " + str(k),
                    convert_num_with_suffix(v, global_suffix),
                    convert_to_percent_str(v, global_flops)
                ])
            return values

        for mod in self.flop_counts.keys():
            if mod == 'Global':
                continue
            mod_depth = mod.count(".") + 1
            if mod_depth > depth:
                continue

            cur_values = process_mod(mod, mod_depth - 1)
            values.extend(cur_values)

        # We do a bit of messing around here to only output the "Global" value
        # if there are any FLOPs in there that aren't already fully contained by
        # a module.
        if 'Global' in self.flop_counts and not is_global_subsumed:
            for idx, value in enumerate(values):
                values[idx][0] = " " + values[idx][0]

            values = process_mod('Global', 0) + values

        if len(values) == 0:
            values = [["Global", "0", "0%"]]

        return tabulate.tabulate(values, headers=header, colalign=("left", "right", "right"))

    def __enter__(self):
        self.flop_counts.clear()
        self._register_forward_hooks()
        super().__enter__()
        return self

    def __exit__(self, *args):
        if self.display:
            print(self.get_table(self.depth))
        self._deregister_forward_hooks()
        super().__exit__(*args)

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs if kwargs else {}
        out = func(*args, **kwargs)
        func_packet = func._overloadpacket
        self._track = partial(self._track, parents_set=set(self.parents), func_name=func)
        tree_map_only(torch.Tensor, self._track, out)
        if func_packet in self.flop_registry:
            flop_count_func = self.flop_registry[func_packet]
            flop_count = flop_count_func(*args, **kwargs, out=out)  # type: ignore[operator]
            if len(set(self.parents)) != len(self.parents):
                print(
                    "The module hierarchy tracking seems to be messed up."
                    "Please file a bug or just run the flop counter without"
                    "tracking the module hierarchy (i.e. `with FlopCounterMode():`)"
                )
            print(f"========= parents: {self.parents}")
            for par in set(self.parents):
                self.flop_counts[par][func_packet] += flop_count
        # 

        return out

class _ForwardHookHandles(NamedTuple):
    forward_pre_hook_handle: RemovableHandle
    forward_hook_handle: RemovableHandle


# =============================================================================
# Flop Counter example
# =============================================================================


def _collect_winfo_by_type(st_lst, target_type, WINFO):
    filter_lst = []
    for st in st_lst:
        winfo = WINFO[st]
        if winfo.reftype.name == target_type:
            filter_lst.append((winfo, st))
    return filter_lst

def _display_by_type(memory_dict, target_type, WINFO):
    print(f"================ Target type: {target_type} ================ ")
    for name, st_lst in memory_dict.items():
        for st in  st_lst:
            winfo = WINFO[st]
            if winfo.reftype == target_type:
                print(f"name: {name},\n \t winfo: {winfo}")

# Example 1

def test1():
    import torch
    class Mod(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.ln = torch.nn.LayerNorm(10)
            self.fc1 = torch.nn.Linear(10, 20)
            self.fc2 = torch.nn.Linear(10, 10)
            self.weight = torch.nn.Parameter(torch.randn(20, 10))
            
        
        def forward(self, x):
            x = self.ln(x)
            x = self.fc1(x)
            x1 = torch.nn.functional.relu(x) # 4, 20
            x1 = torch.matmul(x1, self.weight) # 4, 10
            x1 = self.fc2(x1) # 4 10
            
            return x1
        

    class ComplexModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            
            self.mod_lst = torch.nn.Sequential(*[Mod() for _ in range(2)])
            self.fc = torch.nn.Linear(10, 10)
            
        def forward(self, x):
            # for mod in self.mod_lst:
            #     x = mod(x)
            x = self.mod_lst(x)
            x = self.fc(x)
            return x

    mod = ComplexModel()
    inputs = torch.randn(4, 10)
    # from torch.utils.flop_counter import FlopCounterMode

    with FlopCounterMode(mod, depth=3) as fc:
        out = mod(inputs)
        print(out.shape)
        _display_by_type(fc.memory_dict, _RefType.parameter, fc.WINFO)
        _display_by_type(fc.memory_dict, _RefType.activation,fc.WINFO)
        _display_by_type(fc.memory_dict, _RefType.buffer, fc.WINFO)
        fc.print_mem_stats()
    #     # fc.print_mem_stats()
    #     # fc._display_mem_stats(depth=2)


# Example 2
def test2():
    class DummyModel(nn.Module):
        def __init__(self, layers: int, dim: int):
            super(DummyModel, self).__init__()
            self._module_list = []
            for _ in range(layers):
                self._module_list.extend([nn.Linear(dim, dim), nn.ReLU()])
            self.module = nn.Sequential(*self._module_list)

        def forward(self, x):
            return self.module(x)

    batch_size = 100
    layers = 2
    dim = 10000
    if torch.cuda.is_available():
        torch.set_default_device("cuda")
        torch.cuda.reset_peak_memory_stats()

    model = DummyModel(layers, dim)
    optim = torch.optim.Adam(model.parameters(), fused=True)
    mem_tracker = FlopCounterMode(model)
    mem_tracker.units = "MB"
    with mem_tracker as mt:
        input_batch = torch.randn(batch_size, dim)
        print("After Model and mini-batch init:")
        mt.print_mem_stats()
        output = model(input_batch)
        print("After Forward:")
        mt.print_mem_stats()
        output.sum().backward()
        output = None
        print("After Backward:")
        mt.print_mem_stats()
        optim.step()
        print("After Opt Step:")
        mt.print_mem_stats()
        optim.zero_grad()
        print("After Zero Grad:")
        mt.print_mem_stats()
        MAX_MEMORY = mt.get_max_memory()

    print(f"Tracker measured: {MAX_MEMORY}")
    if torch.cuda.is_available():
        CUDA_MEMORY_MAX = torch.cuda.max_memory_allocated()
        print(f"Cuda measured: {CUDA_MEMORY_MAX}")
        print(f"Peak comparison ratio: {MAX_MEMORY/CUDA_MEMORY_MAX}")

# =============================================================================
"""
# module:
    - parameters
        - gradients
        - optstate
    - buffers
    - activation

#         parameter: 0 B
#         gradient: 0 B
#         optstate: 0 B
#         activation: 4096 B
#         buffer: 0 B
#         TRACKER_total: 4096 B
#         CUDA_total: 0 B


# Q1: map weakinfo instances with module name
# module_name: weakinfo
# weakinfo: {module_name, }
# Q2: alias param of optimizer with module name
"""
# =============================================================================

# Depth = 2
# Module              FLOP    % Total
# ----------------  ------  ---------
# ComplexModel        8800    100.00%
#  - aten.addmm       5600     63.64%
#  - aten.mm          3200     36.36%
#  ComplexModel.fc     800      9.09%
#   - aten.addmm       800      9.09%

# Depth = 3
# Module                      FLOP    % Total
# ------------------------  ------  ---------
# ComplexModel                8800    100.00%
#  - aten.addmm               5600     63.64%
#  - aten.mm                  3200     36.36%
#   ComplexModel.mod_lst.0    4000     45.45%
#    - aten.addmm             2400     27.27%
#    - aten.mm                1600     18.18%
#   ComplexModel.mod_lst.1    4000     45.45%
#    - aten.addmm             2400     27.27%
#    - aten.mm                1600     18.18%
#  ComplexModel.fc             800      9.09%
#   - aten.addmm               800      9.09%

# Sequential depth=3
# Module                            FLOP    % Total
# ------------------------------  ------  ---------
# ComplexModel                      8800    100.00%
#  - aten.addmm                     5600     63.64%
#  - aten.mm                        3200     36.36%
# ****ComplexModel.mod_lst          8000     90.91%
# **** - aten.addmm                 4800     54.55%
# **** - aten.mm                    3200     36.36%
# ********ComplexModel.mod_lst.0    4000     45.45%
# ******** - aten.addmm             2400     27.27%
# ******** - aten.mm                1600     18.18%
# ********ComplexModel.mod_lst.1    4000     45.45%
# ******** - aten.addmm             2400     27.27%
# ******** - aten.mm                1600     18.18%
# ****ComplexModel.fc                800      9.09%
# **** - aten.addmm                  800      9.09%

#     Module            FLOP    % Total
# --------------  ------  ---------
# Mod               4800    100.00%
#  - aten.addmm     3200     66.67%
#  - aten.mm        1600     33.33%
#  Mod.fc1          1600     33.33%
#   - aten.addmm    1600     33.33%
#  Mod.fc2          1600     33.33%
#   - aten.addmm    1600     33.33%
    
#         parameter: 0 B
#         gradient: 0 B
#         optstate: 0 B
#         activation: 4096 B
#         buffer: 0 B
#         TRACKER_total: 4096 B
#         CUDA_total: 0 B