import torch
from torch.utils._pytree import tree_map_only
from torch.utils._python_dispatch import TorchDispatchMode
from torch._subclasses.fake_tensor import FakeTensorMode
from torch.utils.weak import WeakIdKeyDictionary
import weakref
import math

# Track all the memory being used by Tensors.
# Only max is tracked but others can be added.
MEMORY_USE = WeakIdKeyDictionary()
MEMORY_MAX = 0
# Minimum allocation size 
PYTORCH_MIN_ALLOCATE = 2**9


from enum import Enum


_PYTORCH_MIN_ALLOCATE = 2**9
class _RefType(str, Enum):
    parameter = "parameter"
    buffer = "buffer"
    gradient = "gradient"
    activation = "activation"
    optstate = "optstate"

from dataclasses import dataclass
@dataclass
class _WeakRefInfo:
    def __init__(self, size: int, element_size: int, reftype: _RefType) -> None:
        self.size = size
        self.element_size = element_size
        self.reftype = reftype
        self.mem_consumed = self._calculate_mem_consumed()

    def _calculate_mem_consumed(self) -> int:
        return math.ceil((self.size * self.element_size) / _PYTORCH_MIN_ALLOCATE) * _PYTORCH_MIN_ALLOCATE

    def get_mem_consumed(self, st: torch.UntypedStorage) -> int:
        if st.size() != self.size:
            self.size = st.size()
            self.mem_consumed = self._calculate_mem_consumed()
        return self.mem_consumed

    def __repr__(self) -> str:
        return f"_WeakRefInfo: (size: {self.size}, element_size: {self.element_size}, reftype: {self.reftype}, mem_consumed: {self.mem_consumed})"


# The key idea is categorizing the memory type of new created tensor by the stage
# If we make the categorization simple, the memory usage of each module? is okay?

class Stage(Enum):
    MODEL_INIT = "MODEL_INIT"                  # <-> parameter, and buffer?
    BEFORE_FORWARD = "BEFORE_FORWARD"
    FORWARD = "FORWARD"                        # <-> activation
    BACKWARD = "BACKWARD"                      # <-> gradient and parameter
    OPTIMIZER_STEP = "OPTIMIZER_STEP"          # <-> optstate
    

from functools import partial
from collections import defaultdict
from torch.utils.weak import WeakIdKeyDictionary
from typing import Dict, Optional, Any, List, Union, Tuple
from torch.utils.flop_counter import _pytreeify_preserve_structure


from torch.utils._pytree import tree_map, tree_flatten, tree_unflatten



# Use this Mode to call track on every Tensor being created by functions
class MemoryTrackingMode(TorchDispatchMode):
    def __init__(self, 
                 mods: Optional[Union[torch.nn.Module, List[torch.nn.Module]]] = None,
                 optim = None,
                 depth: int = 2,
                 display: bool = True,
                 custom_mapping: Optional[Dict[Any, Any]] = None):
        self.mods = mods
        self.optim = optim
        self.stage = Stage.BEFORE_FORWARD
        # stage: {parent name: {tensor storage set}}
        self.mem_trackers: Dict[Stage, Dict[str, weakref.WeakSet[torch.storage.UntypedStorage]]] = defaultdict(lambda:defaultdict(weakref.WeakSet))

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

    def _show_mem(self, stage: Stage):
        print(f"%%%% Memory usage at {stage}")
        for par, st_lst in self.mem_trackers[stage].items():
            curr_use = 0
            for k in st_lst:
                if k in MEMORY_USE:
                    curr_use += math.ceil(k.size() * k.element_size()/PYTORCH_MIN_ALLOCATE) * PYTORCH_MIN_ALLOCATE
            print(f"Memory usage for {par}: {curr_use}")

    def print_mem_stats(self, stage: Stage = None):
        if stage:
            self._show_mem(stage)
        else:
            for k in Stage:
                self._show_mem(k)


    def _enter_module(self, name):
        def f(module, inputs):
            out = _pytreeify_preserve_structure(self._create_pre_module(name))(inputs)
            return out

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
                self.stage = Stage.FORWARD
                assert self.parents[-1] == name, f"{self.parents[-1]} is not {name}"
                print(f"PushState pop {name} from {self.parents}")
                self.parents.pop()
                args = tree_map(lambda x: x.clone() if isinstance(x, torch.Tensor) else x, args)
                return args

            @staticmethod
            def backward(ctx, *grad_outs):
                self.stage = Stage.BACKWARD
                self.in_backward = True
                print(f"PushState add {name} into {self.parents}")
                self.parents.append(name)
                return grad_outs

        return PushState.apply

    def _create_pre_module(self, name):
        class PopState(torch.autograd.Function):
            @staticmethod
            def forward(ctx, *args):
                self.stage = Stage.FORWARD
                if self.in_backward:
                    self.parents = ["Global"]
                    self.in_backward = False
                print(f"PopState add {name} into {self.parents}")
                self.parents.append(name)
                args = tree_map(lambda x: x.clone() if isinstance(x, torch.Tensor) else x, args)
                return args

            @staticmethod
            def backward(ctx, *grad_outs):
                self.stage = Stage.BACKWARD
                assert self.parents[-1] == name
                print(f"PopState pop {name} from {self.parents}")
                self.parents.pop()
                return grad_outs

        return PopState.apply


    def _forward_pre_hook(self, module, inputs) -> None:
        name = module._private_name
        return self._enter_module(name)(module, inputs)
    
    def _forward_after_hook(self, module, inputs, outputs) -> None:
        name = module._private_name
        return self._exit_module(name)(module, inputs, outputs)
    
    

    def local_pre_hook(self, opt: torch.optim.Optimizer, args: Tuple[Any], kwargs: Dict[Any, Any]):
        self.stage = Stage.OPTIMIZER_STEP
        
        
    def _register_hook_for_optim(self):
        if self.optim is None:
            return
        else:
            self.optim.register_step_pre_hook(self.local_pre_hook)
            # self.optim.register_step_post_hook(self.local_post_hook)

    def attach_name_to_module(self, mods):
        for mod in mods:
            for name, module in dict(mod.named_modules()).items():
                prefix = type(mod).__name__
                if name == "":
                    name = prefix
                else:
                    name = ".".join([prefix, name])
                module._private_name = name



    def __enter__(self):
        self.flop_counts.clear()
        # self._register_forward_hooks()
        self.attach_name_to_module(self.mods)
        self._register_hook_for_optim()
        self.pre_handle = torch.nn.modules.module.register_module_forward_pre_hook(self._forward_pre_hook)
        self.post_handle = torch.nn.modules.module.register_module_forward_hook(self._forward_after_hook)
        
        super().__enter__()
        return self

    def __exit__(self, *args):
        # if self.display:
        #     print(self.get_table(self.depth))
        # self._deregister_forward_hooks()
        self.pre_handle.remove()
        self.post_handle.remove()
        super().__exit__(*args)


    def update_stats(self, msg = ""):
        # print(f"----------------------------{msg}")
        global MEMORY_MAX
        curr_use = 0
        for k, v in MEMORY_USE.items():
            curr_use += math.ceil(k.size() * k.element_size()/PYTORCH_MIN_ALLOCATE) * PYTORCH_MIN_ALLOCATE

        if MEMORY_MAX < curr_use:
            MEMORY_MAX = curr_use

    # Should be called on every Tensor created
    def _track(self, t:torch.Tensor, name: str, parents_set:set[str]):
        def cb(_):
            self.update_stats("Tensor deleted")
        st = t.untyped_storage()
        if MEMORY_USE.get(st, None) is None:
            # print(f"Created new a tensor from {name}, size is {t.size()}, id = {id(st)}")
            wt = weakref.ref(st, cb)
            staged_mem = self.mem_trackers[self.stage]
            for par in parents_set:
                staged_mem[par].add(st)
        else:
            # print(f"Reuse a tensor from {name}, size is {t.size()}, id = {id(st)}")
            wt = MEMORY_USE.get(st)
        MEMORY_USE[st] = wt
        self.update_stats(msg = "new tensor created")
    
    def __torch_dispatch__(self, func, types, args, kwargs=None):
        kwargs = kwargs if kwargs else {}
        out = func(*args, **kwargs)
        func_packet = func._overloadpacket
        self._track = partial(self._track, name=func.__name__, parents_set=set(self.parents))
        tree_map_only(torch.Tensor, self._track, out)
        return out
        # if func_packet in self.flop_registry:
        #     flop_count_func = self.flop_registry[func_packet]
        #     flop_count = flop_count_func(*args, **kwargs, out=out)  # type: ignore[operator]
        #     if len(set(self.parents)) != len(self.parents):
        #         print(
        #             "The module hierarchy tracking seems to be messed up."
        #             "Please file a bug or just run the flop counter without"
        #             "tracking the module hierarchy (i.e. `with FlopCounterMode():`)"
        #         )
        #     print(f"========= parents: {self.parents}")
        #     for par in set(self.parents):
        #         self.flop_counts[par][func_packet] += flop_count


if __name__ == "__main__":
    # Use FakeTensorMode to run the code without any actual data
    with FakeTensorMode():
        # def f(a):
        #     b = a * 10
        #     d = b + 3
        #     return d

        # a = torch.rand(100)
        # f(a)
        # f(a)
        # print(f"Just f: {MEMORY_MAX}")
        # c = f(a)
        # c = f(a)
        # print(f"f with return: {MEMORY_MAX}")
        
        from torch import nn
        class DummyModel(nn.Module):
            def __init__(self, layers: int, dim: int):
                super(DummyModel, self).__init__()
                self._module_list = []
                for _ in range(layers):
                    self._module_list.extend([nn.Linear(dim, dim), nn.ReLU()])
                self.module = nn.Sequential(*self._module_list)

            def forward(self, x):
                return self.module(x)
        
        class SubModel(nn.Module):
            def __init__(self, layers: int = 2, dim: int = 1024):
                super().__init__()
                self.fc1 = torch.nn.Linear(dim, dim*2)
                self.fc2 = torch.nn.Linear(dim * 2, dim)


            def forward(self, x):
                x = self.fc1(x)
                x = self.fc2(x)
                return x
        
        class NewModel(nn.Module):
            def __init__(self, layers: int, dim: int):
                super().__init__()
                self.fc1 = torch.nn.Linear(dim, dim*2)
                self.fc2 = torch.nn.Linear(dim * 2, dim)
                # self.sub1 = SubModel(dim)
                # self.sub2 = SubModel(dim)
                
            def forward(self, x):
                x = self.fc1(x)
                x = self.fc2(x)
                # x = self.sub1(x)
                # x = self.sub2(x)
                return x
        
        def print_status(mgs = "", flag="="):
            print(f"{flag*10} {mgs} {flag*10}")

        batch_size = 100
        layers = 2
        dim = 2**10
        print_status("Ready for Model init", "=")
        model =  NewModel(layers, dim)
        optim = torch.optim.Adam(model.parameters(), fused=False)
        with MemoryTrackingMode(model, optim=optim) as mt:
            print_status("Finished Model init", "-")
            print_status("Ready Opt init", "=")
            
            print_status("Finished Opt init", "=")
            # mem_tracker = FlopCounterMode(model)
            # mem_tracker.units = "MB"
            # TO resolve it
            num_iters = 2

            for i in range(num_iters):
                print_status("Ready for Forward", "=")
                input_batch = torch.randn(batch_size, dim)
                
                mt.print_mem_stats()
                output = model(input_batch)
            
                loss = output.sum()
                print_status("After for Forward", "=")
                mt.print_mem_stats()
                print_status("Ready for backward", "=")

                loss.backward()
                print_status("After for backward", "=")
                # output = None
                # print("After Backward:")
                mt.print_mem_stats()
                print_status("Ready for opt step", "=")
                optim.step()
                # print("After Opt Step:")
                mt.print_mem_stats()
                print_status("After for opt step", "=")
                mt.print_mem_stats()
                print_status("Ready for opt zero grad", "=")
                optim.zero_grad()
                mt.print_mem_stats()
            
