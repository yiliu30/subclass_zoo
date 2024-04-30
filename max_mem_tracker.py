import torch
from torch.utils._pytree import tree_map_only
from torch.utils._python_dispatch import TorchDispatchMode
from torch._subclasses.fake_tensor import FakeTensorMode
from torch.utils.weak import WeakIdKeyDictionary
import weakref

# Track all the memory being used by Tensors.
# Only max is tracked but others can be added.
MEMORY_USE = WeakIdKeyDictionary()
MEMORY_MAX = 0
MEMORY_ID = 0





# Use this Mode to call track on every Tensor being created by functions
class MemoryTrackingMode(TorchDispatchMode):
    def __init__(self, model):
        self.model = model
        self.mem_tracking = WeakIdKeyDictionary()


    def update_stats(self):
        global MEMORY_MAX
        curr_use = 0
        
        for k, v in MEMORY_USE.items():
            curr_use += k.nelement() * k.element_size()
        print(f"MEMORY_USE: {curr_use}")
        if MEMORY_MAX < curr_use:
            MEMORY_MAX = curr_use

    # Should be called on every Tensor created
    def track(self, t):
        def cb(_):
            self.update_stats()

        wt = weakref.ref(t, cb)
        MEMORY_USE[t] = wt
        self.update_stats()


    def _register_weak_info(self, module, name: str):
        # Can we move this into _enter_module used by `pre_forward_hook`?
        # def _grad_hook(param: nn.Parameter, name):
            # if param.grad is not None:
            #     print(f"call Grad hook for {name}")
            #     # TODO: looks like the grad already has weak info
            #     st = param.grad.untyped_storage()
            #     winfo = self.WINFO.get(st, None)
            #     if winfo is not None:
            #         for st_name, st_lst in self.memory_dict.items():
            #             if st in st_lst:
            #                 print(f"The grad(type: {winfo.reftype}) of {st_name} already added into memory dict, force it type into `grad` but from where ?")
            #                 winfo.reftype = _RefType.gradient
                            
            #         self.memory_dict[name].add(st)
            #         return
            #     print(f"Adding to gradient for {name}")
            #     winfo = _WeakRefInfo(st.size(), st.element_size(), _RefType.gradient)
            #     self.memory_dict[name].add(st)
    
        for p_name, param in module.named_parameters():
            st = param.untyped_storage()
            if self.mem_tracking.get(st, None) is None:
                mod_name_set = self.mem_tracking[st] = set([name])
            else:
                mod_name_set = self.mem_tracking[st]
                mod_name_set.add(name)
            # winfo = _WeakRefInfo(st.size(), st.element_size(), _RefType.parameter)
            # self.WINFO[st] = winfo
            # # TODO: Should use the WeakIdKeyDictionary to aviod messing up the garbage collection?
            # print("Adding to memory dict", name)
            # self.memory_dict[name].add(st)
            # _grad_hook = partial(_grad_hook, name=name)
            # grad_hook_handle = param.register_post_accumulate_grad_hook(_grad_hook)

                
    def _register_forward_hooks(self, mod):

        prefix = type(mod).__name__
        for name, module in dict(mod.named_modules()).items():
            if name == "":
                name = prefix
            else:
                name = ".".join([prefix, name])
            self._register_weak_info(module, name)


    def __enter__(self):
        self._register_forward_hooks(self.model)
        super().__enter__()
        return self
    
    def __torch_dispatch__(self, func, types, args, kwargs=None):
        res = func(*args, **kwargs or {})
        func_sign = f"func: {func}, args: {args}, kwargs: {kwargs}"
        print(f"MemoryTrackingMode: {func_sign}")
        tree_map_only(torch.Tensor, self.track, res)
        return res


if __name__ == "__main__":
    
    class DummyModel(torch.nn.Module):
        def __init__(self, layers: int, dim: int):
            super(DummyModel, self).__init__()
            self._module_list = []
            for _ in range(layers):
                self._module_list.extend([torch.nn.Linear(dim, dim), torch.nn.ReLU()])
            self.module = torch.nn.Sequential(*self._module_list)

        def forward(self, x):
            return self.module(x)
    # Use FakeTensorMode to run the code without any actual data
    with FakeTensorMode():
        batch_size = 100
        layers = 5
        dim = 10000
        if torch.cuda.is_available():
            torch.set_default_device("cuda")
            torch.cuda.reset_peak_memory_stats()

        model = DummyModel(layers, dim)
        with MemoryTrackingMode(model) as mt:
            for st, mod_name_set in mt.mem_tracking.items():
                print(f"st: {st}, mod_name_set: {mod_name_set}")
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




        print(f"---------------------")
        model(torch.randn(batch_size, dim))
        