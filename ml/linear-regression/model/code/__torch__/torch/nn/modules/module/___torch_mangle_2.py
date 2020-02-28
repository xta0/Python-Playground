op_version_set = 1
class Module(Module):
  __parameters__ = []
  training : bool
  __annotations__["0"] = __torch__.torch.nn.modules.module.Module
  __annotations__["1"] = __torch__.torch.nn.modules.module.___torch_mangle_0.Module
  __annotations__["2"] = __torch__.torch.nn.modules.module.___torch_mangle_1.Module
  def forward(self: __torch__.torch.nn.modules.module.___torch_mangle_2.Module,
    input: Tensor) -> Tensor:
    _0 = getattr(self, "1")
    _1 = (getattr(self, "0")).forward(input, )
    _2 = (getattr(self, "2")).forward((_0).forward(_1, ), )
    return _2
