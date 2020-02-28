op_version_set = 1
class Module(Module):
  __parameters__ = ["weight", "bias", ]
  weight : Tensor
  bias : Tensor
  training : bool
  def forward(self: __torch__.torch.nn.modules.module.___torch_mangle_1.Module,
    argument_1: Tensor) -> Tensor:
    _0 = torch.addmm(self.bias, argument_1, torch.t(self.weight), beta=1, alpha=1)
    return _0
