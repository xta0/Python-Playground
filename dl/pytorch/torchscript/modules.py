import torch
import numpy as np

fake_weights = np.random.rand(128, 3, 3, 3)
fake_bias = np.random.rand(128)
fake_workplace = {"w": fake_weights, "b": fake_bias}


class SegmentationWrapper(torch.nn.Module):
    def __init__(self, workspace):
        super().__init__()
        self.workspace = workspace
        self.weights = torch.nn.Parameter(torch.from_numpy(
            self.workspace["w"]))
        self.bias = torch.nn.Parameter(torch.from_numpy(self.workspace["b"]))

    def forward(self, input):
        weights = getattr(self, "weights")
        bias = getattr(self, "bias")
        output = torch.nn.functional.conv2d(input,
                                            weights,
                                            bias,
                                            padding=1)
        return output


module = SegmentationWrapper(fake_workplace)
input = torch.rand((1, 3, 224, 224), dtype=float)
output = module(input)

example = torch.ones(1, 3, 224, 224, dtype=float)
sm = torch.jit.trace(module, example)
print(sm._c.dump_to_str(False, False, False))
sm.save("./seg1.zip")

# sm = torch.jit._load_for_mobile("./seg.ptl")
for name, param in sm.named_parameters():
    fp = param.float()
    qp = torch._choose_qparams_per_tensor(fp, False)
    qt = torch.quantize_per_tensor(fp, qp[0], qp[1], torch.quint8)
    setattr(sm, name, qt)
    
sm._save_for_lite_interpreter("./seg2.zip")

