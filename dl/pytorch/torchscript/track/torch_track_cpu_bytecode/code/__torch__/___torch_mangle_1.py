class SegmentationWrapper(Module):
  __parameters__ = ["3", "4", "5", "6", "7", "8", "9", "10", "11", "12", "13", "14", "15", "16", "17", "18", "19", "20", "21", "22", "23", "24", "25", "26", "1", "2", ]
  __annotations__ = []
  __annotations__["3"] = Tensor
  __annotations__["4"] = Tensor
  __annotations__["5"] = Tensor
  __annotations__["6"] = Tensor
  __annotations__["7"] = Tensor
  __annotations__["8"] = Tensor
  __annotations__["9"] = Tensor
  __annotations__["10"] = Tensor
  __annotations__["11"] = Tensor
  __annotations__["12"] = Tensor
  __annotations__["13"] = Tensor
  __annotations__["14"] = Tensor
  __annotations__["15"] = Tensor
  __annotations__["16"] = Tensor
  __annotations__["17"] = Tensor
  __annotations__["18"] = Tensor
  __annotations__["19"] = Tensor
  __annotations__["20"] = Tensor
  __annotations__["21"] = Tensor
  __annotations__["22"] = Tensor
  __annotations__["23"] = Tensor
  __annotations__["24"] = Tensor
  __annotations__["25"] = Tensor
  __annotations__["26"] = Tensor
  __annotations__["1"] = Tensor
  __annotations__["2"] = Tensor
  training : bool
  metadata : Tuple[int, int, str, str, int]
  def get_metadata(self: __torch__.___torch_mangle_1.SegmentationWrapper) -> Tuple[int, int, str, str, int]:
    return self.metadata
  def forward(self: __torch__.___torch_mangle_1.SegmentationWrapper,
    input: Tensor) -> Tensor:
    _0 = getattr(self, "2")
    _1 = getattr(self, "1")
    _2 = getattr(self, "26")
    _3 = getattr(self, "25")
    _4 = getattr(self, "24")
    _5 = getattr(self, "23")
    _6 = getattr(self, "22")
    _7 = getattr(self, "21")
    _8 = getattr(self, "20")
    _9 = getattr(self, "19")
    _10 = getattr(self, "18")
    _11 = getattr(self, "17")
    _12 = getattr(self, "16")
    _13 = getattr(self, "15")
    _14 = getattr(self, "14")
    _15 = getattr(self, "13")
    _16 = getattr(self, "12")
    _17 = getattr(self, "11")
    _18 = getattr(self, "10")
    _19 = getattr(self, "9")
    _20 = getattr(self, "8")
    _21 = getattr(self, "7")
    _22 = getattr(self, "6")
    _23 = getattr(self, "5")
    _24 = getattr(self, "4")
    _25 = getattr(self, "3")
    _26 = torch.sub(input, CONSTANTS.c0, alpha=1)
    input0 = torch.mul(_26, CONSTANTS.c1)
    input1 = torch._convolution(input0, _25, _24, [1, 1], [1, 1], [1, 1], False, [0, 0], 1, False, False, True, True)
    input2 = torch.max_pool2d(input1, [2, 2], [2, 2], [0, 0], [1, 1], False)
    input3 = torch.relu(input2)
    input4 = torch._convolution(input3, _23, _22, [1, 1], [1, 1], [1, 1], False, [0, 0], 1, False, False, True, True)
    input5 = torch.max_pool2d(input4, [2, 2], [2, 2], [0, 0], [1, 1], False)
    input6 = torch.relu(input5)
    input7 = torch._convolution(input6, _21, _20, [1, 1], [1, 1], [1, 1], False, [0, 0], 1, False, False, True, True)
    input8 = torch.max_pool2d(input7, [2, 2], [2, 2], [0, 0], [1, 1], False)
    input9 = torch.relu(input8)
    input10 = torch._convolution(input9, _19, _18, [1, 1], [1, 1], [1, 1], False, [0, 0], 1, False, False, True, True)
    input11 = torch.max_pool2d(input10, [2, 2], [2, 2], [0, 0], [1, 1], False)
    input12 = torch.relu(input11)
    input13 = torch._convolution(input12, _17, _16, [1, 1], [1, 1], [1, 1], False, [0, 0], 1, False, False, True, True)
    input14 = torch.max_pool2d(input13, [2, 2], [2, 2], [0, 0], [1, 1], False)
    input15 = torch.relu(input14)
    input16 = torch._convolution(input15, _15, _14, [1, 1], [1, 1], [1, 1], False, [0, 0], 1, False, False, True, True)
    input17 = torch.relu(input16)
    input18 = torch._convolution(input17, _13, _12, [1, 1], [1, 1], [1, 1], False, [0, 0], 1, False, False, True, True)
    input19 = torch.upsample_nearest2d(input18, None, [2., 2.])
    input20 = torch.add(input19, input13, alpha=1)
    input21 = torch.relu(input20)
    input22 = torch._convolution(input21, _11, _10, [1, 1], [1, 1], [1, 1], False, [0, 0], 1, False, False, True, True)
    input120 = torch.upsample_nearest2d(input22, None, [2., 2.])
    input23 = torch.add(input120, input10, alpha=1)
    input24 = torch.relu(input23)
    input25 = torch._convolution(input24, _9, _8, [1, 1], [1, 1], [1, 1], False, [0, 0], 1, False, False, True, True)
    input121 = torch.upsample_nearest2d(input25, None, [2., 2.])
    input26 = torch.add(input121, input7, alpha=1)
    input27 = torch.relu(input26)
    input28 = torch._convolution(input27, _7, _6, [1, 1], [1, 1], [1, 1], False, [0, 0], 1, False, False, True, True)
    input122 = torch.upsample_nearest2d(input28, None, [2., 2.])
    input29 = torch.add(input122, input4, alpha=1)
    input30 = torch.relu(input29)
    input31 = torch._convolution(input30, _5, _4, [1, 1], [1, 1], [1, 1], False, [0, 0], 1, False, False, True, True)
    input132 = torch.upsample_nearest2d(input31, None, [2., 2.])
    input33 = torch.add(input132, input1, alpha=1)
    input32 = torch.relu(input33)
    input34 = torch._convolution(input32, _3, _2, [1, 1], [1, 1], [1, 1], False, [0, 0], 1, False, False, True, True)
    input35 = torch.sigmoid(input34)
    _27 = torch._convolution(input35, _1, _0, [1, 1], [1, 1], [1, 1], False, [0, 0], 1, False, False, True, True)
    return _27
