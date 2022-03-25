import io
import pickle
import pickletools
from typing import Tuple, Optional

import torch


def pack_weight(weight, bias):
    return (0, weight, bias)


def unpack_weight(packed):
    version, weight, bias = packed
    assert version == 0
    return weight, bias

# Bundled input aware.
class MyObj1(object):
  packed: Tuple[int, int, int]
  bundled: str

  def __init__(self, weight, bias, bundled):
    self.packed = pack_weight(weight, bias)
    self.bundled = bundled

  def __getstate__(self):
    print("__getstate__")
    weight, bias = unpack_weight(self.packed)
    return (
        weight,
        bias,
        self.bundled,
    )

  def __setstate__(self, state):
    print("__setstate__")
    self.packed = pack_weight(state[0], state[1])
    self.bundled = state[2]

ser = pickle.dumps(MyObj1(5, 6, "hi"))
pickletools.dis(io.BytesIO(ser))
loaded = pickle.loads(ser)
print()
print(loaded.bundled)