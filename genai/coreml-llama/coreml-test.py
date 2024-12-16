import coremltools as ct

# Load the model
model = ct.models.MLModel("./BaselineLlamaForCausalLM.mlpackage/Data/com.apple.CoreML/model.mlmodel")
print(model)
"""
input {
  name: "inputIds"
  type {
    multiArrayType {
      shape: 1
      shape: 2048
      dataType: INT32
    }
  }
}
input {
  name: "attentionMask"
  type {
    multiArrayType {
      shape: 1
      shape: 2048
      dataType: INT32
    }
  }
}
output {
  name: "logits"
  type {
    multiArrayType {
      shape: 1
      shape: 2048
      shape: 128256
      dataType: FLOAT16
    }
  }
}
metadata {
  userDefined {
    key: "com.github.apple.coremltools.version"
    value: "8.1"
  }
  userDefined {
    key: "com.github.apple.coremltools.source"
    value: "torch==1.12.0"
  }
  userDefined {
    key: "com.github.apple.coremltools.source_dialect"
    value: "TorchScript"
  }
}
"""