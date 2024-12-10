# pip install transformers==4.44.2
# pip install coremltools numpy

import torch
from transformers.models.llama.modeling_llama import LlamaForCausalLM
from transformers import AutoTokenizer
import coremltools as ct
import numpy as np

# https://huggingface.co/docs/transformers/main/en/model_doc/llama#transformers.LlamaForCausalLM
class BaselineLlamaForCausalLM(LlamaForCausalLM):
    """Baseline LlamaForCausalLM model without key/value caching."""
    
    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: torch.LongTensor,
    ) -> torch.Tensor:
        out = super().forward(
            input_ids,
            attention_mask,
            use_cache=False,
        )
        return out.logits

# Check if MPS is available
# Transformer does not work on mps
# device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")

# print(device)

model_id: str = "meta-llama/Llama-3.1-8B-Instruct"
token = "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"
# pip install sentencepiece
tokenizer = AutoTokenizer.from_pretrained(model_id, use_auth_token=True)
torch_model = BaselineLlamaForCausalLM.from_pretrained(model_id).eval()
# Move model to MPS
# torch_model = torch_model.to(device)

# Prompt
prompt = "who is steph curry"
# Tokenize the prompt
# input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
input_ids = tokenizer(prompt, return_tensors="pt").input_ids

print(input_ids)
# tensor([[128000,  14965,    374,   3094,     71,  55178]])

# Generate response
# with torch.no_grad():
#     attention_mask = torch.ones_like(input_ids)  # Create attention mask
#     logits = torch_model(input_ids=input_ids, attention_mask=attention_mask)
#      # Print logits shape
#     print("Logits shape:", logits.shape)  # Shape: [batch_size, sequence_length, vocab_size]

#     # Decode the response
#     predicted_ids = torch.argmax(logits, dim=-1)
#     response = tokenizer.decode(predicted_ids[0], skip_special_tokens=True)
#     print("Response:", response)
generated_ids = torch_model.generate(
    input_ids=input_ids,
    max_length=50,
    temperature=0.7,
    top_k=50,
    top_p=0.9,
    no_repeat_ngram_size=2,
    early_stopping=True
)
response = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
print("Response:", response)


# convert to coreml

# batch_size, context_size = 1, 2048
# input_shape = (batch_size, context_size)

# # trace the PyTorch model
# example_inputs: tuple[torch.Tensor] = (
#     torch.zeros(input_shape, dtype=torch.int32),
#     torch.zeros(input_shape, dtype=torch.int32),
# )
# traced_model: torch.jit.ScriptModule = torch.jit.trace(
#     torch_model,
#     example_inputs=example_inputs,
# )

# # convert to Core ML format
# inputs: list[ct.TensorType] = [
#     ct.TensorType(shape=input_shape, dtype=np.int32, name="inputIds"),
#     ct.TensorType(shape=input_shape, dtype=np.int32, name="attentionMask"),
# ]

# outputs: list[ct.TensorType] = [ct.TensorType(dtype=np.float16, name="logits")]
# mlmodel: ct.models.MLModel = ct.convert(
#     traced_model,
#     inputs=inputs,
#     outputs=outputs,
#     minimum_deployment_target=ct.target.macOS13,
#     skip_model_load=True,
# )
"""
Core ML by default produces a Float16 precision model. 
Hence for this 8B model, the generated Core ML model will be about 16GB in size ((BitWidth / 2) x #ModelParameters).
We verify that the outputs of the Core ML and PyTorch models (which is in Float32 precision) match within a low tolerance.
"""