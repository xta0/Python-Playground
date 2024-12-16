# pip install transformers==4.44.2
# pip install coremltools numpy

import torch
from transformers.models.llama.modeling_llama import LlamaForCausalLM
from transformers import AutoTokenizer


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

model_id: str = "meta-llama/Llama-2-7b-hf"
torch_model = BaselineLlamaForCausalLM.from_pretrained(model_id).eval()

tokenizer = AutoTokenizer.from_pretrained(model_id, legacy=False)

prompt = "what is genai?"
inputs = tokenizer(prompt, return_tensors='pt')
print("inputs: ", inputs)

# Extract input_ids and attention_mask
input_ids = inputs["input_ids"]
attention_mask = inputs["attention_mask"]

max_new_tokens = 10
generated_ids = input_ids.clone()

for i in range(max_new_tokens):
    print(i)
    # Forward pass
    logits = torch_model(input_ids=generated_ids, attention_mask=torch.ones_like(generated_ids))
    print(logits)
    # Get the last token's logits
    next_token_logits = logits[:, -1, :]
    # Greedy decoding: pick the token with the highest probability
    next_token_id = next_token_logits.argmax(dim=-1).unsqueeze(0)
    
    # Append next token to the generated_ids
    generated_ids = torch.cat([generated_ids, next_token_id], dim=1)
    # Decode the entire sequence back into text
    output_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    print(output_text)
    
    # If the model has an EOS token and we encounter it, break
    if next_token_id.item() == tokenizer.eos_token_id:
        break

# Decode the entire sequence back into text
output_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
print(output_text)

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
# # mlmodel: ct.models.MLModel = ct.convert(
# #     traced_model,
# #     inputs=inputs,
# #     outputs=outputs,
# #     minimum_deployment_target=ct.target.macOS15,
# #     skip_model_load=True,
# # )
# mlmodel.save("BaselineLlamaForCausalLM.mlpackage")  # 15GB, fp32

# """
# Core ML by default produces a Float16 precision model. 
# Hence for this 8B model, the generated Core ML model will be about 16GB in size ((BitWidth / 2) x #ModelParameters).
# We verify that the outputs of the Core ML and PyTorch models (which is in Float32 precision) match within a low tolerance.
# """