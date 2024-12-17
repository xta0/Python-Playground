import coremltools as ct
from transformers import AutoTokenizer
import numpy as np

# Load the model
mlmodel = ct.models.MLModel("./llama2-7b-hf.mlpackage")
print(mlmodel.get_spec().WhichOneof("Type"))
# print(mlmodel)
# Get the model spec and inspect the program code
spec = mlmodel.get_spec()
# print(spec)

program = spec.mlProgram
# print(program)
# functions = program.functions

# # Iterate over functions and operations
# for function in functions:
#     print(f"Function Name: {function.name}")
#     for op in function.block.operations:
#         print(f"Operation Type: {op.type}, Inputs: {op.inputs}, Outputs: {op.outputs}")

# inference
# Tokenize a query
query = "What is GenAI?"
model_id = "meta-llama/Llama-2-7b-hf"
tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=False)
tokenizer.pad_token = tokenizer.eos_token
inputs = tokenizer(
    query,
    return_tensors="np",
    padding="max_length",  # Pad to the max_length
    max_length=24,         # Match the shape expected by the model
    truncation=True        # Ensure it doesn't exceed max_length
)
input_ids_np = inputs["input_ids"]
attention_mask_np = inputs["attention_mask"]

# Ensure input tensors have the correct shape and dtype
input_ids_np = np.array(input_ids_np, dtype=np.int32)
attention_mask_np = np.array(attention_mask_np, dtype=np.int32)

print(input_ids_np.shape, input_ids_np.dtype)  # Debugging
print(attention_mask_np.shape, attention_mask_np.dtype)

# Initial setup for generation
generated_ids = input_ids_np  # Start with the input query tokens
batch_size, context_size = generated_ids.shape


max_new_tokens = 10  # Number of tokens to generate
context_length = 24  # Fixed sequence length expected by Core ML model

# Start with input IDs and attention mask
generated_ids = input_ids_np  # Original input IDs
attention_mask = attention_mask_np

# Generate tokens in a loop
for i in range(max_new_tokens):
    # Predict logits using the Core ML model
    coreml_outputs = mlmodel.predict({
        "inputIds": generated_ids,
        "attentionMask": attention_mask_np
    })
    coreml_logits_np = coreml_outputs["logits"]  # Shape: [batch_size, seq_len, vocab_size]
    
    # Get the logits for the last token
    next_token_logits = coreml_logits_np[:, -1, :]  # Shape: [batch_size, vocab_size]
    
    # Select the next token (greedy decoding - pick the token with max probability)
    next_token_id = np.argmax(next_token_logits, axis=-1).reshape((batch_size, 1))  # Shape: [batch_size, 1]
    
  # Append the new token and slide the window
    generated_ids = np.concatenate([generated_ids, next_token_id], axis=1)
    attention_mask = np.concatenate([attention_mask, np.array([[1]], dtype=np.int32)], axis=1)
    
    # Keep only the last 'context_length' tokens
    if generated_ids.shape[1] > context_length:
        generated_ids = generated_ids[:, -context_length:]  # Trim to last 24 tokens
        attention_mask = attention_mask[:, -context_length:]
    
    # Decode the latest token to a string
    next_token_str = tokenizer.decode(next_token_id[0], skip_special_tokens=True)
    print(f"Step {i + 1}, Decoded Token: {next_token_str}")
    
    # Decode the entire generated sequence so far
    generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    print(f"Generated Text So Far: {generated_text}")
    
    # Stop if the model generates the end-of-sequence token (e.g., <|endoftext|>)
    if next_token_id[0][0] == tokenizer.eos_token_id:
        print("End of sequence token encountered. Stopping generation.")
        break

# # quantization
# op_config = ct.optimize.coreml.OpLinearQuantizerConfig(
#     mode="linear_symmetric",
#     dtype="int4",
#     granularity="per_block",
#     block_size=32,
# )
# config = ct.optimize.coreml.OptimizationConfig(global_config=op_config)
# mlmodel_int4 = ct.optimize.coreml.linear_quantize_weights(
#     mlmodel, config=config
# )

# mlmodel.save("llama2-7b-hf-quant.mlpackage")  # 13GB, fp32