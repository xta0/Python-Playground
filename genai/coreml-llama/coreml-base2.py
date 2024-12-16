from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model_id: str = "meta-llama/Llama-3.1-8B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=False)
model = AutoModelForCausalLM.from_pretrained(model_id)
model.eval()

prompt = "What is GenAI?"
input_ids = tokenizer(prompt, return_tensors='pt').input_ids.to(model.device)

# Use model.generate to do all steps at once
output_ids = model.generate(
    input_ids,
    max_new_tokens=100,   # adjust as needed
    temperature=0.7,      # adjust as needed
    top_p=0.9,            # adjust as needed
    do_sample=True        # sampling rather than greedy
)

output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
print(output_text)