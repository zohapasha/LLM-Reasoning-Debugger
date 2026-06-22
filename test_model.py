from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model_name = "Qwen/Qwen2.5-1.5B-Instruct"

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(model_name)

print("Loading model... (this downloads the model the first time, may take a few minutes)")
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="cuda"
)

question = "What is 17 times 4? Show your reasoning step by step."

messages = [
    {"role": "user", "content": question}
]

input_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tokenizer(input_text, return_tensors="pt").to("cuda")

print("Generating answer...")
output = model.generate(**inputs, max_new_tokens=200)

answer = tokenizer.decode(output[0], skip_special_tokens=True)

print("\n--- MODEL'S ANSWER ---")
print(answer)