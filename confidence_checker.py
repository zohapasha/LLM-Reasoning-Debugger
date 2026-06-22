from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import re

model_name = "Qwen/Qwen2.5-1.5B-Instruct"

print("Loading model...")
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    dtype=torch.float16,
    device_map="cuda"
)

hard_questions = [
    "What is 8347 multiplied by 263?",
    "A bat and ball cost $1.10 together. The bat costs $1.00 more than the ball. How much does the ball cost?",
    "What is the capital of Burkina Faso?",
    "If you flip a fair coin 5 times, what is the probability of getting exactly 3 heads?",
    "Is the number 9,007,199,254,740,993 a prime number? Answer only Yes or No.",
]

def ask_model(question):
    prompt = (
        question
        + " Give your reasoning. Then, on the very last two lines, write exactly this format:\n"
        + "RESULT: <only the bare answer, no extra words>\n"
        + "CONFIDENCE: <a number from 0 to 100>"
    )
    messages = [{"role": "user", "content": prompt}]
    input_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(input_text, return_tensors="pt").to("cuda")
    output = model.generate(**inputs, max_new_tokens=300, do_sample=False)
    full_text = tokenizer.decode(output[0], skip_special_tokens=True)
    reply_only = full_text.rsplit("assistant", 1)[-1] if "assistant" in full_text else full_text
    return reply_only.strip()

for q in hard_questions:
    print(f"\n--- Q: {q} ---")
    reply = ask_model(q)
    print(reply)