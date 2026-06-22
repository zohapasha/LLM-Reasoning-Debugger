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

problems = [
    {"question": "What is 17 times 4?", "answer": "68"},
    {"question": "If a train travels 60 miles in 1.5 hours, what is its speed in miles per hour?", "answer": "40"},
    {"question": "Sara has 23 apples. She gives away 9 and buys 15 more. How many apples does she have now?", "answer": "29"},
    {"question": "All cats are animals. Some animals are pets. Can we conclude that some cats are pets? Answer only Yes or No.", "answer": "No"},
    {"question": "If today is Wednesday, what day will it be in 10 days?", "answer": "Saturday"},
    {"question": "A box has 5 red balls and 3 blue balls. If you remove 2 red balls, how many balls are left in total?", "answer": "6"},
    {"question": "Tom is taller than Jerry. Jerry is taller than Sam. Is Tom taller than Sam? Answer only Yes or No.", "answer": "Yes"},
    {"question": "What is 144 divided by 12?", "answer": "12"},
    {"question": "A store had 50 shirts, sold 18, then restocked 25. How many shirts does it have now?", "answer": "57"},
    {"question": "If no birds are reptiles, and all sparrows are birds, are sparrows reptiles? Answer only Yes or No.", "answer": "No"},
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

    if "assistant" in full_text:
        reply_only = full_text.rsplit("assistant", 1)[-1]
    else:
        reply_only = full_text

    return reply_only.strip()

def extract_confidence(reply_only):
    matches = re.findall(r"CONFIDENCE:\s*(\d+)", reply_only, re.IGNORECASE)
    if matches:
        return int(matches[-1])
    return None

def extract_final_answer(reply_only):
    matches = re.findall(r"RESULT:\s*(.+)", reply_only, re.IGNORECASE)
    if matches:
        raw = matches[-1].strip()
        # Remove markdown formatting like ** or *
        raw = re.sub(r"[*_`]", "", raw)
        # Remove trailing punctuation
        raw = raw.strip(" .")
        return raw
    return None

def normalize_answer(answer_text):
    if answer_text is None:
        return None
    text = answer_text.lower().strip()
    # Remove common units/words that don't affect correctness
    text = re.sub(r"\b(mph|miles per hour|balls|apples|shirts|days?)\b", "", text)
    # Remove all punctuation and extra whitespace
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def answers_match(model_answer, correct_answer):
    norm_model = normalize_answer(model_answer)
    norm_correct = normalize_answer(correct_answer)
    if norm_model is None or norm_correct is None:
        return False
    return norm_model == norm_correct

results = []
correct_count = 0

for i, item in enumerate(problems):
    print(f"\nRunning problem {i+1}/{len(problems)}...")
    reply_only = ask_model(item["question"])
    extracted = extract_final_answer(reply_only)
    confidence = extract_confidence(reply_only)

    is_correct = answers_match(extracted, item["answer"])
    if is_correct:
        correct_count += 1

    results.append({
        "question": item["question"],
        "correct_answer": item["answer"],
        "model_extracted_answer": extracted,
        "is_correct": is_correct,
        "full_reply": reply_only
    })

    print(f"Q: {item['question']}")
    print(f"Correct answer: {item['answer']} | Model said: {extracted} (confidence: {confidence}) | Match: {is_correct}")
    

print(f"\n=== FINAL SCORE: {correct_count}/{len(problems)} ===")