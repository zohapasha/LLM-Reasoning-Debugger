from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import torch.nn.functional as F
import re

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

model_name = "Qwen/Qwen2.5-1.5B-Instruct"

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(model_name)

print("Loading model...")
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="cuda",
    dtype=torch.bfloat16,
    attn_implementation="eager"
)
model.eval()
print("Model loaded. Server ready.")

class Question(BaseModel):
    question: str

class JudgeRequest(BaseModel):
    question: str
    correct_answer: str
    model_answer: str

def extract_confidence(reply_only):
    matches = re.findall(r"CONFIDENCE:\s*(\d+)", reply_only, re.IGNORECASE)
    if matches:
        return int(matches[-1])
    return None

def extract_final_answer(reply_only):
    matches = re.findall(r"RESULT:\s*(.+?)(?:\n|CONFIDENCE:|$)", reply_only, re.IGNORECASE | re.DOTALL)
    if matches:
        raw = matches[-1].strip()
        raw = re.sub(r"[*_`]", "", raw)
        return raw.strip(" .")
    return None

def compute_entropy(probs_tensor, top_k=50):
    top_probs, _ = torch.topk(probs_tensor, min(top_k, probs_tensor.shape[-1]))
    top_probs = top_probs[top_probs > 1e-10]
    top_probs = top_probs / torch.sum(top_probs)
    entropy = -torch.sum(top_probs * torch.log2(top_probs))
    return entropy.item()

PUNCTUATION_AND_FILLER = {".", ",", "\"", "'", ":", ";", "!", "?", "-", "(", ")", " ", "\n", "the", "a", "is", "of", "to"}

def ask_with_entropy(question, max_new_tokens=400):
    prompt = (
        question
        + " Give your reasoning. Then, on the very last two lines, write exactly this format:\n"
        + "RESULT: <only the bare answer, no extra words>\n"
        + "CONFIDENCE: <a number from 0 to 100>"
    )
    messages = [{"role": "user", "content": prompt}]
    input_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(input_text, return_tensors="pt").to("cuda")
    input_ids = inputs["input_ids"]
    prompt_len = input_ids.shape[1]

    generated_tokens = []
    token_data = []
    stop_tokens = {tokenizer.eos_token_id}

    if hasattr(tokenizer, "special_tokens_map") and "additional_special_tokens" in tokenizer.special_tokens_map:
        for tok in tokenizer.special_tokens_map["additional_special_tokens"]:
            stop_tokens.add(tokenizer.convert_tokens_to_ids(tok))

    full_ids = input_ids[0].tolist()
    current_input_ids = input_ids
    past_key_values = None

    with torch.no_grad():
        for _ in range(max_new_tokens):
            outputs = model(
                current_input_ids,
                past_key_values=past_key_values,
                use_cache=True,
                output_attentions=True,
            )
            past_key_values = outputs.past_key_values

            next_token_logits = outputs.logits[0, -1, :].float()
            probs = F.softmax(next_token_logits, dim=-1)

            top_prob, top_token_id = torch.max(probs, dim=-1)

            probs_copy = probs.clone()
            probs_copy[top_token_id] = 0
            second_prob, second_token_id = torch.max(probs_copy, dim=-1)

            entropy_bits = compute_entropy(probs)
            chosen_token_id = top_token_id.item()

            if chosen_token_id in stop_tokens:
                break

            last_layer_attn = outputs.attentions[-1][0]
            avg_attn = last_layer_attn.mean(dim=0)
            attn_row = avg_attn[-1]

            top_attended_idx = torch.argmax(attn_row).item()
            top_attended_score = round(attn_row[top_attended_idx].item(), 4)
            top_attended_token = tokenizer.decode([full_ids[top_attended_idx]])
            attended_to_prompt = top_attended_idx < prompt_len

            sorted_indices = torch.argsort(attn_row, descending=True)
            meaningful_idx = None
            meaningful_score = None
            for idx in sorted_indices.tolist():
                candidate_token = tokenizer.decode([full_ids[idx]]).strip().lower()
                if candidate_token not in PUNCTUATION_AND_FILLER and candidate_token != "":
                    meaningful_idx = idx
                    meaningful_score = round(attn_row[idx].item(), 4)
                    break
            top_attended_meaningful_token = tokenizer.decode([full_ids[meaningful_idx]]) if meaningful_idx is not None else None

            generated_tokens.append(chosen_token_id)
            raw_token_str = tokenizer.decode([chosen_token_id])

            token_data.append({
                "token": raw_token_str,
                "probability": round(top_prob.item(), 4),
                "runner_up_token": tokenizer.decode([second_token_id.item()]),
                "runner_up_probability": round(second_prob.item(), 4),
                "entropy_bits": round(entropy_bits, 4),
                "top_attended_token": top_attended_token,
                "top_attended_score": top_attended_score,
                "attended_to_prompt": attended_to_prompt,
                "top_attended_meaningful_token": top_attended_meaningful_token,
                "top_attended_meaningful_score": meaningful_score
            })

            full_ids.append(chosen_token_id)
            current_input_ids = top_token_id.view(1, 1)

    reply_only = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
    return reply_only, token_data

def compute_hallucination_risk(token_data, window_size=6, entropy_threshold=2.0):
    n = len(token_data)
    if n == 0:
        return {
            "risk_zones": [], "overall_risk_label": "No data",
            "avg_entropy": 0, "max_window_entropy": 0,
            "flagged_token_count": 0, "total_tokens": 0, "flagged_pct": 0
        }

    entropies = [t["entropy_bits"] for t in token_data]
    flagged = [False] * n
    window_entropies = []

    for i in range(n):
        start = max(0, i - window_size // 2)
        end = min(n, i + window_size // 2 + 1)
        window_avg = sum(entropies[start:end]) / (end - start)
        window_entropies.append(window_avg)
        if window_avg > entropy_threshold:
            for j in range(start, end):
                flagged[j] = True

    risk_zones = []
    i = 0
    while i < n:
        if flagged[i]:
            zone_start = i
            while i < n and flagged[i]:
                i += 1
            zone_end = i
            zone_text = "".join(t["token"] for t in token_data[zone_start:zone_end]).strip()
            zone_avg_entropy = sum(entropies[zone_start:zone_end]) / (zone_end - zone_start)
            risk_zones.append({
                "start_token_index": zone_start,
                "end_token_index": zone_end,
                "text": zone_text,
                "avg_entropy_bits": round(zone_avg_entropy, 3),
                "token_count": zone_end - zone_start
            })
        else:
            i += 1

    avg_entropy = sum(entropies) / n
    max_window_entropy = max(window_entropies) if window_entropies else 0
    flagged_count = sum(flagged)
    flagged_pct = round(100 * flagged_count / n, 1)

    if flagged_pct == 0:
        overall_label = "Low risk - no sustained high-uncertainty passages detected."
    elif flagged_pct < 15:
        overall_label = "Moderate risk - a small portion of this response shows sustained uncertainty."
    else:
        overall_label = "High risk - large portions show sustained high uncertainty, matching typical hallucination patterns."

    return {
        "risk_zones": risk_zones,
        "overall_risk_label": overall_label,
        "avg_entropy": round(avg_entropy, 3),
        "max_window_entropy": round(max_window_entropy, 3),
        "flagged_token_count": flagged_count,
        "total_tokens": n,
        "flagged_pct": flagged_pct
    }

def summarize_confidence(token_confidences):
    if not token_confidences:
        return None
    probs = [tc["probability"] for tc in token_confidences]
    avg_confidence = sum(probs) / len(probs)
    low_count = sum(1 for p in probs if p < 0.6)
    moderate_count = sum(1 for p in probs if 0.6 <= p <= 0.9)
    high_count = sum(1 for p in probs if p > 0.9)
    total = len(probs)
    low_pct = round(100 * low_count / total, 1)

    if avg_confidence > 0.85 and low_pct < 10:
        label = "Low uncertainty - the model was consistently confident."
    elif low_pct > 30:
        label = "High uncertainty - a large portion of this response was shaky."
    else:
        label = "Mixed confidence - parts of this response were solid, others uncertain."

    return {
        "average_confidence": round(avg_confidence * 100, 1),
        "low_confidence_token_pct": low_pct,
        "high_confidence_token_count": high_count,
        "moderate_confidence_token_count": moderate_count,
        "low_confidence_token_count": low_count,
        "total_tokens": total,
        "label": label
    }

@app.post("/ask")
def ask(q: Question):
    reply_only, token_data = ask_with_entropy(q.question)
    summary = summarize_confidence(token_data)
    hallucination_risk = compute_hallucination_risk(token_data)

    return {
        "full_reasoning": reply_only,
        "final_answer": extract_final_answer(reply_only),
        "stated_confidence": extract_confidence(reply_only),
        "token_confidences": token_data,
        "confidence_summary": summary,
        "hallucination_risk": hallucination_risk
    }

@app.post("/judge")
def judge_endpoint(req: JudgeRequest):
    model_clean = " ".join(str(req.model_answer).lower().split()).strip(" ._`*")
    correct_clean = " ".join(str(req.correct_answer).lower().split()).strip(" ._`*")

    if not model_clean:
        return {"is_correct": False}

    if model_clean == correct_clean or correct_clean in model_clean or model_clean in correct_clean:
        return {"is_correct": True}

    model_nums = re.findall(r"\d+", model_clean)
    correct_nums = re.findall(r"\d+", correct_clean)
    if model_nums and correct_nums and model_nums[-1] == correct_nums[-1]:
        return {"is_correct": True}

    judge_prompt = (
        f"System: You are an automated evaluation grader. Review the answers objectively.\n"
        f"Question: {req.question}\n"
        f"Correct Answer: {req.correct_answer}\n"
        f"Model Answer: {req.model_answer}\n\n"
        f"Is the Model Answer correct based on the Ground Truth? Answer 'YES' or 'NO'."
    )
    messages = [{"role": "user", "content": judge_prompt}]
    input_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(input_text, return_tensors="pt").to("cuda")

    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=4, do_sample=False)

    raw_reply = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip().upper()
    is_correct = any(word in raw_reply for word in ["YES", "TRUE", "CORRECT"])

    return {"is_correct": is_correct}