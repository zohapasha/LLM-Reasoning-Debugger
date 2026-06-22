import csv
import json
import requests
import pandas as pd
import time

QUESTIONS = [
    # ---------- EASY TIER ----------
    {"question": "What is 24 multiplied by 6?", "correct_answer": "144", "tier": "easy"},
    {"question": "What is the capital of France?", "correct_answer": "Paris", "tier": "easy"},
    {"question": "What is the capital of Japan?", "correct_answer": "Tokyo", "tier": "easy"},
    {"question": "How many continents are there on Earth?", "correct_answer": "7", "tier": "easy"},
    {"question": "What is the boiling point of water in Celsius at sea level?", "correct_answer": "100", "tier": "easy"},
    {"question": "What is 100 divided by 4?", "correct_answer": "25", "tier": "easy"},
    {"question": "What is the largest planet in our solar system?", "correct_answer": "Jupiter", "tier": "easy"},
    {"question": "How many days are in a leap year?", "correct_answer": "366", "tier": "easy"},
    {"question": "What is the chemical symbol for water?", "correct_answer": "H2O", "tier": "easy"},
    {"question": "What is 15 plus 27?", "correct_answer": "42", "tier": "easy"},

    # ---------- MEDIUM TIER ----------
    {"question": "What was the first country in the world to grant women the national right to vote?", "correct_answer": "New Zealand", "tier": "medium"},
    {"question": "In what year did New Zealand grant women the national right to vote?", "correct_answer": "1893", "tier": "medium"},
    {"question": "Which planet in our solar system has the most moons?", "correct_answer": "Saturn", "tier": "medium"},
    {"question": "In which year did World War II end?", "correct_answer": "1945", "tier": "medium"},
    {"question": "What is the smallest country in the world by land area?", "correct_answer": "Vatican City", "tier": "medium"},
    {"question": "Who wrote the play Romeo and Juliet?", "correct_answer": "William Shakespeare", "tier": "medium"},
    {"question": "What is the speed of light in a vacuum, in kilometers per second, rounded to the nearest hundred thousand?", "correct_answer": "300000", "tier": "medium"},
    {"question": "What is the tallest mountain in the world, measured from sea level?", "correct_answer": "Mount Everest", "tier": "medium"},
    {"question": "In which city were the 2016 Summer Olympics held?", "correct_answer": "Rio de Janeiro", "tier": "medium"},
    {"question": "What is the currency used in Japan?", "correct_answer": "Yen", "tier": "medium"},

    # ---------- HARD TIER ----------
    {"question": "How many paintings did Vincent van Gogh sell during his lifetime?", "correct_answer": "DISPUTED - popularly believed to be one, but actually unclear/likely more than one", "tier": "trap"},
    {"question": "What specific discovery did Albert Einstein actually win his Nobel Prize for?", "correct_answer": "The photoelectric effect (NOT relativity)", "tier": "trap"},
    {"question": "Is the Great Wall of China visible from space with the naked eye?", "correct_answer": "No - this is a popular myth, confirmed false by NASA and multiple astronauts", "tier": "trap"},
    {"question": "Was Napoleon Bonaparte actually short for his time?", "correct_answer": "No - he was average height for a Frenchman of his era; the myth stems from a French/English inch mixup and British propaganda cartoons", "tier": "trap"},
    {"question": "Did Albert Einstein ever fail math in school?", "correct_answer": "No - this is a myth; he excelled at math, the confusion comes from failing non-math sections of a college entrance exam", "tier": "trap"},
    {"question": "Do goldfish really only have a 3-second memory?", "correct_answer": "No - this is a myth; goldfish can remember things for weeks or months", "tier": "trap"},
    {"question": "Do humans only use 10 percent of their brain?", "correct_answer": "No - this is a myth; humans use virtually all of their brain, just not every region simultaneously at all times", "tier": "trap"},
    {"question": "Did Vikings actually wear horned helmets?", "correct_answer": "No - this is a 19th-century invention from art and opera costuming; no archaeological evidence supports it", "tier": "trap"},
]

API_URL = "http://127.0.0.1:8000/ask"
JUDGE_URL = "http://127.0.0.1:8000/judge"
OUTPUT_CSV = "eval_results.csv"

def run_evaluation():
    results = []
    total_q = len(QUESTIONS)
    print(f"Starting evaluation on {total_q} internal questions...")
    
    for idx, item in enumerate(QUESTIONS):
        question = item['question']
        correct_answer = item['correct_answer']
        tier = item['tier']
        
        print(f"[{idx+1}/{total_q}] Processing ({tier}): {question[:45]}...")
        
        start_time = time.time()
        try:
            response = requests.post(API_URL, json={"question": question})
            elapsed = time.time() - start_time
            
            if response.status_code == 200:
                data = response.json()
                model_answer = data.get("final_answer") or ""
                stated_confidence = data.get("stated_confidence")
                
                avg_token_prob = data["confidence_summary"]["average_confidence"] if data["confidence_summary"] else 0.0
                flagged_pct = data["hallucination_risk"]["flagged_pct"]
                avg_entropy = data["hallucination_risk"]["avg_entropy"]
                
                is_correct = False
                if model_answer:
                    judge_payload = {
                        "question": question,
                        "correct_answer": correct_answer,
                        "model_answer": model_answer
                    }
                    try:
                        judge_res = requests.post(JUDGE_URL, json=judge_payload)
                        if judge_res.status_code == 200:
                            is_correct = judge_res.json().get("is_correct", False)
                    except Exception as judge_err:
                        print(f"Judge call failed, fallback to basic check: {judge_err}")
                        is_correct = correct_answer.lower() in model_answer.lower()

                results.append({
                    "question": question,
                    "tier": tier,
                    "correct_answer": correct_answer,
                    "model_answer": model_answer,
                    "is_correct_auto": is_correct,
                    "stated_confidence": stated_confidence if stated_confidence is not None else 100,
                    "actual_avg_confidence": avg_token_prob,
                    "flagged_pct": flagged_pct,
                    "avg_entropy": avg_entropy,
                    "time_seconds": round(elapsed, 2)
                })
            else:
                print(f"Error from API: {response.text}")
        except Exception as e:
            print(f"Failed to process question due to: {e}")
            
    out_df = pd.DataFrame(results)
    out_df.to_csv(OUTPUT_CSV, index=False)
    print(f"\nEvaluation complete! Labeled dataset saved to {OUTPUT_CSV}")

if __name__ == "__main__":
    run_evaluation()