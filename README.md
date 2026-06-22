# LLM Reasoning Debugger

A local web app that looks past what a language model *says* about its own confidence, and shows what's actually happening inside it.

Most LLMs will tell you they're 100% confident in an answer whether they're right or completely wrong. This tool pulls real, internal signals straight out of the model itself, token-by-token probability, entropy, attention, and uses them to show where an answer is genuinely solid versus where it's shaky or possibly hallucinated.

Everything runs locally on your own GPU. No cloud, no API keys, no cost per query.

## What it does

- **Ask any question** through a clean local web interface and get a full reasoned answer.
- **Token-by-token confidence** — see exactly how sure the model was about each word it generated, pulled from its real internal probabilities, not a self-reported number.
- **Entropy-based uncertainty** — a more rigorous signal than simple top-choice probability, captures how genuinely "torn" the model was between multiple options at each step.
- **Hallucination risk detection** — automatically flags sustained passages of high uncertainty, the pattern associated with fabricated or ungrounded content, and shows you exactly which words triggered the flag.
- **Attention visualization** — for any word the model generated, see which earlier word (in your question, or its own prior reasoning) it was actually focused on.
- **Stated vs. actual confidence comparison** — the core insight of the project: a side-by-side of what the model claims about itself versus what its internals actually show.

## Results

Tested across 28 hand-verified questions spanning easy facts, medium-difficulty trivia, and deliberately tricky "myth" questions (e.g. "is the Great Wall of China visible from space?").

| Tier   | Accuracy | Mean Stated Confidence | Mean Entropy |
|--------|----------|------------------------|--------------|
| Easy   | 90%      | 100%                   | 0.57         |
| Medium | 60%      | 100%                   | 0.59         |
| Trap   | 50%      | 100%                   | 1.34         |

Accuracy drops sharply with difficulty, exactly as expected. Stated confidence stays flat at 100% regardless. Entropy rises in step with difficulty, a real internal signal the model's own words don't reveal.

**Expected Calibration Error (ECE):**
- Stated confidence: **32.14%** (poorly calibrated — confidence doesn't track correctness)
- Actual average token probability: **15.16%** (meaningfully better calibrated)

The model's self-reported confidence is essentially uninformative. Its real internal probabilities carry genuine signal about whether it's likely to be right.

## Project structure

- `backend.py` — FastAPI server that loads the model, runs generation with token-level entropy/attention extraction, and computes hallucination risk. Also exposes a `/judge` endpoint for semantic answer grading.
- `index.html` — the frontend. Type a question, see the full breakdown rendered live.
- `eval_runner.py` — runs the full 28-question benchmark against the backend and saves results to CSV/JSON.
- `run_analysis.py` — computes AUROC and Expected Calibration Error from the eval results.

## Running it locally

1. Install dependencies:
   ```
   pip install fastapi uvicorn torch transformers requests pandas scikit-learn
   ```
2. Start the backend:
   ```
   python -m uvicorn backend:app
   ```
3. Open `index.html` in your browser.
4. (Optional) Run the evaluation suite:
   ```
   python eval_runner.py
   python run_analysis.py
   ```

Requires an NVIDIA GPU with CUDA support. Tested on an RTX 4070 with 8GB VRAM running Qwen2.5-1.5B-Instruct.

## Future directions

- Scale the evaluation set beyond 28 questions for more statistically robust results
- Multi-model comparison (run the same question through different models side-by-side)
- Explore this as the basis for a research paper on confidence calibration in small language models

## Why this matters

Confidently wrong AI output is a real, active problem in deploying LLMs safely. This project demonstrates, with real measured data, that a model's own claimed confidence can't be trusted, but the signals already present inside the model can tell a more honest story, if you know how to look.
