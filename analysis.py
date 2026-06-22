import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score

def compute_ece(accuracies, confidences, num_bins=5):
   
    bin_boundaries = np.linspace(0, 1, num_bins + 1)
    ece = 0.0
    n_samples = len(accuracies)
    
    for i in range(num_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]  
        
        in_bin = (confidences >= bin_lower) & (confidences < bin_upper)
        if i == num_bins - 1:
            in_bin = in_bin | (confidences == bin_upper)
            
        prop_in_bin = np.mean(in_bin)
        
        if prop_in_bin > 0:
            accuracy_in_bin = np.mean(accuracies[in_bin])
            avg_confidence_in_bin = np.mean(confidences[in_bin])
            ece += prop_in_bin * np.abs(avg_confidence_in_bin - accuracy_in_bin)
            
    return round(ece * 100, 2) 

def run_paper_analysis():
    df = pd.read_csv("eval_results.csv")
    
    y_true_correct = df['is_correct_auto'].astype(int).values
    y_true_error = (df['is_correct_auto'] == False).astype(int).values
    
    print("=== 1. THREE-WAY PREDICTOR COMPARISON (AUROC) ===")
    print("Evaluating which internal metric best correlates with model generation errors:")
    
    try:
        auroc_entropy = roc_auc_score(y_true_error, df['avg_entropy'].values)
        print(f" -> Token Entropy Metric AUROC (Predicting Error):       {auroc_entropy:.3f}")
    except Exception:
        print(" -> Token Entropy Metric AUROC: Variance too low to calculate.")
        
    try:
        auroc_flagged = roc_auc_score(y_true_error, df['flagged_pct'].values)
        print(f" -> High-Risk Passages Flagged % AUROC (Predicting Error): {auroc_flagged:.3f}")
    except Exception:
        print(" -> Flagged % AUROC: Variance too low.")
        
    try:
        auroc_stated = roc_auc_score(y_true_correct, (df['stated_confidence'] / 100.0).values)
        print(f" -> Stated Confidence Metric AUROC (Predicting Accuracy): {auroc_stated:.3f}")
    except Exception:
        print(" -> Stated Confidence AUROC: Variance too low.")
        
    print("\n=== 2. CONFIDENCE CALIBRATION ERROR (ECE) ===")
    
    df['stated_conf_norm'] = df['stated_confidence'] / 100.0
    df['avg_token_conf_norm'] = df['actual_avg_confidence'] / 100.0
    
    ece_stated = compute_ece(y_true_correct, df['stated_conf_norm'].values, num_bins=4)
    ece_tokens = compute_ece(y_true_correct, df['avg_token_conf_norm'].values, num_bins=4)
    
    print(f" -> Expected Calibration Error (ECE) on Stated Confidence:    {ece_stated}%")
    print(f" -> Expected Calibration Error (ECE) on Avg Token Probability: {ece_tokens}%")
    
    print("\n=== 3. TIER PERFORMANCE BREAKDOWN ===")
    tier_summary = df.groupby('tier').agg(
        total_questions=('question', 'count'),
        evaluated_accuracy=('is_correct_auto', 'mean'),
        mean_stated_confidence=('stated_confidence', 'mean'),
        mean_entropy_score=('avg_entropy', 'mean')
    ).round(3)
    print(tier_summary)

if __name__ == "__main__":
    run_paper_analysis()