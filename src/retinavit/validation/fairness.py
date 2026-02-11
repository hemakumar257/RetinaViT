import torch
import numpy as np
import pandas as pd
from typing import Dict, List, Any
from sklearn.metrics import accuracy_score, confusion_matrix

class FairnessAuditor:
    """
    Analyzes model performance across different subgroups to identify bias.
    """
    def __init__(self, results_df: pd.DataFrame):
        """
        results_df should contain: ['y_true', 'y_pred', 'subgroup']
        """
        self.df = results_df

    def compute_subgroup_metrics(self) -> pd.DataFrame:
        subgroups = self.df['subgroup'].unique()
        metrics = []
        
        for group in subgroups:
            group_data = self.df[self.df['subgroup'] == group]
            acc = accuracy_score(group_data['y_true'], group_data['y_pred'])
            
            # Simplified Equalized Odds (TPR/FPR comparison)
            # For simplicity in this demo, we focus on Accuracy and Sensitivity
            cm = confusion_matrix(group_data['y_true'], group_data['y_pred'], labels=[0, 1, 2, 3, 4])
            
            # Macro-averaged sensitivity (Recall)
            recalls = []
            for i in range(5):
                if cm[i, :].sum() > 0:
                    recalls.append(cm[i, i] / cm[i, :].sum())
            
            metrics.append({
                "subgroup": group,
                "count": len(group_data),
                "accuracy": acc,
                "macro_recall": np.mean(recalls) if recalls else 0.0
            })
            
        return pd.DataFrame(metrics)

    def assess_disparity(self, threshold: float = 0.8) -> Dict[str, Any]:
        """
        Checks if any subgroup performance is below threshold relative to the best group.
        (80% rule/Impact Ratio)
        """
        metrics = self.compute_subgroup_metrics()
        max_acc = metrics['accuracy'].max()
        metrics['disparity_ratio'] = metrics['accuracy'] / max_acc
        
        flagged_groups = metrics[metrics['disparity_ratio'] < threshold]['subgroup'].tolist()
        
        return {
            "metrics": metrics.to_dict(orient="records"),
            "is_fair": len(flagged_groups) == 0,
            "flagged_groups": flagged_groups
        }

if __name__ == "__main__":
    # Mock data: Accuracy drops for "Poor Quality" group
    data = {
        "y_true": np.random.randint(0, 5, 100),
        "y_pred": np.random.randint(0, 5, 100),
        "subgroup": ["Good"] * 70 + ["Poor"] * 30
    }
    # Induce better performance for "Good"
    df = pd.DataFrame(data)
    df.loc[df['subgroup'] == 'Good', 'y_pred'] = df.loc[df['subgroup'] == 'Good', 'y_true']
    
    auditor = FairnessAuditor(df)
    results = auditor.assess_disparity()
    print("Fairness Audit Results:")
    for group in results['metrics']:
        print(f"Group {group['subgroup']}: Acc={group['accuracy']:.2f}, Disparity={group['disparity_ratio']:.2f}")
    
    if not results['is_fair']:
        print(f"WARNING: Potential bias detected in groups: {results['flagged_groups']}")
