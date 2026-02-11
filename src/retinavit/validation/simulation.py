import time
import numpy as np
from typing import Dict, List, Any

class ClinicalWorkflowSimulator:
    """
    Simulates a hospital workflow to compare AI-assisted vs. manual screening.
    """
    def __init__(self, avg_manual_time_per_case: float = 300.0): # seconds
        self.avg_manual_time = avg_manual_time_per_case
        self.results = []

    def simulate_case(self, model_latency: float, quality_label: str) -> Dict[str, float]:
        """
        Simulates a single case processing time.
        AI-assisted time = pre-processing + AI inference + Clinician review (usually faster if AI is confident).
        """
        # Base AI time
        ai_time = model_latency
        
        # Clinician review time adjustment
        # If quality is poor, clinician takes much longer. 
        # If AI confidence is high and quality is good, clinician is faster.
        if quality_label == "Good":
            clinician_time = self.avg_manual_time * 0.4 # 60% faster
        elif quality_label == "Fair":
            clinician_time = self.avg_manual_time * 0.7 # 30% faster
        else: # Poor
            clinician_time = self.avg_manual_time * 1.2 # 20% slower due to re-acquisition need
            
        total_ai_assisted = ai_time + clinician_time
        
        return {
            "manual_time": self.avg_manual_time,
            "ai_assisted_time": total_ai_assisted,
            "time_saved": self.avg_manual_time - total_ai_assisted
        }

    def run_simulation_batch(self, n_cases: int, model_latency: float):
        case_qualities = np.random.choice(["Good", "Fair", "Poor"], n_cases, p=[0.7, 0.2, 0.1])
        batch_results = []
        for q in case_qualities:
            res = self.simulate_case(model_latency, q)
            batch_results.append(res)
            
        total_time_manual = sum(c["manual_time"] for c in batch_results)
        total_time_ai = sum(c["ai_assisted_time"] for c in batch_results)
        
        return {
            "total_cases": n_cases,
            "avg_time_saved_pct": (1 - total_time_ai / total_time_manual) * 100,
            "total_hours_saved": (total_time_manual - total_time_ai) / 3600
        }

if __name__ == "__main__":
    sim = ClinicalWorkflowSimulator()
    print("Running 1000-case clinical simulation...")
    results = sim.run_simulation_batch(1000, model_latency=0.5)
    print(f"Simulation Complete:")
    print(f"- Average Time Saved: {results['avg_time_saved_pct']:.2f}%")
    print(f"- Total Clinician Hours Saved per 1000 cases: {results['total_hours_saved']:.2f} hrs")
