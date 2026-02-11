import torch
import pytest
import numpy as np
from retinavit.training.robustness import generate_adversarial_examples
from retinavit.training.calibration import compute_ece

class MockRobustModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.head = torch.nn.Linear(3, 5)
    def forward(self, x):
        return {"logits": self.head(x.mean(dim=[-1, -2]))}

def test_pgd_generation():
    model = MockRobustModel()
    images = torch.randn(2, 3, 224, 224)
    labels = torch.tensor([0, 1])
    
    adv = generate_adversarial_examples(model, images, labels, eps=0.01)
    assert adv.shape == images.shape
    assert torch.all(adv >= 0) and torch.all(adv <= 1)
    
def test_ece_calculation():
    probs = np.array([[0.9, 0.1], [0.4, 0.6], [0.8, 0.2]])
    labels = np.array([0, 1, 1]) # Last one is wrong
    ece = compute_ece(probs, labels, n_bins=10)
    assert 0 <= ece <= 1

if __name__ == "__main__":
    import numpy as np
    test_pgd_generation()
    test_ece_calculation()
    print("Robustness and Calibration tests passed!")
