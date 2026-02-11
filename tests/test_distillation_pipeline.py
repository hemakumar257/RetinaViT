import torch
import torch.nn as nn
from retinavit.models.mobile_student import MobileRetinaNet
from retinavit.training.train_distill import distillation_loss

def test_distillation_loss():
    from retinavit.training.train_distill import distillation_loss
    student_logits = torch.randn(2, 5)
    teacher_logits = torch.randn(2, 5)
    labels = torch.tensor([0, 1])
    
    loss = distillation_loss(student_logits, teacher_logits, labels)
    assert loss.item() > 0

def test_mobile_student_forward():
    cfg = {"model": {"num_classes": 5}}
    student = MobileRetinaNet(cfg)
    x = torch.randn(1, 3, 224, 224)
    out = student(x)
    assert out.shape == (1, 5)

if __name__ == "__main__":
    test_distillation_loss()
    test_mobile_student_forward()
    print("Distillation pipeline tests passed!")
