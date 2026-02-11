"""Minimal Vision Transformer placeholder.

This is a non-functional placeholder class intended to be replaced by a real
implementation (e.g., using PyTorch or Flax) in Phase 2/3.
"""

class VisionTransformer:
    def __init__(self, image_size=224, patch_size=16, num_classes=1000, **kwargs):
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_classes = num_classes

    def forward(self, x):
        raise NotImplementedError("Integrate with PyTorch/Flax in later phases")
