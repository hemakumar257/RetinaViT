"""Simple Trainer placeholder."""

class Trainer:
    def __init__(self, model, optimizer=None, device="cpu"):
        self.model = model
        self.optimizer = optimizer
        self.device = device

    def fit(self, dataloader, epochs: int = 1):
        """Placeholder training loop."""
        for epoch in range(epochs):
            for batch in dataloader:
                pass
