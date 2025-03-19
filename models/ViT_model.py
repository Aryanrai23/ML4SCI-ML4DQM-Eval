import torch.nn as nn
from transformers import ViTForImageClassification

class CustomViT(nn.Module):
    def __init__(self, num_labels=2):
        super(CustomViT, self).__init__()
        self.vit = ViTForImageClassification.from_pretrained(
            "google/vit-base-patch16-224-in21k",
            num_labels=num_labels
        )

    def forward(self, x):
        return self.vit(x).logits

model = CustomViT(num_labels=2)  # Define model instance
