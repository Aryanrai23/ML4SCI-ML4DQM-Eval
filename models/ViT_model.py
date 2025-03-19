import torch.nn as nn
from transformers import ViTForImageClassification

class CustomViT(nn.Module):
    def __init__(self, num_labels=2):
        super(CustomViT, self).__init__()
        self.conv1 = nn.Conv2d(1, 3, kernel_size=1)  # Convert 1-channel to 3-channel
        self.vit = ViTForImageClassification.from_pretrained(
            "google/vit-base-patch16-224-in21k",
            num_labels=num_labels
        )

    def forward(self, x):
        x = self.conv1(x)  # Convert grayscale to RGB
        return self.vit(x).logits
