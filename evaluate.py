import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, roc_curve, auc
from train import model, test_loader  # Import trained model and DataLoader

# Load trained model
model.load_state_dict(torch.load("models/vit_model.pth"))
model.eval()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

y_true = []
y_scores = []

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        probabilities = torch.softmax(outputs, dim=1)[:, 1]  # Get probabilities for class 1
        
        y_true.extend(labels.cpu().numpy())
        y_scores.extend(probabilities.cpu().numpy())

# Compute Accuracy
y_pred = np.round(y_scores)  # Convert probabilities to binary predictions
accuracy = accuracy_score(y_true, y_pred)
print(f"Test Accuracy: {accuracy:.4f}")

# Compute ROC Curve & AUC
fpr, tpr, _ = roc_curve(y_true, y_scores)
roc_auc = auc(fpr, tpr)
print(f"AUC Score: {roc_auc:.4f}")

# Plot ROC Curve
plt.figure()
plt.plot(fpr, tpr, color="blue", lw=2, label=f"AUC = {roc_auc:.4f}")
plt.plot([0, 1], [0, 1], color="gray", linestyle="--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Receiver Operating Characteristic (ROC) Curve")
plt.legend(loc="lower right")
plt.savefig("roc_curve.png")
plt.savefig("roc_curve.png")  # Save the figure
plt.close()  # Close the plot to free memory

#..