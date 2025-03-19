import torch
import h5py
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, DataLoader

# Load the preprocessed dataset from HDF5
def load_preprocessed_data():
    """ Load preprocessed tensor data from HDF5 file """
    with h5py.File("data/preprocessed_data.h5", "r") as f:
        X_train = torch.tensor(f["X_train"][:], dtype=torch.float32) / 255.0  # Normalize
        X_test = torch.tensor(f["X_test"][:], dtype=torch.float32) / 255.0
        y_train = torch.tensor(f["y_train"][:], dtype=torch.long)
        y_test = torch.tensor(f["y_test"][:], dtype=torch.long)
    return X_train, X_test, y_train, y_test

# Load Data
X_train, X_test, y_train, y_test = load_preprocessed_data()
print(f"Train data shape: {X_train.shape}, Test data shape: {X_test.shape}")

# Create PyTorch DataLoaders
train_dataset = TensorDataset(X_train, y_train)
test_dataset = TensorDataset(X_test, y_test)

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)

# Visualize a few samples from the dataset
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.imshow(X_train[0].permute(1, 2, 0).numpy())  # Ensure correct channel format
plt.title("Sample Training Image")

plt.subplot(1, 2, 2)
plt.imshow(X_test[0].permute(1, 2, 0).numpy())
plt.title("Sample Test Image")

plt.savefig("data_visualization.png")
plt.show()

print("Data Visualization saved as 'data_visualization.png'")
