import numpy as np
import h5py
import torch
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import torchvision.transforms as transforms

# Load the dataset
def load_data():
    """ Load the dataset from .npy files. """
    print("Loading dataset...")
    data_1 = np.load("/home/aryan/Desktop/ML4SCI-ML4DQM-Eval/data/Run355456_Dataset_jqkne.npy")
    data_2 = np.load("/home/aryan/Desktop/ML4SCI-ML4DQM-Eval/data/Run357479_Dataset_iodic.npy")
    
    labels_1 = np.zeros(len(data_1))
    labels_2 = np.ones(len(data_2))

    X = np.vstack([data_1, data_2])
    y = np.hstack([labels_1, labels_2])

    print(f"Dataset loaded: {X.shape[0]} samples, {X.shape[1:]} shape")
    return X, y

# Normalize Features (Min-Max Scaling)
def normalize_data(X):
    """ Apply Min-Max normalization. """
    scaler = MinMaxScaler()
    X_flat = X.reshape(X.shape[0], -1)
    X_scaled = scaler.fit_transform(X_flat)
    return X_scaled.reshape(X.shape)

# Convert NumPy arrays to PyTorch tensors
def convert_to_tensor(X, y):
    """ Convert NumPy arrays to PyTorch tensors """
    X_tensor = torch.tensor(X, dtype=torch.float32).unsqueeze(1)  # Adding channel dimension
    y_tensor = torch.tensor(y, dtype=torch.long)
    return X_tensor, y_tensor

# Train-Test Split
def split_data(X, y):
    """ Split into 80% training and 20% testing. """
    return train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Save Data in HDF5 Format as PyTorch Tensors
def save_data(X_train, X_test, y_train, y_test):
    """ Save preprocessed tensor data in HDF5 format. """
    with h5py.File("data/preprocessed_data.h5", "w") as f:
        f.create_dataset("X_train", data=X_train.numpy())
        f.create_dataset("X_test", data=X_test.numpy())
        f.create_dataset("y_train", data=y_train.numpy())
        f.create_dataset("y_test", data=y_test.numpy())

# Run all preprocessing steps
def preprocess_and_save():
    print("Loading dataset...")
    X, y = load_data()

    print("Normalizing data...")
    X = normalize_data(X)

    print("Splitting into train and test sets...")
    X_train, X_test, y_train, y_test = split_data(X, y)

    print("Converting to tensors...")
    X_train, y_train = convert_to_tensor(X_train, y_train)
    X_test, y_test = convert_to_tensor(X_test, y_test)

    print("Saving preprocessed tensor data...")
    save_data(X_train, X_test, y_train, y_test)

    print("Preprocessing complete!")

# Run the preprocessing script
if __name__ == "__main__":
    preprocess_and_save()
