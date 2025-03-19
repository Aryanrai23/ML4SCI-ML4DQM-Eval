import numpy as np
import h5py
import torch
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import torchvision.transforms as transforms

# Define transformations for image preprocessing
transform = transforms.Compose([
    transforms.ToPILImage(),         # Convert NumPy to PIL image
    transforms.Resize((224, 224)),   # Resize to 224x224
    transforms.Grayscale(num_output_channels=3),  # Convert grayscale to 3 channels
    transforms.ToTensor()            # Convert to PyTorch tensor
])

# Load the dataset.
def load_data():
    """ Load the dataset from .npy files. """
    print("Loading dataset...")
    data_1 = np.load("/home/aryan/Desktop/ML4SCI-ML4DQM-Eval/data/Run355456_Dataset_jqkne.npy")
    data_2 = np.load("/home/aryan/Desktop/ML4SCI-ML4DQM-Eval/data/Run357479_Dataset_iodic.npy")
    
    labels_1 = np.zeros(len(data_1))
    labels_2 = np.ones(len(data_2))

    X = np.vstack([data_1, data_2])  # Stack along first axis
    y = np.hstack([labels_1, labels_2])  # Stack labels

    print(f"Dataset loaded: {X.shape[0]} samples, {X.shape[1:]} shape")
    return X, y

# Normalize the Features (Min-Max Scaling)
def normalize_data(X):
    """ Apply Min-Max normalization. """
    scaler = MinMaxScaler()
    X_flat = X.reshape(X.shape[0], -1)  # Flatten for scaler
    X_scaled = scaler.fit_transform(X_flat)
    return X_scaled.reshape(X.shape).astype(np.float32)  # Ensure float32

# Convert images in batches to avoid memory overload
def convert_to_tensor_batch(X, y, batch_size=5000):
    """ Process images in smaller batches to avoid memory overflow """
    X_batches = []
    y_batches = []

    for i in range(0, len(X), batch_size):
        print(f"Processing batch {i // batch_size + 1}/{len(X) // batch_size}")
        batch_X = X[i:i + batch_size]
        batch_y = y[i:i + batch_size]

        X_batch_tensors = [transform(np.uint8(img)) for img in batch_X]
        X_batches.append(torch.stack(X_batch_tensors))
        y_batches.append(torch.tensor(batch_y, dtype=torch.long))

    return torch.cat(X_batches), torch.cat(y_batches)

# Train-Test Split
def split_data(X, y):
    """ Split into 80% training and 20% testing. """
    return train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Save Data in HDF5 Format as PyTorch Tensors
def save_data(X_train, X_test, y_train, y_test):
    """ Save preprocessed tensor data in HDF5 format with chunking """
    with h5py.File("data/preprocessed_data.h5", "w") as f:
        f.create_dataset("X_train", data=X_train.numpy(), chunks=True)  # Enable chunking
        f.create_dataset("X_test", data=X_test.numpy(), chunks=True)
        f.create_dataset("y_train", data=y_train.numpy(), chunks=True)
        f.create_dataset("y_test", data=y_test.numpy(), chunks=True)

# Run all preprocessing steps
def preprocess_and_save():
    print("Loading dataset...")
    X, y = load_data()

    print("Normalizing data...")
    X = normalize_data(X)  # Ensure correct float32 dtype

    print("Splitting into train and test sets...")
    X_train, X_test, y_train, y_test = split_data(X, y)

    print("Converting to tensors in batches...")
    X_train, y_train = convert_to_tensor_batch(X_train, y_train)
    X_test, y_test = convert_to_tensor_batch(X_test, y_test)

    print("Saving preprocessed tensor data...")
    save_data(X_train, X_test, y_train, y_test)

    print("Preprocessing complete!")

# Run the preprocessing script
if __name__ == "__main__":
    preprocess_and_save()
#..
