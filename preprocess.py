import numpy as np
import h5py
from main import load_data  # Import dataset loading function from main.py
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
import torchvision.transforms as transforms

# Handle Zero-Padded Entries
def mask_zero_entries(X):
    """ Replace zero-valued entries with NaN to avoid model bias. """
    return np.where(X == 0, np.nan, X)

# Normalize Features (Min-Max Scaling)
def normalize_data(X):
    """ Apply Min-Max normalization. """
    scaler = MinMaxScaler()
    X_flat = X.reshape(X.shape[0], -1)
    X_scaled = scaler.fit_transform(X_flat)
    return X_scaled.reshape(X.shape)

# Data Augmentation (Rotation, Flip, Crop)
data_transforms = transforms.Compose([
    transforms.RandomRotation(10),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomResizedCrop(size=(64, 72), scale=(0.8, 1.0))
])

# Handle Class Imbalance (SMOTE)
def balance_data(X, y):
    """ Handle class imbalance using SMOTE. """
    X_flat = X.reshape(X.shape[0], -1)
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X_flat, y)
    return X_resampled.reshape(-1, 64, 72), y_resampled

# Train-Test Split
def split_data(X, y):
    """ Split into 80% training and 20% testing. """
    return train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Save Data in HDF5 Format
def save_data(X_train, X_test, y_train, y_test):
    """ Save preprocessed data in HDF5 format. """
    with h5py.File("data/preprocessed_data.h5", "w") as f:
        f.create_dataset("X_train", data=X_train)
        f.create_dataset("X_test", data=X_test)
        f.create_dataset("y_train", data=y_train)
        f.create_dataset("y_test", data=y_test)

# Run all preprocessing steps
def preprocess_and_save():
    print("Loading dataset from main.py...")
    X, y = load_data()  # Use the dataset loading function from main.py

    print("Handling zero-padded entries...")
    X = mask_zero_entries(X)

    print("Normalizing data...")
    X = normalize_data(X)

    print("Balancing data using SMOTE...")
    X, y = balance_data(X, y)

    print("Splitting into train and test sets...")
    X_train, X_test, y_train, y_test = split_data(X, y)

    print("Saving preprocessed data...")
    save_data(X_train, X_test, y_train, y_test)

    print("Preprocessing complete!")

# Run the preprocessing script
if __name__ == "__main__":
    preprocess_and_save()
