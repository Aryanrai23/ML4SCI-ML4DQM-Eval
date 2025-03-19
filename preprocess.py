
# Handle Zero-Padded Entries
def mask_zero_entries(X):
    """ Replace zero-valued entries with NaN to avoid model bias. """
    return np.where(X == 0, np.nan, X)
 


#normalize Features (Min-Max Scaling)
from sklearn.preprocessing import MinMaxScaler

def normalize_data(X):
    """ Apply Min-Max normalization. """
    scaler = MinMaxScaler()
    X_flat = X.reshape(X.shape[0], -1)
    X_scaled = scaler.fit_transform(X_flat)
    return X_scaled.reshape(X.shape)



#Data Augmentation (Rotation, Flip, Crop)
import torchvision.transforms as transforms

data_transforms = transforms.Compose([
    transforms.RandomRotation(10),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomResizedCrop(size=(64, 72), scale=(0.8, 1.0))
])



#Handle Class Imbalance (SMOTE)
from imblearn.over_sampling import SMOTE

def balance_data(X, y):
    """ Handle class imbalance using SMOTE. """
    X_flat = X.reshape(X.shape[0], -1)
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X_flat, y)
    return X_resampled.reshape(-1, 64, 72), y_resampled


from sklearn.model_selection import train_test_split

def split_data(X, y):
    """ Split into 80% training and 20% testing. """
    return train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)



import h5py

def save_data(X_train, X_test, y_train, y_test):
    """ Save preprocessed data in HDF5 format. """
    with h5py.File("data/preprocessed_data.h5", "w") as f:
        f.create_dataset("X_train", data=X_train)
        f.create_dataset("X_test", data=X_test)
        f.create_dataset("y_train", data=y_train)
        f.create_dataset("y_test", data=y_test)
