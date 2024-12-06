import os
import cv2 as cv
import numpy as np
import torch
from torch.utils.data import Dataset

class Data(Dataset):
    def __init__(self, base_folder_aug, base_folder_orig):
        self.dataset_aug = []
        self.dataset_train = []
        self.dataset_test = []
        self.dataset_valid = []
        onehot = np.eye(6)  

        
        self._load_data(base_folder_aug, self.dataset_aug, is_augmented=True)

        
        self._load_data(base_folder_orig, self.dataset_train, data_type="Train")
        self._load_data(base_folder_orig, self.dataset_test, data_type="Test")
        self._load_data(base_folder_orig, self.dataset_valid, data_type="Valid")

        print(f"Augmented Images (Train): {len(self.dataset_aug)}")
        print(f"Original Images (Train): {len(self.dataset_train)}")
        print(f"Original Images (Test): {len(self.dataset_test)}")
        print(f"Original Images (Valid): {len(self.dataset_valid)}")

    def _load_data(self, base_folder, dataset, data_type="Train", is_augmented=False):
        onehot = np.eye(6)  

        for fold_num in range(1, 6):
            if is_augmented:
                folder = os.path.join(base_folder, f"fold{fold_num}_AUG/Train/")
            else:
                folder = os.path.join(base_folder, f"fold{fold_num}/{data_type}/")
            
            if not os.path.exists(folder):
                continue

            for class_idx, class_name in enumerate(os.listdir(folder)):
                class_folder = os.path.join(folder, class_name)
                for img_name in os.listdir(class_folder):
                    img_path = os.path.join(class_folder, img_name)
                    image = cv.resize(cv.imread(img_path), (32, 32)) / 255
                    dataset.append([image, onehot[class_idx]])

    def __len__(self):
        return len(self.dataset_aug)

    def __getitem__(self, idx):
        features, label = self.dataset_aug[idx]
        return (
            torch.tensor(features, dtype=torch.float32).permute(2, 0, 1),  # Convert to CHW format
            torch.tensor(label, dtype=torch.float32)
        )


if __name__ == "__main__":
    aug_path = "././Dataset/Augmented Images/Augmented Images/FOLDS_AUG/"
    orig_path = "././Dataset/Original Images/Original Images/FOLDS/"
    
    data = Data(base_folder_aug=aug_path, base_folder_orig=orig_path)
