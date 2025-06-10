import os
import cv2
import torch
from torch.utils.data import Dataset
import numpy as np

class ImageDataset(Dataset):
    def __init__(self, path, transform=None):
        """
        OpenCV를 사용하여 이미지를 로드하는 PyTorch Dataset 클래스.

        Parameters:
            path (str): 이미지가 저장된 루트 디렉토리.
            transform (callable, optional): 데이터 변환 함수.
        """
        self.image_paths = []
        self.labels = []
        self.transform = transform

        class_folders = [d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]
        class_folders.sort()  # 클래스 정렬 (일관된 라벨링을 위해)

        for class_idx, class_name in enumerate(class_folders):
            class_path = os.path.join(path, class_name)
            img_files = os.listdir(class_path)

            for img_file in img_files:
                img_path = os.path.join(class_path, img_file)
                if img_path.lower().endswith(('jpg', 'png')):  # 이미지 파일만 추가
                    self.image_paths.append(img_path)
                    self.labels.append(class_idx)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        try:
            img_path = self.image_paths[idx]
            image = cv2.imread(img_path)  # BGR 형식으로 로드
            if image is None:
                raise ValueError(f"Image at {img_path} could not be loaded.")

            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # RGB 변환
            image = image.astype(np.float32) / 255.0  # 0~1 스케일링
            image = torch.from_numpy(image).permute(2, 0, 1)  # (H, W, C) → (C, H, W) 변환

            if self.transform:
                image = self.transform(image)

            return {"images": image, "class_labels": torch.tensor(self.labels[idx], dtype=torch.long)}
        except IndexError:
            print(f"Index {idx} is out of range for dataset of size {len(self.image_paths)}")
            raise
