import os
from pathlib import Path
from src.utils import utils
from torchvision import transforms
from torch.utils.data import Dataset


def default_transforms():
    return transforms.Compose([
        utils.PointSampler(1024),
        utils.Normalize(),
        utils.ToTensor()
    ])


class PointCloudData(Dataset):
    def __init__(self, root_dir, valid=False, folder="train", transform=default_transforms()):
        self.root_dir = Path(root_dir)  # 确保路径类型正确
        # 只保留非隐藏文件夹作为类别（避免 .dvc、.git 被识别为类别）
        folders = [
            d for d in sorted(os.listdir(self.root_dir))
            if (self.root_dir / d).is_dir() and not d.startswith(".")
        ]
        self.classes = {folder: i for i, folder in enumerate(folders)}
        self.transforms = transform if not valid else default_transforms()
        self.valid = valid
        self.files = []

        for category in self.classes.keys():
            new_dir = self.root_dir / category / folder
            if not new_dir.exists():
                print(f"[警告] 路径不存在: {new_dir}")
                continue

            for file in os.listdir(new_dir):
                if file.endswith('.off'):
                    sample = {
                        'pcd_path': new_dir / file,
                        'category': category
                    }
                    self.files.append(sample)

    def __len__(self):
        return len(self.files)

    def __preproc__(self, file):
        verts, faces = utils.read_off(file)
        if self.transforms:
            pointcloud = self.transforms((verts, faces))
        return pointcloud

    def __getitem__(self, idx):
        pcd_path = self.files[idx]['pcd_path']
        category = self.files[idx]['category']
        with open(pcd_path, 'r') as f:
            pointcloud = self.__preproc__(f)
        return {
            'pointcloud': pointcloud,
            'category': int(self.classes[category])
        }

