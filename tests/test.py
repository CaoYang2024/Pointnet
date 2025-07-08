import sys
from pathlib import Path

# Add the root of the project to PYTHONPATH
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

import argparse
import torch
from src.models import model
from src.pipelines import dataset
from src.utils import utils
from torchvision import transforms
from torch.utils.data import DataLoader

def test(model_path, root_dir, batch_size=32):
    test_transforms = transforms.Compose([
        utils.PointSampler(1024),
        utils.Normalize(),
        utils.ToTensor()
    ])

    test_ds = dataset.PointCloudData(root_dir, valid=True, folder="test", transform=test_transforms)
    test_loader = DataLoader(dataset=test_ds, batch_size=batch_size)

    print(f"Number of test samples: {len(test_ds)}, number of classes: {len(test_ds.classes)}")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    pointnet = model.PointNet()
    pointnet.load_state_dict(torch.load(model_path, map_location=device))
    pointnet.to(device)
    pointnet.eval()

    correct = total = 0
    with torch.no_grad():
        for data in test_loader:
            inputs = data["pointcloud"].to(device).float()
            labels = data["category"].to(device)
            outputs, _, _ = pointnet(inputs.transpose(1, 2))
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100.0 * correct / total
    print(f"✅ Test accuracy: {accuracy:.2f}%")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test PointNet Model")
    parser.add_argument("--model_path", type=str, required=True, help="Path to saved model .pth file")
    parser.add_argument("--root_dir", type=str, required=True, help="Path to dataset root")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for testing")

    args = parser.parse_args()

    test(
        model_path=args.model_path,
        root_dir=args.root_dir,
        batch_size=args.batch_size
    )
