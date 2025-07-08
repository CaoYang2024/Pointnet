import random
import os
import torch
from path import Path
import wandb
import hydra
from omegaconf import DictConfig
from src.models import model
from src.pipelines import dataset
from src.utils import utils
from src.pipelines.args import parse_args

from torchvision import transforms
from torch.utils.data import DataLoader

random.seed = 42


def pointnetloss(outputs, labels, m3x3, m64x64, alpha=0.0001):
    criterion = torch.nn.NLLLoss()
    bs = outputs.size(0)
    id3x3 = torch.eye(3, requires_grad=True).repeat(bs, 1, 1)
    id64x64 = torch.eye(64, requires_grad=True).repeat(bs, 1, 1)
    if outputs.is_cuda:
        id3x3 = id3x3.cuda()
        id64x64 = id64x64.cuda()
    diff3x3 = id3x3 - torch.bmm(m3x3, m3x3.transpose(1, 2))
    diff64x64 = id64x64 - torch.bmm(m64x64, m64x64.transpose(1, 2))
    return criterion(outputs, labels) + alpha * (torch.norm(diff3x3) + torch.norm(diff64x64)) / float(bs)


def train(cfg):
    path = Path(cfg.root_dir)

    folders = [dir for dir in sorted(os.listdir(path)) if os.path.isdir(path / dir)]
    classes = {folder: i for i, folder in enumerate(folders)}

    train_transforms = transforms.Compose([
        utils.PointSampler(1024),
        utils.Normalize(),
        utils.RandRotation_z(),
        utils.RandomNoise(),
        utils.ToTensor()
    ])
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(device)
    pointnet = model.PointNet()
    pointnet.to(device)
    optimizer = torch.optim.Adam(pointnet.parameters(), lr=cfg.lr)

    train_ds = dataset.PointCloudData(path, transform=train_transforms)
    valid_ds = dataset.PointCloudData(path, valid=True, folder='test', transform=train_transforms)
    print('Train dataset size: ', len(train_ds))
    print('Valid dataset size: ', len(valid_ds))
    print('Number of classes: ', len(train_ds.classes))

    train_loader = DataLoader(dataset=train_ds, batch_size=cfg.batch_size, shuffle=True)
    valid_loader = DataLoader(dataset=valid_ds, batch_size=cfg.batch_size * 2)

    try:
        os.mkdir(cfg.save_model_path)
    except OSError:
        pass


    wandb.init(
        project="pointnet-project",
        config={
            "batch_size": cfg.batch_size,
            "lr": cfg.lr,
            "epochs": cfg.epochs
        },
        name=f"run_{wandb.util.generate_id()}"
    )

    print('Start training')
    for epoch in range(cfg.epochs):
        pointnet.train()
        running_loss = 0.0
        for i, data in enumerate(train_loader, 0):
            inputs, labels = data['pointcloud'].to(device).float(), data['category'].to(device)
            optimizer.zero_grad()
            outputs, m3x3, m64x64 = pointnet(inputs.transpose(1, 2))

            loss = pointnetloss(outputs, labels, m3x3, m64x64)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if i % 10 == 9:
                avg_loss = running_loss / 10
                print(f'[Epoch: {epoch + 1}, Batch: {i + 1}/{len(train_loader)}], loss: {avg_loss:.3f}')
                wandb.log({"train_loss": avg_loss, "epoch": epoch})
                running_loss = 0.0

        pointnet.eval()
        correct = total = 0
        with torch.no_grad():
            for data in valid_loader:
                inputs, labels = data['pointcloud'].to(device).float(), data['category'].to(device)
                outputs, __, __ = pointnet(inputs.transpose(1, 2))
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        val_acc = 100. * correct / total
        print(f'Valid accuracy: {val_acc:.2f} %')
        wandb.log({"val_acc": val_acc, "epoch": epoch})

        checkpoint = Path(cfg.save_model_path) / f'save_{epoch}.pth'
        torch.save(pointnet.state_dict(), checkpoint)
        print('Model saved to ', checkpoint)

    wandb.finish()


@hydra.main(config_path="config", config_name="config", version_base="1.3")
def main(cfg: DictConfig):
    train(cfg)

if __name__ == "__main__":
    main()