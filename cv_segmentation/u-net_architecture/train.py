import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
from model import NET

# from utils import (
#     load_checkpoint,
#     save_checkpoint,
#     get_loaders,
#     check_accuracy,
#     save_prediction_as_imgs
# )

# from utils import *

# initialising hyperparameters
learning_rate = 1e-4
device = "cuda" if torch.cuda.is_available() else "cpu"
batch_size = 32
num_epochs = 100
num_workers = 2
image_height = 160 # 1280 originally
image_width = 240 # 1918 original;y
pin_memory = True
load_model = True
train_img_dir = "/data/train_images/"
train_mask_dir = "data/train_masks/"
val_img_dir = "data/val_images/"
val_mask_dir = "data/val_masks/"


def train_fn(loader, model, optimizer, loss_fn, scaler):
    pass
    loop = tqdm(loader)

    for batch_idx, (data,targets) in enumerate(loop):
        data = data.to(device=device)
        target = targets.float()
        targets = targets.float().unsqueeze(1).to(device=device)

        # forward
        with torch.cuda.amp.autocast():
            predictions = model(data)
            loss = loss_fn(predictions, targets)

        # backward
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # update tqdm loss
        loop.set_postfix(loss=loss.item())



def main():
    train_transform = A.compose(
        [
            A.resize(height=image_height, width=image_width),
            A.rotate(limit=35, p=1.0),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.1),
            A.normalize(
                mean=[0.0, 0.0, 0.0],
                std=[1.0, 1.0, 1.0],
                max_pixel_value=255.0,
            ),
            ToTensorV2(),
        ],
    )

    val_transform = A.Compose(
        [
            A.resize(height=image_height, width=image_width),
            A.Normalize(
                mean=[0.0, 0.0, 0.0, 0.0],

            ),
            ToTensorV2(),
        ],
    )

    model = NET(in_channels=3, out_channels=1).to(device)
    loss_fn = nn.BCEWithLogitsLoss()





if __name__ == "__main__":
    main()


