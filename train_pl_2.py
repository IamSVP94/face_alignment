import cv2
import argparse
from pathlib import Path
import albumentations as A
from pytorch_lightning import Trainer
from torch.utils.data import DataLoader
from pytorch_lightning.loggers import TensorBoardLogger

from src import glob_search
from models.Onet import ONet, EuclideanLoss
from src.constants import BASE_DIR, num_workers, AVAIL_GPUS
from src.utils_pl import FacesLandmarks_pl, CustomFaceDataset
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor


def main(args):
    # PARAMS
    start_learning_rate = 1e-3

    EXPERIMENT_NAME = 'FACIAL_LANDMARKS'

    logdir = BASE_DIR / f'logs/'
    logdir.mkdir(parents=True, exist_ok=True)

    # AUGMENTATIONS
    train_transforms = [
        A.Rotate([-15.0, 15.0], border_mode=cv2.BORDER_REPLICATE, p=0.8),
        A.RandomCrop(width=48, height=48, p=0.3),
        A.RandomCropFromBorders(crop_left=0.2, crop_right=0.2, crop_top=0.2, crop_bottom=0.2, p=0.1),
        A.GaussNoise(var_limit=(10.0, 30.0), p=0.5),
        A.RandomBrightnessContrast(p=0.2),
        A.ToGray(p=0.3),
        A.PixelDropout(p=0.2),
    ]

    train_imgs = glob_search(args.train_dir)
    val_imgs = glob_search(args.val_dir)

    # DATASETS
    train = CustomFaceDataset(img_list=train_imgs, augmentation=train_transforms)
    val = CustomFaceDataset(img_list=val_imgs)

    train_loader = DataLoader(train, batch_size=args.batch_size, shuffle=True, num_workers=num_workers, drop_last=False)
    val_loader = DataLoader(val, batch_size=1, shuffle=False, num_workers=num_workers, drop_last=False)

    # LOGGER
    tb_logger = TensorBoardLogger(save_dir=logdir, name=EXPERIMENT_NAME)
    lr_monitor = LearningRateMonitor(logging_interval='epoch')
    best_metric_saver = ModelCheckpoint(
        mode='min', save_top_k=1, save_last=True, monitor='loss/validation',
        auto_insert_metric_name=False, filename='epoch={epoch:02d}-val_loss={loss/validation:.4f}',
    )

    # MODEL
    model_pl = FacesLandmarks_pl(
        model=ONet(),  # model from paper
        loss_fn=EuclideanLoss(),  # sum mse each point loss
        start_learning_rate=start_learning_rate,
        max_epochs=args.epochs
    )

    # TRAIN
    trainer = Trainer(
        max_epochs=args.epochs, num_sanity_val_steps=0, devices=-1,
        accelerator=args.device, logger=tb_logger, log_every_n_steps=3,
        callbacks=[lr_monitor, best_metric_saver],
    )

    weights = None  # start from checkpoint
    trainer.fit(model=model_pl, train_dataloaders=train_loader, val_dataloaders=val_loader, ckpt_path=weights)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--train_dir', type=str, required=True, help='')
    parser.add_argument('-v', '--val_dir', type=str, required=True, help='', )
    parser.add_argument('--epochs', type=int, default=100, help='', )
    parser.add_argument('--batch_size', type=int, default=2048, help='', )
    parser.add_argument('--device', choices=['cuda', 'cpu'], default='cuda', help='')
    args = parser.parse_args()

    args.train_dir = Path(args.train_dir).resolve()
    assert args.train_dir.exists()

    args.val_dir = Path(args.val_dir).resolve()
    assert args.val_dir.exists()

    if AVAIL_GPUS == 0:
        args.device = 'cpu'
        args.batch_size = 1

    main(args)
