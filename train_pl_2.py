import torch
import torchlm
import argparse
from pathlib import Path
import albumentations as A
from src import glob_search
from pytorch_lightning import Trainer
from torch.utils.data import DataLoader
from models.torch_models import ONet, ResNet18
from pytorch_lightning.loggers import TensorBoardLogger
from src.constants import BASE_DIR, num_workers, AVAIL_GPUS
from src.utils_pl import FacesLandmarks_pl, CustomFaceDataset
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor


def main(args):
    # PARAMS
    start_learning_rate = 1e-5

    EXPERIMENT_NAME = 'FACIAL_LANDMARKS'

    logdir = BASE_DIR / f'logs/'
    logdir.mkdir(parents=True, exist_ok=True)

    # ''' TORCHLM
    train_transforms = [
        torchlm.LandmarksRandomHSV(prob=0.5),
        torchlm.LandmarksRandomShear(shear_factor=0.5, prob=0.2),
        torchlm.LandmarksRandomPatches(prob=0.15),
        torchlm.LandmarksRandomPatchesMixUp(prob=0.15),
        torchlm.LandmarksRandomBackground(prob=0.2),
        torchlm.LandmarksRandomBackgroundMixUp(alpha=0.4, prob=0.1),

        torchlm.LandmarksRandomMask(prob=0.05),
        torchlm.LandmarksRandomMaskMixUp(prob=0.1),
        torchlm.LandmarksRandomBlur(kernel_range=(5, 25), prob=0.1),
        torchlm.LandmarksRandomBrightness(prob=0.25),
        torchlm.bind(A.ToGray(always_apply=True), prob=0.25),
        torchlm.LandmarksRandomRotate(40, bins=8, prob=0.25),
    ]
    # '''

    train_imgs = glob_search(args.train_dir)
    val_imgs = glob_search(args.val_dir)

    # DATASETS
    train = CustomFaceDataset(img_list=train_imgs, augmentation=train_transforms)
    val = CustomFaceDataset(img_list=val_imgs)

    train_loader = DataLoader(train, batch_size=args.batch_size, shuffle=True, num_workers=num_workers, drop_last=False)
    val_loader = DataLoader(val, batch_size=args.batch_size, shuffle=False, num_workers=num_workers, drop_last=False)

    ''' for checking augmentations
    print(68, train.augmentation)
    for idx, (img, gt_t) in enumerate(train_loader):
        if idx > 3:
            exit()
        plt_show_img(CustomFaceDataset.draw(img, gt_t), mode='cv2')
    # '''

    # LOGGER
    tb_logger = TensorBoardLogger(save_dir=logdir, name=EXPERIMENT_NAME)
    lr_monitor = LearningRateMonitor(logging_interval='epoch')
    best_metric_saver = ModelCheckpoint(
        mode='min', save_top_k=1, save_last=True, monitor='loss/validation',
        auto_insert_metric_name=False, filename='epoch={epoch:02d}-val_loss={loss/validation:.4f}',
    )

    # MODEL
    model_pl = FacesLandmarks_pl(
        model=ResNet18(
            pretrained_weights=args.pretrained),
        loss_fn=torch.nn.MSELoss(),
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
    parser.add_argument('-v', '--val_dir', type=str, required=True, help='')
    parser.add_argument('-p', '--pretrained', type=str, default=None, help='')
    parser.add_argument('--epochs', type=int, default=100, help='', )
    parser.add_argument('--batch_size', type=int, default=512, help='', )
    parser.add_argument('--device', choices=['cuda', 'cpu'], default='cuda', help='')
    args = parser.parse_args()

    args.train_dir = Path(args.train_dir).resolve()
    assert args.train_dir.exists()

    args.val_dir = Path(args.val_dir).resolve()
    assert args.val_dir.exists()

    if args.pretrained is not None:
        args.pretrained = Path(args.pretrained).resolve()
        assert args.pretrained.exists()

    if AVAIL_GPUS == 0:
        args.device = 'cpu'
        args.batch_size = 1

    main(args)
