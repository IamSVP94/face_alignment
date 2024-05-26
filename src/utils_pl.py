import cv2
import torch
from pathlib import Path
import pytorch_lightning as pl
import torchlm
from torch.utils.data import Dataset
from typing import List, Tuple
from warmup_scheduler import GradualWarmupScheduler

from src import read_pts, draw_points
from src.constants import input_size

# DATASET PARAMS: (scalefactor=0.00392156862745098, RGB)
# dataset_mean=[0.5632053267791614, 0.43602156885061955, 0.37790724830547534] (mean=0.4590447146450854)
# dataset_std=[0.2237302839140605, 0.19938469717035237, 0.189132422117997] (mean=0.20408246773413663)
# SIZE PARAMS: (dataset_len=8776)
# min_width=62    min_height=69   widths_mean=339.2739288969918   heights_mean=339.5
# max_width=3242  max_height=3312 widths_median=215.0     heights_median=215.0

DATASET_MEAN = [0.37790724830547495, 0.4360215688506197, 0.5632053267791609]  # BGR
DATASET_STD = [0.18913242211799638, 0.19938469717035187, 0.22373028391406174]  # BGR

''' ALBUMENTATION
final_transforms = [
    A.Resize(height=input_size[0], width=input_size[1], always_apply=True),
    A.Normalize(
        mean=(0.37790724830547495, 0.4360215688506197, 0.5632053267791609),  # BGR
        std=(0.18913242211799638, 0.19938469717035187, 0.22373028391406174),  # BGR
        max_pixel_value=255.0,
        always_apply=True
    ),
    ToTensor(always_apply=True),
]
# '''

final_transforms = [
    torchlm.LandmarksResize(size=input_size),
    torchlm.LandmarksNormalize(mean=DATASET_MEAN, std=DATASET_STD, force_norm_before_mean_std=True),
    torchlm.LandmarksToTensor(),
]


class CustomFaceDataset(Dataset):
    def __init__(self, img_list: List[str], augmentation=None) -> None:
        self.imgs = img_list
        self.flandmarks = [read_pts(path.with_suffix('.pts')) for path in img_list]  # OOM?

        if augmentation is None:
            augmentation = []
        augmentation.extend(final_transforms)
        self.augmentation = torchlm.LandmarksCompose(augmentation)

    def __len__(self) -> int:
        return len(self.imgs)

    def __getitem__(self, item: int) -> Tuple:
        img_path = self.imgs[item]
        img = cv2.imread(str(img_path))  # BGR
        gt_np = self.flandmarks[item]

        # apply augmentations
        img, gt_t = self.augmentation(img, gt_np)
        gt_t = torch.Tensor(gt_t).reshape(-1)  # flatten
        return img, gt_t

    @staticmethod
    def draw(img_t, pred_t, radius=1):
        transform = torchlm.LandmarksCompose([
            torchlm.LandmarksToNumpy(),
            torchlm.LandmarksUnNormalize(mean=DATASET_MEAN, std=DATASET_STD, force_denorm_after_mean_std=True),
        ])
        img, pred = transform(img_t[0], pred_t[0])
        points = pred.reshape(-1, 2)
        return draw_points(img, [points], radius=radius)


class FacesLandmarks_pl(pl.LightningModule):
    def __init__(self, model, max_epochs=None, *args, loss_fn=torch.nn.CrossEntropyLoss(),
                 start_learning_rate=1e-3, checkpoint=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.start_learning_rate = start_learning_rate
        self.model = model
        if checkpoint is not None and Path(checkpoint).exists():
            self.model.load_state_dict(torch.load(str(checkpoint)))
        self.loss_fn = loss_fn
        self.save_hyperparameters(ignore=['model', 'loss_fn'])
        self.max_epochs = max_epochs

    def forward(self, img):
        pred = self.model(img)
        return pred

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.start_learning_rate,
            # weight_decay=5e-2,
        )

        warmup_epochs = 5
        verbose = True
        if self.max_epochs:
            lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=self.max_epochs - warmup_epochs,
                eta_min=0,
                verbose=verbose)
        else:
            lr_scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer,
                step_size=15,
                gamma=0.5,
                verbose=verbose)

        warmup_scheduler = GradualWarmupScheduler(
            optimizer,
            multiplier=1,
            total_epoch=warmup_epochs,
            after_scheduler=lr_scheduler
        )
        return {"optimizer": optimizer, "lr_scheduler": warmup_scheduler, "monitor": 'val_loss'}

    def train_dataloader(self):
        return self.loader['train']

    def training_step(self, batch, batch_idx):
        return self._shared_step(batch, stage='train')

    def validation_step(self, batch, batch_idx):
        return self._shared_step(batch, stage='validation')

    def test_step(self, batch, batch_idx):
        return self._shared_step(batch, stage='test')

    def predict_step(self, batch, batch_idx):
        imgs = batch  # Inference Dataset only!
        preds = self.forward(imgs)
        return preds

    def _shared_step(self, batch, stage):
        imgs, labels = batch
        gts = labels.squeeze(1)
        preds = self.forward(imgs)
        loss = self.loss_fn(preds, gts)
        self.log(f'loss/{stage}', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return {'loss': loss, 'preds': preds}
