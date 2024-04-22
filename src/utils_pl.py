import cv2
import torch
from pathlib import Path
import albumentations as A
import pytorch_lightning as pl
from torch.utils.data import Dataset
from typing import List, Tuple, Union, Sequence
from warmup_scheduler import GradualWarmupScheduler
from albumentations import BasicTransform, BaseCompose
from albumentations.pytorch import ToTensorV2 as ToTensor

from src import read_pts
from src.constants import input_size

# DATASET PARAMS: (scalefactor=0.00392156862745098, RGB)
# dataset_mean=[0.46515819699220795, 0.39155367530522783, 0.35058879452168296] (mean=0.40243355560637295)
# dataset_std=[0.2667869035720346, 0.2450091688854225, 0.2371264524072626] (mean=0.2496408416215732)
# SIZE PARAMS: (dataset_len=7875)
# min_width=91    min_height=97   widths_mean=675.4233650793651   heights_mean=677.1892063492063
# max_width=4211  max_height=5400 widths_median=435.0     heights_median=457.0
final_transforms = [
    A.Resize(height=input_size[0], width=input_size[1], always_apply=True),
    A.Normalize(
        mean=(0.35058879452168296, 0.39155367530522783, 0.46515819699220795),  # BGR
        std=(0.2371264524072626, 0.2450091688854225, 0.2667869035720346),  # BGR
        max_pixel_value=255.0,
        always_apply=True
    ),
    ToTensor(always_apply=True),
]


class CustomFaceDataset(Dataset):
    def __init__(self,
                 img_list: List[str],
                 augmentation: Sequence[Union[BasicTransform, BaseCompose]] = None,
                 input_size: Tuple[int, int] = (48, 48),
                 ) -> None:
        self.imgs = img_list
        self.flandmarks = [read_pts(path.with_suffix('.pts')) for path in img_list]

        if augmentation is None:
            augmentation = []
        augmentation.extend(final_transforms)

        self.augmentation = A.Compose(
            augmentation,
            keypoint_params=A.KeypointParams(format='xy', remove_invisible=False),
        )

    def __len__(self) -> int:
        return len(self.imgs)

    def __getitem__(self, item: int) -> Tuple:
        img_path = self.imgs[item]
        img = cv2.imread(str(img_path))  # BGR
        gt_np = self.flandmarks[item]
        # apply augmentations
        with_augs = self.augmentation(image=img, keypoints=gt_np)
        img, gt_t = with_augs['image'], with_augs['keypoints']
        gt_t = torch.Tensor(gt_t).to(torch.float64).reshape(-1)  # flatten
        return img, gt_t


class FacesLandmarks_pl(pl.LightningModule):
    def __init__(self, model, *args, loss_fn=torch.nn.CrossEntropyLoss(),
                 start_learning_rate=1e-3, checkpoint=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.start_learning_rate = start_learning_rate
        self.model = model
        if checkpoint is not None and Path(checkpoint).exists():
            self.model.load_state_dict(torch.load(str(checkpoint)))
        self.loss_fn = loss_fn
        self.save_hyperparameters(ignore=['model', 'loss_fn'])

    def forward(self, img):
        pred = self.model(img)
        return pred

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.start_learning_rate,
            weight_decay=5e-2,
        )

        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.5, verbose=True)
        warmup_scheduler = GradualWarmupScheduler(optimizer, multiplier=1, total_epoch=5, after_scheduler=lr_scheduler)
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
        gts = labels.squeeze(1).to(torch.long)
        preds = self.forward(imgs)
        loss = self.loss_fn(preds, gts)
        self.log(f'loss/{stage}', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return {'loss': loss, 'preds': preds}
