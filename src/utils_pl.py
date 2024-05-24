import cv2
import numpy as np
import torch
from pathlib import Path
import albumentations as A
import pytorch_lightning as pl
from torch.utils.data import Dataset
from typing import List, Tuple, Union, Sequence
from warmup_scheduler import GradualWarmupScheduler
from albumentations import BasicTransform, BaseCompose
from albumentations.pytorch import ToTensorV2 as ToTensor

from src import read_pts, draw_points
from src.constants import input_size

# DATASET PARAMS: (scalefactor=0.00392156862745098, RGB)
# dataset_mean=[0.5632053267791609, 0.4360215688506197, 0.37790724830547495] (mean=0.4590447146450852)
# dataset_std=[0.22373028391406174, 0.19938469717035187, 0.18913242211799638] (mean=0.20408246773413666)
# SIZE PARAMS: (dataset_len=8776)
# min_width=62    min_height=69   widths_mean=339.2739288969918   heights_mean=339.5
# max_width=3242  max_height=3312 widths_median=215.0     heights_median=215.0
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


class CustomFaceDataset(Dataset):
    def __init__(self, img_list: List[str], augmentation: Sequence[Union[BasicTransform, BaseCompose]] = None) -> None:
        self.imgs = img_list
        self.flandmarks = [read_pts(path.with_suffix('.pts')) for path in img_list]  # OOM?

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

    @staticmethod
    def draw(img_t, pred_t, radius=1):
        img = img_t[0].detach().cpu().numpy().transpose(1, 2, 0)
        img = np.interp(img, (img.min(), img.max()), (0, 255))
        points = pred_t[0].detach().cpu().numpy().reshape((-1, 2))
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
        gts = labels.squeeze(1).to(torch.long)
        preds = self.forward(imgs)
        loss = self.loss_fn(preds, gts)
        self.log(f'loss/{stage}', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return {'loss': loss, 'preds': preds}


class CED:

    def fr_and_auc(self, nmes, method="simpson", thres=0.08, step=0.0001,
                   image_name=None,
                   path2saveimage=None,
                   plot_ced=False):

        num_data = len(nmes)
        xs = np.arange(0, thres + step, step)
        ys = np.array([np.count_nonzero(nmes <= x) for x in xs]) / float(num_data)
        fr = 1.0 - ys[-1]

        if method == "simpson":
            auc = integrate.simpson(ys, x=xs) / thres
        elif method == "trapez":
            auc = integrate.trapezoid(ys, x=xs) / thres

        if plot_ced:
            if image_name is None:
                plt.plot(xs, ys)
            else:
                plt.plot(xs, ys, label=f"{image_name}")
            plt.xlabel("NME")
            plt.ylabel("Images Proportion")
            plt.title(f"CED: AUC = {float('{:.4f}'.format(auc))}")
            plt.grid()

            if path2saveimage is not None:
                plt.savefig(f"{path2saveimage}/{image_name}_ced.png")

        return fr, auc

    def nme_batch(self, pred_landmarks, true_landmarks, rect):
        '''
        Parameters:
            pred_landmarks: np.array shape [batch_size, 68, 2]
            true_landmarks: np.array shape [batch_size, 68, 2]
            rect: np.array [x1,y1,x2,y2]
        Return:
            NME: np.array [batch_size]
        '''

        NME = np.array(list(starmap(self.nme_image, zip(pred_landmarks, true_landmarks, rect))))

        return NME

    def nme_image(self, pred_landmarks, true_landmarks, rect):

        '''
          Parameters:
            pred_landmarks: np.array shape: (68,2)
            true_landmarks: np.array shape: (68,2)
            rect: np.array [x1,y1,x2,y2]
          Return:
            nme: float
        '''

        W = abs(rect[2] - rect[0])
        H = abs(rect[3] - rect[1])

        # version 1
        # eucledian_distance = np.sqrt((true_points - predicted_points).pow(2).sum(dim=1))
        # version 2
        eucledian_distance = np.linalg.norm(true_landmarks - pred_landmarks, axis=1)

        d = np.sqrt(H * W)
        N = true_landmarks.shape[0]

        NME = eucledian_distance.sum() / (N * d)

        return NME
