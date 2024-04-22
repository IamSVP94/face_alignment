import argparse

import cv2
import dlib
import numpy as np
import torch
from tqdm import tqdm
from pathlib import Path
import albumentations as A
from src import glob_search, plt_show_img, read_pts, draw_points
from models.Onet import ONet
from albumentations.pytorch import ToTensorV2 as ToTensor

from src.constants import AVAIL_GPUS
from src.utils_pl import final_transforms


def main(args):
    # load model
    model = ONet()
    state_dict = torch.load(str(args.checkpoint_pl))['state_dict']
    remove_prefix = 'model.'
    state_dict = {k[len(remove_prefix):] if k.startswith(remove_prefix) else k: v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)
    model.to(args.device)

    # dlib face detector
    detector = dlib.get_frontal_face_detector()

    model.eval()
    with torch.no_grad():
        colors = np.linspace([0, 0, 0], [255, 255, 255], 68)
        colors[17:21, 2] = 0  # Left eyebrow
        colors[22:26, 1] = 0  # Right eyebrow
        colors[36:41, 2] = 0  # Left eye
        colors[42:47, 1] = 0  # Right Eye

        dataset_imgs = glob_search(args.src_dir, shuffle=False, seed=2, return_pbar=True)
        for img_path in dataset_imgs:
            annot_path = img_path.with_suffix('.pts')
            if annot_path.exists():
                gt = read_pts(annot_path)
                if len(gt) != 68:
                    continue

                xs, ys = zip(*gt)
                cx, cy = np.mean(xs), np.mean(ys)

                img = cv2.imread(str(img_path))
                img_dir = '/'.join(img_path.parts[len(args.src_dir.parts) - len(img_path.parts):-1])  # saving hierarchy

                dets, scores, idx = detector.run(image=cv2.cvtColor(img, cv2.COLOR_RGB2BGR), upsample_num_times=0)
                for i, (d, s, id) in enumerate(zip(dets, scores, idx)):
                    if d.left() < cx < d.right() and d.top() < cy < d.bottom():  # needed face
                        xmin, xmax = max(d.left(), 0), min(d.right(), img.shape[1])
                        ymin, ymax = max(d.top(), 0), min(d.bottom(), img.shape[0])

                        xmin = max(xmin - (xmax - xmin) * (args.scale - 1), 0)
                        xmax = min(xmax + (xmax - xmin) * (args.scale - 1), img.shape[1])
                        ymin = max(ymin - (ymax - ymin) * (args.scale - 1), 0)
                        ymax = min(ymax + (ymax - ymin) * (args.scale - 1), img.shape[0])

                        orig_crop = img[int(ymin):int(ymax), int(xmin):int(xmax)]
                        transforms = A.Compose(final_transforms)
                        crop_t = transforms(image=orig_crop)['image'].unsqueeze(0).to(args.device)

                        orig_crop_h, orig_crop_w = orig_crop.shape[:2]
                        crop_t_h, crop_t_w = crop_t.shape[2:]
                        scale_w, scale_h = orig_crop_w / crop_t_w, orig_crop_h / crop_t_h

                        preds = model(crop_t).squeeze().reshape(68, 2).cpu().numpy()
                        preds[:, 0] = preds[:, 0] * scale_w + xmin
                        preds[:, 1] = preds[:, 1] * scale_h + ymin

                        # debug
                        # plt_show_img(draw_points(img, [gt], colors=colors))
                        # plt_show_img(draw_points(img, [preds], colors=colors))

                        new_img_path = args.dst_dir / img_dir / img_path.name
                        new_img_path.parent.mkdir(parents=True, exist_ok=True)
                        new_annot_path = new_img_path.with_suffix('.pts')

                        cv2.imwrite(str(new_img_path), img)
                        np.savetxt(
                            str(new_annot_path), preds, newline="\n", fmt='%1.6f',
                            comments="", header="version: 2\nn_points: 68\n{", footer="}",
                        )
    print(f'Saved to {args.dst_dir}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--src_dir', type=str, required=True, help='')
    parser.add_argument('-c', '--checkpoint_pl', type=str, required=True, help='')
    parser.add_argument('-d', '--dst_dir', type=str, default=None, help='')
    parser.add_argument('--scale', type=float, default=1.0, help='')
    parser.add_argument('--device', choices=['cuda', 'cpu'], default='cuda', help='')
    args = parser.parse_args()

    args.src_dir = Path(args.src_dir).resolve()
    assert args.src_dir.exists()

    args.checkpoint_pl = Path(args.checkpoint_pl).resolve()
    assert args.checkpoint_pl.exists()

    args.dst_dir = Path(
        args.dst_dir).resolve() if args.dst_dir else args.src_dir.parent / f'{args.src_dir.name}_preds'
    args.dst_dir.mkdir(exist_ok=True, parents=True)

    if AVAIL_GPUS == 0:
        args.device = 'cpu'
        args.batch_size = 1

    main(args)
