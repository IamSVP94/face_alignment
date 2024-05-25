import cv2
import argparse
import numpy as np
from pathlib import Path
from src.utils import glob_search, read_pts, draw_points, plt_show_img, get_random_colors


def main(args):
    colors = np.linspace([0, 0, 0], [255, 255, 255], 68)
    colors[17:21, 2] = 0  # Left eyebrow
    colors[22:26, 1] = 0  # Right eyebrow
    colors[36:41, 2] = 0  # Left eye
    colors[42:47, 1] = 0  # Right Eye
    imgs_bar = glob_search(args.src_dir, shuffle=True, return_pbar=True)
    for img_path in imgs_bar:
        imgs_bar.set_description(f"{img_path}")

        img = cv2.imread(str(img_path))

        annot_path = img_path.with_suffix('.pts')
        if annot_path.exists():
            points = read_pts(annot_path)

            radius = max(1, img.shape[1] // 150)
            drawed = draw_points(img, [points], radius=radius, colors=colors)
            plt_show_img(drawed, mode=args.mode, title=f"{img_path.parent.name} with {len(points)} points")
        else:
            plt_show_img(img, mode=args.mode, title=f"{img_path.parent.name} without annotations!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--src_dir', type=str, help='',
                        # required=True,
                        default='/home/vid/hdd/datasets/FACES/landmarks_task/torch_abs/test_preds/',
                        )
    parser.add_argument('-m', '--mode', choices=['cv2', 'plt'], default='cv2', help='')
    args = parser.parse_args()

    args.src_dir = Path(args.src_dir).resolve()
    assert args.src_dir.exists()

    main(args)
