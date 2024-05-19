import argparse
from pathlib import Path

import cv2
import dlib
import numpy as np

from src import glob_search, read_pts, plt_show_img, draw_points, cv2_add_title

np.set_printoptions(precision=3, suppress=True)


def draw_face_rect(image, dets, scores, idx, color=(0, 0, 255)):
    img_c = image.copy()
    for i, (d, s, id) in enumerate(zip(dets, scores, idx)):
        cv2.rectangle(img_c, (d.left(), d.top()), (d.right(), d.bottom()), color[::-1], 1)
        img_c = cv2_add_title(
            img_c,
            f"{i=} {id=} {s=:.3}",
            text_pos=(d.left(), d.top()),
            color=color[::-1],
            font_scale=0.5,
        )
    return img_c


def letterbox(frame, new_shape, color=(0, 0, 0)):
    img_orig_height, img_orig_width, _ = frame.shape
    model_input_width, model_input_height = new_shape
    w_ratio, h_ratio = img_orig_width / model_input_width, img_orig_height / model_input_height
    if max(w_ratio, h_ratio) < 1:
        new_img_width, new_img_height = img_orig_width, img_orig_height
    else:
        if w_ratio > h_ratio:
            new_img_width = model_input_width
            new_img_height = img_orig_height / w_ratio
        else:
            new_img_height = model_input_height
            new_img_width = img_orig_width / h_ratio
    new_img_width, new_img_height = map(round, [new_img_width, new_img_height])
    top = bottom = left = right = 0  # initial paddings
    if model_input_width - new_img_width != 0:
        left = int((model_input_width - new_img_width) / 2)
        right = model_input_width - (new_img_width + left)
    if model_input_height - new_img_height != 0:
        top = int((model_input_height - new_img_height) / 2)
        bottom = model_input_height - (new_img_height + top)
    img_ = cv2.resize(frame, (new_img_width, new_img_height), interpolation=cv2.INTER_LINEAR)
    img_ = cv2.copyMakeBorder(img_, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return img_, top, left, new_img_width, new_img_height


def main(args):
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(f'models/dlib/shape_predictor_68_face_landmarks.dat')

    imgs_bar = glob_search(args.src_dir, return_pbar=True)
    for img_idx, img_path in enumerate(imgs_bar):
        imgs_bar.set_description(f"{img_path}")

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

                crop = img[int(ymin):int(ymax), int(xmin):int(xmax)]

                rects = detector(crop, 1)

                # loop over the face detections
                for (i, rect) in enumerate(rects):
                    points = predictor(crop, rect)
                    points = np.array([[p.x + xmin, p.y + ymin] for p in points.parts()])

                    new_img_path = args.dst_dir / img_dir / img_path.name
                    new_img_path.parent.mkdir(parents=True, exist_ok=True)
                    new_annot_path = new_img_path.with_suffix('.pts')

                    cv2.imwrite(str(new_img_path), img)
                    np.savetxt(
                        str(new_annot_path), points, newline="\n", fmt='%1.6f',
                        comments="", header="version: 2\nn_points: 68\n{", footer="}",
                    )
    print(f'Saved to {args.dst_dir}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--src_dir', type=str,
                        # required=True,
                        # default='/home/vid/hdd/datasets/FACES/landmarks_task/torch/test/300W/',
                        default='/home/vid/hdd/datasets/FACES/landmarks_task/300W/test/',
                        help='')
    parser.add_argument('-d', '--dst_dir', type=str, default=None, help='')
    parser.add_argument('--scale', type=float, default=1.0, help='')
    args = parser.parse_args()

    args.src_dir = Path(args.src_dir).resolve()
    assert args.src_dir.exists()

    args.dst_dir = Path(
        args.dst_dir).resolve() if args.dst_dir else args.src_dir.parent / f'{args.src_dir.name}_dlib'
    args.dst_dir.mkdir(exist_ok=True, parents=True)

    main(args)
