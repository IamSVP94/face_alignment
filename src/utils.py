import cv2
import random
import numpy as np
from tqdm import tqdm
from pathlib import Path
import matplotlib as mpl
import matplotlib.pyplot as plt
from typing import Union, List, Tuple

mpl.rcParams['figure.dpi'] = 200  # plot quality
mpl.rcParams['figure.subplot.left'] = 0.01
mpl.rcParams['figure.subplot.right'] = 1 - mpl.rcParams['figure.subplot.left']


# Jaw line [0, 16]
# Left eyebrow [17, 21]
# Right eyebrow [22, 26]
# Nose bridge [27, 30]
# Lower nose [30, 35]
# Left eye [36, 41]
# Right Eye [42, 47]
# Outer lip [48, 59]
# Inner lip [60, 67]



def glob_search(directories: Union[str, Path, List[str], List[Path]],
                pattern: str = '**/*',
                formats: Union[List[str], Tuple[str], str] = ('png', 'jpg', 'jpeg'),
                shuffle: bool = False,
                seed: int = 2,
                sort: bool = False,
                exception_if_empty=True,
                return_pbar=False):
    if isinstance(directories, (str, Path)):
        directories = [Path(directories)]
    files = []
    for directory in directories:
        if isinstance(directory, (str)):
            directory = Path(directory)
        if formats:
            if formats == '*':
                files.extend(directory.glob(f'{pattern}.{formats}'))
            else:
                for format in formats:
                    files.extend(directory.glob(f'{pattern}.{format.lower()}'))
                    files.extend(directory.glob(f'{pattern}.{format.upper()}'))
                    files.extend(directory.glob(f'{pattern}.{format.capitalize()}'))
        else:
            files.extend(directory.glob(f'{pattern}'))
    if exception_if_empty:
        if not len(files):
            raise Exception(f'There are no such files!')
    if shuffle:
        random.Random(seed).shuffle(files)
    if sort:
        files = sorted(files)
    if return_pbar:
        return tqdm(files, leave=True, colour='blue')
    return files


def get_random_colors(n=1):
    colors = []
    for i in range(n):
        randomcolor = (random.randint(0, 150), random.randint(50, 200), random.randint(50, 200))
        colors.append(randomcolor)
    return colors


def max_show_img_size_reshape(img, max_show_img_size, return_coef=False):  # h,w format
    img_c = img.copy()
    h, w = img_c.shape[:2]
    coef = 1
    if h > max_show_img_size[0] or w > max_show_img_size[1]:
        h_coef = h / max_show_img_size[0]
        w_coef = w / max_show_img_size[1]
        if h_coef < w_coef:  # save the biggest side
            new_img_width = max_show_img_size[1]
            coef = w / new_img_width
            new_img_height = h / coef
        else:
            new_img_height = max_show_img_size[0]
            coef = h / new_img_height
            new_img_width = w / coef
        new_img_height, new_img_width = map(int, [new_img_height, new_img_width])
        img_c = cv2.resize(img_c, (new_img_width, new_img_height), interpolation=cv2.INTER_LINEAR)
    if return_coef:
        return img_c, coef
    return img_c


def plt_show_img(img,
                 title: str = None,
                 add_coef: bool = False,
                 mode: str = 'plt',
                 max_img_size: Tuple[str] = (900, 1800)) -> None:
    """
    Display an image using either matplotlib or OpenCV.

    Parameters:
        img (np.ndarray): The image to be displayed.
        title (str, optional): The title of the image. Defaults to None.
        mode (str, optional): The mode to use for displaying the image. It can be either 'plt' for matplotlib or 'cv2' for OpenCV. Defaults to 'plt'.
        max_img_size (Tuple[str], optional): The maximum size of the image to be displayed. Defaults to (900, 900).

    Returns:
        None: This function does not return anything.
    """
    assert mode in ['cv2', 'plt']
    img_show = np.interp(img, (img.min(), img.max()), (0, 255)) if add_coef else img.copy()
    img_show = img_show.astype(np.uint8)
    if mode == 'plt':
        img_show = cv2.cvtColor(img_show, cv2.COLOR_BGR2RGB)
    title = str(title) if title is not None else 'image'
    if mode == 'plt':
        fig = plt.figure()
        ax = fig.add_subplot()
        ax.imshow(img_show)
        if title:
            ax.set_title(title)
        fig.show()
    elif mode == 'cv2':
        if max_img_size is not None:
            img_show = max_show_img_size_reshape(img_show, max_img_size)
        cv2.imshow(title, img_show)
        cv2.waitKey(0)
        cv2.destroyWindow(title)


def read_pts(path):
    return np.loadtxt(path, comments=("version:", "n_points:", "{", "}"))



def draw_points(image, points_list, colors=None, radius=2):
    img = image.copy()
    if colors is None:
        colors = get_random_colors(len(points_list[0]))
    for points in points_list:
        for idx, point in enumerate(points):
            x, y = map(int, point)
            cv2.circle(img, (x, y), radius, colors[idx], thickness=-1)
    return img


def cv2_add_title(img, title, color=(255, 0, 0), filled=True,
                  text_pos=None,
                  font=cv2.FONT_HERSHEY_COMPLEX, font_scale=1.0,
                  thickness=1, where='bottom', add_pad=None, opacity=None):
    img_c = img.copy()
    (text_w, text_h), _ = cv2.getTextSize(title, font, font_scale, thickness)
    if text_pos is None:
        text_pos_x, text_pos_y = 0, 0
    else:
        text_pos_x, text_pos_y = text_pos[:2]
    if where == 'bottom':
        text_pos_x, text_pos_y = text_pos_x, text_pos_y - 5
    if where == 'top':
        text_pos_x, text_pos_y = text_pos_x, text_pos_y + text_h
    if filled:
        cv2.rectangle(img_c, (text_pos_x, text_pos_y - text_h - 1), (text_pos_x + text_w, text_pos_y + 4), color, -1)
        color = (255, 255, 255)
    cv2.putText(img_c, title, (text_pos_x, text_pos_y), font, font_scale, color, thickness)
    if opacity:
        img_c = cv2.addWeighted(img_c, opacity, img, 1 - opacity, gamma=0)  # add opacity
    if add_pad:
        img_c = cv2.copyMakeBorder(img_c, add_pad, add_pad, add_pad, add_pad, cv2.BORDER_CONSTANT, None, color)
    return img_c
