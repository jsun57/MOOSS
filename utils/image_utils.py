import torch
import numpy as np
from skimage.util.shape import view_as_windows
import os
import imageio


def preprocess_obs(obs, bits=5):
    """Preprocessing image, see https://arxiv.org/abs/1807.03039."""
    bins = 2**bits
    assert obs.dtype == torch.float32
    if bits < 8:
        obs = torch.floor(obs / 2**(8 - bits))
    obs = obs / bins
    obs = obs + torch.rand_like(obs) / bins
    obs = obs - 0.5
    return obs


def sync_crop(imgs, output_size, next_imgs):
    """
    Vectorized way to do random crop using sliding windows
    and picking out random ones

    args:
        imgs, batch images with shape (B,C,H,W)
    """
    assert imgs.shape == next_imgs.shape
    # batch size
    n = imgs.shape[0]
    img_size = imgs.shape[-1]
    crop_max = img_size - output_size
    imgs = np.transpose(imgs, (0, 2, 3, 1))
    w1 = np.random.randint(0, crop_max, n)
    h1 = np.random.randint(0, crop_max, n)
    # creates all sliding windows combinations of size (output_size)
    windows = view_as_windows(imgs, (1, output_size, output_size, 1))[..., 0, :, :,0]
    # selects a random window for each batch element
    cropped_imgs = windows[np.arange(n), w1, h1]

    
    ''' process next observations '''
    next_imgs = np.transpose(next_imgs, (0, 2, 3, 1))
    next_windows = view_as_windows(next_imgs, (1, output_size, output_size, 1))[..., 0, :, :,0]
    # selects a random window for each batch element
    next_cropped_imgs = next_windows[np.arange(n), w1, h1]

    return cropped_imgs, next_cropped_imgs


def random_crop(imgs, output_size):
    """
    Vectorized way to do random crop using sliding windows
    and picking out random ones

    args:
        imgs, batch images with shape (B,C,H,W)
    """
    # batch size
    n = imgs.shape[0]
    img_size = imgs.shape[-1]
    crop_max = img_size - output_size
    imgs = np.transpose(imgs, (0, 2, 3, 1))
    w1 = np.random.randint(0, crop_max, n)
    h1 = np.random.randint(0, crop_max, n)
    # creates all sliding windows combinations of size (output_size)
    windows = view_as_windows(imgs, (1, output_size, output_size, 1))[..., 0, :, :,0]
    # selects a random window for each batch element
    cropped_imgs = windows[np.arange(n), w1, h1]
    return cropped_imgs


def center_crop(imgs, output_size):
    """
    Vectorized way to do random crop using sliding windows
    and picking out random ones

    args:
        imgs, batch images with shape (B,C,H,W)
    """
    # batch size
    n = imgs.shape[0]
    img_size = imgs.shape[-1]
    crop_max = img_size - output_size

    new_h, new_w = output_size, output_size
    top = (img_size - new_h) // 2
    left = (img_size - new_w) // 2
    cropped_imgs = imgs[:, :, top:top + new_h, left:left + new_w]

    return cropped_imgs


def random_crop_2(imgs, output_size):

    n1 = imgs.shape[0]
    n2 = imgs.shape[1]
    n = n1 * n2
    imgs = imgs.reshape(n, *imgs.shape[2:])
    img_size = imgs.shape[-1]
    crop_max = img_size - output_size
    imgs = np.transpose(imgs, (0, 2, 3, 1))
    w1 = np.random.randint(0, crop_max, n)
    h1 = np.random.randint(0, crop_max, n)
    windows = view_as_windows(imgs, (1, output_size, output_size, 1))[..., 0, :, :, 0]
    cropped_imgs = windows[np.arange(n), w1, h1]
    return cropped_imgs.reshape(n1, n2, *cropped_imgs.shape[1:])


def center_crop_image(image, output_size):
    h, w = image.shape[1:]
    new_h, new_w = output_size, output_size

    top = (h - new_h) // 2
    left = (w - new_w) // 2

    image = image[:, top:top + new_h, left:left + new_w]
    return image


def conv2d_output_shape(h, w, kernel_size=1, stride=1, padding=0, dilation=1):
    """
    Returns output H, W after convolution/pooling on input H, W.
    """
    kh, kw = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, ) * 2
    sh, sw = stride if isinstance(stride, tuple) else (stride, ) * 2
    ph, pw = padding if isinstance(padding, tuple) else (padding, ) * 2
    d = dilation
    h = (h + (2 * ph) - (d * (kh - 1)) - 1) // sh + 1
    w = (w + (2 * pw) - (d * (kw - 1)) - 1) // sw + 1
    return h, w


class VideoRecorder(object):
    def __init__(self, dir_name, height=256, width=256, camera_id=0, fps=30):
        self.dir_name = dir_name
        self.height = height
        self.width = width
        self.camera_id = camera_id
        self.fps = fps
        self.frames = []

    def init(self, enabled=True):
        self.frames = []
        self.enabled = self.dir_name is not None and enabled

    def record(self, env):
        if self.enabled:
            try:
                frame = env.render(mode='rgb_array',
                                   height=self.height,
                                   width=self.width,
                                   camera_id=self.camera_id)
            except:
                frame = env.render(mode='rgb_array', )

            self.frames.append(frame)

    def save(self, file_name):
        if self.enabled:
            path = os.path.join(self.dir_name, file_name)
            imageio.mimsave(path, self.frames, fps=self.fps)
