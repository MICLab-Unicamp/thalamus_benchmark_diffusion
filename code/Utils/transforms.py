import numpy as np
from skimage.transform import rescale, rotate
from torchvision.transforms import Compose


def My_transforms(scale=None, angle=None, flip_prob=None, sigma=None, ens_treshold=None):
    transform_list = []
    
    if scale is not None:
        transform_list.append(Scale(scale))
    if angle is not None:
        transform_list.append(Rotate(angle))
    if flip_prob is not None:
        transform_list.append(HorizontalFlip(flip_prob))
    if sigma is not None:
        transform_list.append(Gaussian_noise(sigma))

    return Compose(transform_list)
    
    
class Scale(object):

    def __str__(self):
        return 'scale = ' + str(self.scale)
        
    def __init__(self, scale):
        self.scale = scale

    def __call__(self, sample):
        image, mask = sample

        img_size = image.shape[0]

        scale = np.random.uniform(low=1.0 - self.scale, high=1.0 + self.scale)

        image = rescale(
            image,
            (scale, scale),
            multichannel=True,
            preserve_range=True,
            mode="constant",
            anti_aliasing=False,
        )
        mask = rescale(
            mask,
            (scale, scale),
            order=0,
            multichannel=True,
            preserve_range=True,
            mode="constant",
            anti_aliasing=False,
        )

        if scale < 1.0:
            diff = (img_size - image.shape[0]) / 2.0
            padding = ((int(np.floor(diff)), int(np.ceil(diff))),) * 2 + ((0, 0),)
            image = np.pad(image, padding, mode="constant", constant_values=0)
            mask = np.pad(mask, padding, mode="constant", constant_values=0)
        else:
            x_min = (image.shape[0] - img_size) // 2
            x_max = x_min + img_size
            image = image[x_min:x_max, x_min:x_max, ...]
            mask = mask[x_min:x_max, x_min:x_max, ...]

        return image, mask


class Rotate(object):

    def __init__(self, angle):
        self.angle = angle

    def __call__(self, sample):
        image, mask = sample
        angle = np.random.uniform(low=-self.angle, high=self.angle)
        image = rotate(image.T, angle, resize=False, preserve_range=True, mode="constant").T
        mask = rotate(
            mask.T, angle, resize=False, order=0, preserve_range=True, mode="constant"
        ).T
        return image, mask


class HorizontalFlip(object):

    def __init__(self, flip_prob):
        self.flip_prob = flip_prob

    def __call__(self, sample):
        image, mask = sample

        if np.random.rand() > self.flip_prob:
            return image, mask

        image = np.flipud(image.T).T.copy()
        mask = np.flipud(mask.T).T.copy()

        return image, mask
        
        
class Gaussian_noise(object):

    def __init__(self, sigma):
        self.sigma = sigma

    def __call__(self, sample):
        image, mask = sample

        image = image + np.random.normal(0, self.sigma, image.shape)
        return image, mask
        
        
class Gaussian_noise(object):

    def __init__(self, sigma):
        self.sigma = sigma

    def __call__(self, sample):
        image, mask = sample

        image = image + np.random.normal(0, self.sigma, image.shape)
        return image, mask