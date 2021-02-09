# Adapted from https://github.com/ZijunDeng/pytorch-semantic-segmentation/blob/master/utils/joint_transforms.py

import math
import numbers
import random
import numpy as np
import torchvision.transforms.functional as tf
# import cv2
from PIL import Image, ImageOps, ImageFilter


class Compose(object):
    def __init__(self, augmentations):
        self.augmentations = augmentations
        self.PIL2Numpy = False

    def __call__(self, img, mask):
        if isinstance(img, np.ndarray):
            img = Image.fromarray(img, mode="RGB")
            mask = Image.fromarray(mask, mode="L")
            self.PIL2Numpy = True

        assert img.size == mask.size
        for a in self.augmentations:
            img, mask = a(img, mask)

        if self.PIL2Numpy:
            img, mask = np.array(img), np.array(mask, dtype=np.uint8) 

        return img, mask


class RandomCrop(object):
    def __init__(self, size, padding=0):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size
        self.padding = padding

    def __call__(self, img, mask):
        if self.padding > 0:
            img = ImageOps.expand(img, border=self.padding, fill=0)
            mask = ImageOps.expand(mask, border=self.padding, fill=0)

        assert img.size == mask.size
        w, h = img.size
        tw, th = self.size
        if w == tw and h == th:
            return img, mask
        if w < tw or h < th:
            return (
                img.resize((tw, th), Image.BILINEAR),
                mask.resize((tw, th), Image.NEAREST),
            )

        x1 = random.randint(0, w - tw)
        y1 = random.randint(0, h - th)
        return (
            img.crop((x1, y1, x1 + tw, y1 + th)),
            mask.crop((x1, y1, x1 + tw, y1 + th)),
        )


class AdjustGamma(object):
    def __init__(self, gamma):
        self.gamma = gamma

    def __call__(self, img, mask):
        assert img.size == mask.size
        return tf.adjust_gamma(img, random.uniform(1, 1 + self.gamma)), mask


class AdjustSaturation(object):
    def __init__(self, saturation):
        self.saturation = saturation

    def __call__(self, img, mask):
        assert img.size == mask.size
        return tf.adjust_saturation(img, 
                                    random.uniform(1 - self.saturation, 
                                                   1 + self.saturation)), mask


class AdjustHue(object):
    def __init__(self, hue):
        self.hue = hue

    def __call__(self, img, mask):
        assert img.size == mask.size
        return tf.adjust_hue(img, random.uniform(-self.hue, 
                                                  self.hue)), mask


class AdjustBrightness(object):
    def __init__(self, bf):
        self.bf = bf

    def __call__(self, img, mask):
        assert img.size == mask.size
        return tf.adjust_brightness(img, 
                                    random.uniform(1 - self.bf, 
                                                   1 + self.bf)), mask

class AdjustContrast(object):
    def __init__(self, cf):
        self.cf = cf

    def __call__(self, img, mask):
        assert img.size == mask.size
        return tf.adjust_contrast(img, 
                                  random.uniform(1 - self.cf, 
                                                 1 + self.cf)), mask

class CenterCrop(object):
    def __init__(self, size):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size

    def __call__(self, img, mask):
        assert img.size == mask.size
        w, h = img.size
        th, tw = self.size
        x1 = int(round((w - tw) / 2.))
        y1 = int(round((h - th) / 2.))
        return (
            img.crop((x1, y1, x1 + tw, y1 + th)),
            mask.crop((x1, y1, x1 + tw, y1 + th)),
        )

class RandomRotangle(object):
    def __init__(self, p):
        self.p = p

    def __call__(self, img, mask):
        if random.random() < self.p:
            angle = random.randint(0, 3)*90
            if angle != 0:
                return (
                    img.rotate(angle, Image.BILINEAR),
                    mask.rotate(angle, Image.NEAREST),
                )
        return img, mask


class  RandomRot90(object):
    def __init__(self,p):
        self.p = p

    def __call__(self, img, mask):
        if random.random() < self.p:
            return (
                img.transpose(Image.ROTATE_90),
                mask.transpose(Image.ROTATE_90),
            )
        return img, mask

class  RandomRot180(object):
    def __init__(self,p):
        self.p = p

    def __call__(self, img, mask):
        if random.random() < self.p:
            return (
                img.transpose(Image.ROTATE_180),
                mask.transpose(Image.ROTATE_180),
            )
        return img, mask

class  RandomRot270(object):
    def __init__(self,p):
        self.p = p

    def __call__(self, img, mask):
        if random.random() < self.p:
            return (
                img.transpose(Image.ROTATE_270),
                mask.transpose(Image.ROTATE_270),
            )
        return img, mask

class  RandomTransPose(object):
    def __init__(self,p):
        self.p = p

    def __call__(self, img, mask):
        if random.random() < self.p:
            return (
                img.transpose(Image.TRANSPOSE),
                mask.transpose(Image.TRANSPOSE),
            )
        return img, mask

class RandomHorizontallyFlip(object):
    def __init__(self, p):
        self.p = p

    def __call__(self, img, mask):
        if random.random() < self.p:
            return (
                img.transpose(Image.FLIP_LEFT_RIGHT),
                mask.transpose(Image.FLIP_LEFT_RIGHT),
            )
        return img, mask


class RandomVerticallyFlip(object):
    def __init__(self, p):
        self.p = p

    def __call__(self, img, mask):
        if random.random() < self.p:
            return (
                img.transpose(Image.FLIP_TOP_BOTTOM),
                mask.transpose(Image.FLIP_TOP_BOTTOM),
            )
        return img, mask


class FreeScale(object):
    def __init__(self, size):
        self.size = tuple(reversed(size))  # size: (h, w)

    def __call__(self, img, mask):
        assert img.size == mask.size
        return (
            img.resize(self.size, Image.BILINEAR),
            mask.resize(self.size, Image.NEAREST),
        )


class RandomTranslate(object):
    def __init__(self, offset):
        self.offset = offset # tuple (delta_x, delta_y)

    def __call__(self, img, mask):
        assert img.size == mask.size
        x_offset = int(2 * (random.random() - 0.5) * self.offset[0])
        y_offset = int(2 * (random.random() - 0.5) * self.offset[1])
        
        x_crop_offset = x_offset
        y_crop_offset = y_offset
        if x_offset < 0:
            x_crop_offset = 0
        if y_offset < 0:
            y_crop_offset = 0
        
        cropped_img = tf.crop(img, 
                              y_crop_offset, 
                              x_crop_offset, 
                              img.size[1]-abs(y_offset), 
                              img.size[0]-abs(x_offset))

        if x_offset >= 0 and y_offset >= 0:
            padding_tuple = (0, 0, x_offset, y_offset)

        elif x_offset >= 0 and y_offset < 0:
            padding_tuple = (0, abs(y_offset), x_offset, 0)
        
        elif x_offset < 0 and y_offset >= 0:
            padding_tuple = (abs(x_offset), 0, 0, y_offset)
        
        elif x_offset < 0 and y_offset < 0:
            padding_tuple = (abs(x_offset), abs(y_offset), 0, 0)
        
        return (
              tf.pad(cropped_img, 
                     padding_tuple, 
                     padding_mode='reflect'),
              tf.affine(mask,
                        translate=(-x_offset, -y_offset),
                        scale=1.0,
                        angle=0.0,
                        shear=0.0,
                        fillcolor=250))


# class RandomRotate(object):
#     def __init__(self, degree):
#         self.degree = degree
#
#     def __call__(self, img, mask):
#         rotate_degree = random.random() * 2 * self.degree - self.degree
#         return (
#             tf.affine(img,
#                       translate=(0, 0),
#                       scale=1.0,
#                       angle=rotate_degree,
#                       resample=Image.BILINEAR,
#                       fillcolor=(0,0,0),#RGB图像为（0,0,0）
#                       shear=0.0),
#             tf.affine(mask,
#                       translate=(0, 0),
#                       scale=1.0,
#                       angle=rotate_degree,
#                       resample=Image.NEAREST,
#                       fillcolor=250,
#                       shear=0.0))

class RandomRotate(object):
    def __init__(self, degree):
        self.degree = degree

    def __call__(self, img, mask):
        rotate_degree = random.random() * 2 * self.degree - self.degree
        return img.rotate(rotate_degree, Image.BILINEAR), mask.rotate(rotate_degree, Image.NEAREST)

# class Rotate_flip_img(object):# opecv augmentation
#     def __init__(self, degree):
#         self.degree = degree
#
#     def __call__(self, img, mask):
#
#         k = np.random.randint(
#             6)  # [k = 0-2:rotate random between 0-360degree, 3: vertical-flip, 4: horizontal-flip, 5: both-flip]
#         ###### Image rotation ######
#         h, w = img.shape[: 2]
#         center = (w / 2, h / 2)
#         angle = np.random.randint(-self.degree, self.degree)
#         scale = 1.0
#         if k < 3:
#             M = cv2.getRotationMatrix2D(center, angle, scale)
#             new_img = cv2.warpAffine(img, M, (h, w), borderMode=cv2.BORDER_REFLECT101)
#             new_gt = cv2.warpAffine(mask, M, (h, w), borderMode=cv2.BORDER_REFLECT101)
#         ###### Image rotation ######
#         ###### Image flip ######
#         if k == 3:
#             flip_axis = 0
#         elif k == 4:
#             flip_axis = 1
#         elif k == 5:
#             flip_axis = -1
#         if k > 2 and k < 6:
#             new_img = cv2.flip(img, flip_axis)
#             new_gt = cv2.flip(mask, flip_axis)
#         ###### Image flip ######
#         return new_img, new_gt

class Rotate_flip_img(object):# opecv augmentation
    def __init__(self, degree):
        self.degree = degree

    def __call__(self, img, mask):

        k = np.random.randint(
            6)  # [k = 0-2:rotate random between 0-360degree, 3: vertical-flip, 4: horizontal-flip, 5: both-flip]
        ###### Image rotation ######
        # h, w = img.shape[: 2]
        # center = (w / 2, h / 2)
        # angle = np.random.randint(-self.degree, self.degree)
        # scale = 1.0
        if k < 3:
            rotate_degree = random.random() * 2 * self.degree - self.degree
            new_img = img.rotate(rotate_degree, Image.BILINEAR)
            new_gt = mask.rotate(rotate_degree, Image.NEAREST)
        ###### Image rotation ######
        ###### Image flip ######
        if k == 3:
            new_img = img.transpose(Image.TRANSPOSE)
            new_gt = mask.transpose(Image.TRANSPOSE)
        elif k == 4:
            new_img = img.transpose(Image.FLIP_LEFT_RIGHT)
            new_gt = mask.transpose(Image.FLIP_LEFT_RIGHT)
        elif k == 5:
            new_img = img.transpose(Image.FLIP_TOP_BOTTOM)
            new_gt = mask.transpose(Image.FLIP_TOP_BOTTOM)

        ###### Image flip ######
        return new_img, new_gt


class Scale(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, img, mask):
        assert img.size == mask.size
        w, h = img.size
        if (w >= h and w == self.size) or (h >= w and h == self.size):
            return img, mask
        if w > h:
            ow = self.size
            oh = int(self.size * int(h)/int(w))
            return (
                img.resize((ow, oh), Image.BILINEAR),
                mask.resize((ow, oh), Image.NEAREST),
            )
        else:
            oh = self.size
            ow = int(self.size * w / h)
            return (
                img.resize((ow, oh), Image.BILINEAR),
                mask.resize((ow, oh), Image.NEAREST),
            )


class RandomSizedCrop(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, img, mask):
        assert img.size == mask.size
        for attempt in range(10):
            area = img.size[0] * img.size[1]
            target_area = random.uniform(0.45, 1.0) * area
            aspect_ratio = random.uniform(0.5, 2)

            w = int(round(math.sqrt(target_area * aspect_ratio)))
            h = int(round(math.sqrt(target_area / aspect_ratio)))

            if random.random() < 0.5:
                w, h = h, w

            if w <= img.size[0] and h <= img.size[1]:
                x1 = random.randint(0, img.size[0] - w)
                y1 = random.randint(0, img.size[1] - h)

                img = img.crop((x1, y1, x1 + w, y1 + h))
                mask = mask.crop((x1, y1, x1 + w, y1 + h))
                assert img.size == (w, h)

                return (
                    img.resize((self.size, self.size), Image.BILINEAR),
                    mask.resize((self.size, self.size), Image.NEAREST),
                )

        # Fallback
        scale = Scale(self.size)
        crop = CenterCrop(self.size)
        return crop(*scale(img, mask))


class RandomSized(object):
    def __init__(self, size):
        self.size = size
        self.scale = Scale(self.size)
        self.crop = RandomCrop(self.size)

    def __call__(self, img, mask):
        assert img.size == mask.size

        w = int(random.uniform(0.75, 1.25) * img.size[0])
        h = int(random.uniform(0.75, 1.25) * img.size[1])

        img, mask = (
            img.resize((w, h), Image.BILINEAR),
            mask.resize((w, h), Image.NEAREST),
        )

        return self.crop(*self.scale(img, mask))

class GaussianBlur(object):
    def __init__(self, p):
        self.p = p

    def __call__(self, img, mask):
        if random.random() < self.p:
            return (
                img.filter(ImageFilter.GaussianBlur(radius=random.random())),
                mask,
            )
        return img, mask