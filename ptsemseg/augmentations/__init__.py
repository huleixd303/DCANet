import logging
from ptsemseg.augmentations.augmentations import *

logger = logging.getLogger('ptsemseg')

key2aug = {'gamma': AdjustGamma,
           'hue': AdjustHue,
           'brightness': AdjustBrightness,
           'saturation': AdjustSaturation,
           'contrast': AdjustContrast,
           'rcrop': RandomCrop,
           'hflip': RandomHorizontallyFlip,
           'vflip': RandomVerticallyFlip,
           'scale': Scale,
           'rot90': RandomRot90,
           'rot180': RandomRot180,
           'rot270': RandomRot270,
           'rotangle': RandomRotangle,
           'transpose': RandomTransPose,
           'rsize': RandomSized,
           'rsizecrop': RandomSizedCrop,
           'rotate': RandomRotate,
           'gaussianblur': GaussianBlur,
           'translate': RandomTranslate,
           'flip_rotate': Rotate_flip_img,
           'ccrop': CenterCrop,}

def get_composed_augmentations(aug_dict):
    if aug_dict is None:
        logger.info("Using No Augmentations")
        return None

    augmentations = []
    for aug_key, aug_param in aug_dict.items():
        augmentations.append(key2aug[aug_key](aug_param))
        logger.info("Using {} aug with params {}".format(aug_key, aug_param))
    return Compose(augmentations)


