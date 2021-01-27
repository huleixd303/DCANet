import sys, os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import torch
import argparse
import timeit
import numpy as np
import scipy.misc as misc
from torchvision import transforms
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from PIL import Image

from torch.utils import data
from tqdm import tqdm
import glob
from ptsemseg.models import test_get_model
from ptsemseg.models import get_model
from ptsemseg.loader import get_loader, get_data_path
from ptsemseg.utils import convert_state_dict
from ptsemseg.metrics import runningScore

torch.backends.cudnn.benchmark = True

# try:
#     import pydensecrf.densecrf as dcrf
# except:
#     print(
#         "Failed to import pydensecrf,\
#            CRF post-processing will not work"
#     )

def Aerial_val(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_file_name = os.path.split(args.model_path)[1]
    model_name = model_file_name[: model_file_name.find("_")]

    # Setup image
    #print("Read Input Image from : {}".format(args.img_path))
    #img = Image.open(args.img_path)


    data_loader = get_loader(args.dataset)
    data_path = get_data_path(args.dataset)
    loader = data_loader(data_path, is_transform=True, img_norm=args.img_norm)
    n_classes = loader.n_classes

    # Setup Model
    model = test_get_model(model_name, n_classes, version=args.dataset)
    state = convert_state_dict(torch.load(args.model_path)["model_state"])
    model.load_state_dict(state)
    model.eval()
    model.to(device)
    print("Read Input Image from : {}".format(args.img_path))
    imgset = glob.glob(os.path.join(args.img_path, '*.tif'))
    lbl = glob.glob(os.path.join(args.lbl_path, '*.tif'))
    for ids in imgset:
        img = Image.open(ids)
        img = img.resize((loader.img_size[0], loader.img_size[1]))

        gt = Image.open()

        tf = transforms.Compose([transforms.ToTensor(),
                                      transforms.Normalize([0.406, 0.428, 0.394],
                                                           [0.201, 0.183, 0.176])])
        img = tf(img)
        # Setup Model
        # model = test_get_model(model_name, n_classes, version=args.dataset)
        # state = convert_state_dict(torch.load(args.model_path)["model_state"])
        # model.load_state_dict(state)
        # model.eval()
        # model.to(device)

        images = (img.to(device)).unsqueeze(0)
        pred = model(images)

        # if args.dcrf:
        #     unary = outputs.data.cpu().numpy()
        #     unary = np.squeeze(unary, 0)
        #     unary = -np.log(unary)
        #     unary = unary.transpose(2, 1, 0)
        #     w, h, c = unary.shape
        #     unary = unary.transpose(2, 0, 1).reshape(loader.n_classes, -1)
        #     unary = np.ascontiguousarray(unary)
        #
        #     resized_img = np.ascontiguousarray(resized_img)
        #
        #     d = dcrf.DenseCRF2D(w, h, loader.n_classes)
        #     d.setUnaryEnergy(unary)
        #     d.addPairwiseBilateral(sxy=5, srgb=3, rgbim=resized_img, compat=1)
        #
        #     q = d.inference(50)
        #     mask = np.argmax(q, axis=0).reshape(w, h).transpose(1, 0)
        #     decoded_crf = loader.decode_segmap(np.array(mask, dtype=np.uint8))
        #     dcrf_path = args.out_path[:-4] + "_drf.png"
        #     misc.imsave(dcrf_path, decoded_crf)
        #     print("Dense CRF Processed Mask Saved at: {}".format(dcrf_path))

        # pred = np.squeeze(outputs.data.max(1)[1].cpu().numpy(), axis=0)
        pred[pred > 0.5] = 1
        pred[pred <= 0.5] = 0
        pred = np.squeeze(pred.data.cpu().numpy())
        # if model_name in ["pspnet", "icnet", "icnetBN"]:
        #     pred = pred.astype(np.float32)
        #     # float32 with F mode, resize back to orig_size
        #     pred = misc.imresize(pred, orig_size, "nearest", mode="F")

        #decoded = loader.decode_segmap(pred)
        pred = pred*255
        decoded = Image.fromarray(pred.astype('uint8'))

        print("Classes found: ", np.unique(pred))
        out_ids_name = os.path.split(ids)[1]
        outpath = os.path.join(args.out_path, out_ids_name)
        decoded.save(outpath)
        # misc.imsave(outpath, decoded)
        print("Segmentation Mask Saved at: {}".format(args.out_path))

if __name__ == "__main__":
        parser = argparse.ArgumentParser(description="Params")
        parser.add_argument(
            "--model_path",
            nargs="?",
            type=str,
            default="fcn8s_pascal_1_26.pkl",
            help="Path to the saved model",
        )
        parser.add_argument(
            "--dataset",
            nargs="?",
            type=str,
            default="pascal",
            help="Dataset to use ['pascal, camvid, ade20k etc']",
        )

        parser.add_argument(
            "--img_norm",
            dest="img_norm",
            action="store_true",
            help="Enable input image scales normalization [0, 1] \
                                          | True by default",
        )
        parser.add_argument(
            "--no-img_norm",
            dest="img_norm",
            action="store_false",
            help="Disable input image scales normalization [0, 1] |\
                                          True by default",
        )
        parser.set_defaults(img_norm=True)

        parser.add_argument(
            "--dcrf",
            dest="dcrf",
            action="store_true",
            help="Enable DenseCRF based post-processing | \
                                          False by default",
        )
        parser.add_argument(
            "--no-dcrf",
            dest="dcrf",
            action="store_false",
            help="Disable DenseCRF based post-processing | \
                                          False by default",
        )
        parser.set_defaults(dcrf=False)

        parser.add_argument(
            "--img_path", nargs="?", type=str, default=None, help="Path of the input image"
        )
        parser.add_argument(
            "--lbl_path", nargs="?", type=str, default=None, help="Path of the input lbl"
        )
        parser.add_argument(
            "--out_path",
            nargs="?",
            type=str,
            default=None,
            help="Path of the output segmap",
        )
        args = parser.parse_args()
        Aerial_val(args)