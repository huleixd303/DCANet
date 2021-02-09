import copy
import torchvision.models as models

from ptsemseg.models.fcn import *
from ptsemseg.models.segnet import *
from ptsemseg.models.unet import *
from ptsemseg.models.pspnet import *
from ptsemseg.models.icnet import *
from ptsemseg.models.linknet import *
from ptsemseg.models.frrn import *
from ptsemseg.models.SAR_unet import *
from ptsemseg.models.Dlinknet import *
#from ptsemseg.models.encnet import *
#from ptsemseg.models.base import *
from ptsemseg.models.mutiDlinknet import *
from ptsemseg.models.pyramidnet import *
from ptsemseg.models.difpyramidnet import *
from ptsemseg.models.contextDlinknet import *
from ptsemseg.models.attentiondifnet import *
from ptsemseg.models.SKDlinknet import *
from ptsemseg.models.SDDlinknet import *
from ptsemseg.models.difxception import *
from ptsemseg.models.DCANet import *
from ptsemseg.models.ESFNet import *
from ptsemseg.models.RCNNUNet import *
from ptsemseg.models.CasNet import *
from ptsemseg.models.BMANet import *
from ptsemseg.models.CANet import *
from ptsemseg.models.ResUnet import *
from ptsemseg.models.AMTUnet import *
from ptsemseg.models.NestedUNet import *
from ptsemseg.models.Attention_UNet import *
from ptsemseg.models.CE_Net import *
from ptsemseg.models.UNet3plus import *

def get_model(model_dict, n_classes, version=None):
    name = model_dict['arch']
    model = _get_model_instance(name)
    param_dict = copy.deepcopy(model_dict)
    param_dict.pop('arch')

    if name in ["frrnA", "frrnB"]:
        model = model(n_classes, **param_dict)

    elif name in ["fcn32s", "fcn16s", "fcn8s"]:
        model = model(n_classes=n_classes, **param_dict)
        vgg16 = models.vgg16(pretrained=True)
        model.init_vgg16_params(vgg16)

    elif name == "segnet":
        model = model(n_classes=n_classes, **param_dict)
        vgg16 = models.vgg16(pretrained=True)
        model.init_vgg16_params(vgg16)

    elif name == "unet":
        model = model(n_classes=n_classes, **param_dict)

    elif name == "Attention_UNet":
        model = model(n_classes=n_classes, **param_dict)

    elif name == "unet++":
        model = model(n_classes=n_classes, **param_dict)

    elif name =="SARunet":
        model = model(n_classes=n_classes, **param_dict)

    elif name =="RCNNUNet":
        model = model(n_classes=n_classes, **param_dict)

    elif name =="CasNet":
        model = model(n_classes=n_classes, **param_dict)

    elif name == "linknet":
        model = model(n_classes=n_classes, **param_dict)

    elif name == "ResUnet":
        model = model(n_classes=n_classes, **param_dict)

    elif name == "AMTUnet":
        model = model(n_classes=n_classes, **param_dict)

    elif name == "Dlinknet":
        model = model(n_classes=n_classes, **param_dict)

    elif name == "mutiDlinknet":
        model = model(n_classes=n_classes, **param_dict)

    elif name == "pyramidnet":
        model = model(n_classes=n_classes, **param_dict)

    elif name == "difpyramidnet":
        model = model(n_classes= n_classes, **param_dict)

    elif name == "BMANet":
        model = model(n_classes= n_classes, **param_dict)

    elif name == "CANet":
        model = model(n_classes= n_classes, **param_dict)

    elif name == "ESFNet":
        model = model(n_classes= n_classes, **param_dict)

    elif name == "contextDlinknet":
        model = model(n_classes=n_classes, **param_dict)

    elif name == "attentiondifnet":
        model = model(n_classes=n_classes, **param_dict)

    elif name =="SKDlinkNet":
        model = model(n_classes=n_classes, **param_dict)

    elif name =="SDDlinknet":
        model = model(n_classes=n_classes, **param_dict)

    elif name =="difxception":
        model = model(n_classes=n_classes, **param_dict)

    elif name =="DCANet":
        model = model(n_classes=n_classes, **param_dict)

    elif name == "pspnet":
        model = model(n_classes=n_classes, **param_dict)

    elif name == "icnet":
        model = model(n_classes=n_classes, **param_dict)

    elif name == "icnetBN":
        model = model(n_classes=n_classes, **param_dict)

    else:
        model = model(n_classes=n_classes, **param_dict)

    return model

def test_get_model(model_name, n_classes, version = None):
    name = model_name
    model = _get_model_instance(name)
    param_dict = {}

    if name in ["frrnA", "frrnB"]:
        model = model(n_classes, **param_dict)

    elif name in ["fcn32s", "fcn16s", "fcn8s"]:
        model = model(n_classes=n_classes, **param_dict)
        vgg16 = models.vgg16(pretrained=True)
        model.init_vgg16_params(vgg16)

    elif name == "segnet":
        model = model(n_classes=n_classes, **param_dict)
        vgg16 = models.vgg16(pretrained=True)
        model.init_vgg16_params(vgg16)

    elif name == "unet":
        model = model(n_classes=n_classes, **param_dict)

    elif name == "Attention_UNet":
        model = model(n_classes=n_classes, **param_dict)

    elif name == "unet++":
        model = model(n_classes=n_classes, **param_dict)

    elif name =="SARunet":
        model = model(n_classes=n_classes, **param_dict)

    elif name =="RCNNUNet":
        model = model(n_classes=n_classes, **param_dict)

    elif name =="CasNet":
        model = model(n_classes=n_classes, **param_dict)

    elif name == "linknet":
        model = model(n_classes=n_classes, **param_dict)

    elif name == "ResUnet":
        model = model(n_classes=n_classes, **param_dict)

    elif name == "Dlinknet":
        model = model(n_classes=n_classes, **param_dict)

    elif name == "AMTUnet":
        model = model(n_classes=n_classes, **param_dict)

    elif name == "mutiDlinknet":
        model = model(n_classes=n_classes, **param_dict)

    elif name == "pyramidnet":
        model = model(n_classes=n_classes, **param_dict)

    elif name == "difpyramidnet":
        model = model(n_classes= n_classes, **param_dict)

    elif name == "BMANet":
        model = model(n_classes= n_classes, **param_dict)

    elif name == "CANet":
        model = model(n_classes= n_classes, **param_dict)

    elif name == "ESFNet":
        model = model(n_classes= n_classes, **param_dict)

    elif name == "contextDlinknet":
        model = model(n_classes= n_classes, **param_dict)

    elif name == "attentiondifnet":
        model = model(n_classes=n_classes, **param_dict)

    elif name =="SKDlinkNet":
        model = model(n_classes=n_classes, **param_dict)

    elif name == "SDDlinknet":
        model = model(n_classes=n_classes, **param_dict)

    elif name =="difxception":
        model = model(n_classes=n_classes, **param_dict)

    elif name =="DCANet":
        model = model(n_classes=n_classes, **param_dict)

    elif name == "pspnet":
        model = model(n_classes=n_classes, **param_dict)

    elif name == "icnet":
        model = model(n_classes=n_classes, **param_dict)

    elif name == "icnetBN":
        model = model(n_classes=n_classes, **param_dict)

    else:
        model = model(n_classes=n_classes, **param_dict)

    return model



def _get_model_instance(name):
    try:
        return {
            "fcn32s": fcn32s,
            "fcn8s": fcn8s,
            "fcn16s": fcn16s,
            "unet": SAR_unet_base,
            "unet++": NestedUNet,
            "segnet": segnet,
            "pspnet": pspnet,
            "icnet": icnet,
            "icnetBN": icnet,
            "linknet": linknet,
            "frrnA": frrn,
            "frrnB": frrn,
            "SARunet":SAR_unet,
            "RCNNUNet":RCNNUNet,
            "Dlinknet":DinkNet34_less_pool,
            "mutiDlinknet":MutiDlinknet,
            "pyramidnet":DlinkPyramidNet,
            "difpyramidnet":DifPyramidNet,
            "contextDlinknet":ContextNet,
            "attentiondifnet":AttentionPyramidNet,
            "SKDlinkNet":SKDlinkNet,
            "SDDlinknet":SDDlinknet,
            "difxception":Difxception,
            "CasNet":CasNet,
            "ESFNet":ESFNet,
            "BMANet":BMANet,
            "CE_Net": CE_Net_,
            "UNet3+": UNet_3Plus,
            "ResUnet":ResUnet,
            "Attention_UNet": AttU_Net,
            "AMTUnet":AMTUnet,
            "CANet":CANet,
            "DCANet":DCANet
        }[name]
    except:
        raise("Model {} not available".format(name))
