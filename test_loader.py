import sys, os
import torch
import argparse
import timeit
import numpy as np
import scipy.misc as misc
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

from torch.utils import data
from tqdm import tqdm

from ptsemseg.models import get_model
from ptsemseg.loader import get_loader, get_data_path
from ptsemseg.utils import convert_state_dict

import yaml
from pathlib import Path
import natsort
import cv2 as cv

def test(args,cfg):

    os.environ["CUDA_VISIBLE_DEVICES"] = "1"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_file_name = os.path.split(args.model_path)[1]
    model_name = model_file_name[: model_file_name.find("_")]

    IMG_Path=Path(args.img_path)
    IMG_File=natsort.natsorted(list(IMG_Path.glob("*.tif")),alg=natsort.PATH)
    IMG_Str=[]
    for i in IMG_File:
        IMG_Str.append(str(i))
    # Setup image
    print("Read Input Image from : {}".format(args.img_path))

    data_loader = get_loader(args.dataset)
    data_path = get_data_path(args.dataset,config_file=cfg)
    loader = data_loader(data_path, is_transform=True, img_norm=args.img_norm)
    n_classes = loader.n_classes

    v_loader = data_loader(
        data_path,
        is_transform=True,
        split=cfg['data']['val_split'],
        img_size=(cfg['data']['img_rows'], cfg['data']['img_cols']),)

    valloader = data.DataLoader(v_loader,
                                batch_size=cfg['training']['batch_size'],
                                num_workers=cfg['training']['n_workers'])

    # Setup Model
    model = get_model(cfg['model'], n_classes)
    state = convert_state_dict(torch.load(args.model_path)["model_state"])
    # state=torch.load(args.model_path)["model_state"]
    model.load_state_dict(state)
    model.eval()
    model.to(device)

    with torch.no_grad():
        for i_val, (img_path,images_val, labels_val) in tqdm(enumerate(valloader)):
            img_name=img_path[0]
            images_val = images_val.to(device)
            outputs = model(images_val)

            pred = np.squeeze(outputs.data.max(1)[1].cpu().numpy(), axis=0)
            decoded = loader.decode_segmap(pred)
            out_path="test_out/CAN_res50_4band_data07/"+Path(img_name).stem+".png"
            decoded_bgr = cv.cvtColor(decoded, cv.COLOR_RGB2BGR)
            # misc.imsave(out_path, decoded)
            cv.imwrite(out_path, decoded_bgr)

    # print("Classes found: ", np.unique(pred))
    # print("Segmentation Mask Saved at: {}".format(args.out_path))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Params")
    parser.add_argument(
        "--model_path",
        nargs="?",
        type=str,
        default="pretrain/data07/mv3_res50_4band_my_best_model.pkl",
        help="Path to the saved model",
    )
    parser.add_argument(
        "--dataset",
        nargs="?",
        type=str,
        default="my_test",
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
        "--img_path", nargs="?", type=str,
        default="dataset/07_train38_v2_4band/val", help="Path of the input image"
    )
    parser.add_argument(
        "--out_path",
        nargs="?",
        type=str,
        default="tk.png",
        help="Path of the output segmap",
    )
    parser.add_argument(
        "--config",
        nargs="?",
        type=str,
        default="configs/mv3_1_true_2_res50_4band_GID_data07_for_test.yml",
        help="Configuration file to use"
    )
    args = parser.parse_args()
    with open(args.config) as fp:
        cfg = yaml.load(fp)
    test(args,cfg)
