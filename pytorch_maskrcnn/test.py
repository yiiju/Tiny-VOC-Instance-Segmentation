import torch
import torch.nn as nn
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torchvision import transforms
from pycocotools.coco import COCO
import pycocotools.mask as mask_util


import os
import json
import argparse
import numpy as np
from PIL import Image

import detection.transforms as T
import detection.utils as utils

from tinyDataset import TinyDataset
from utils import mkExpDir
from options import args


def get_transform():
    if args.norm:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])
        ])
    else:
        transform = transforms.Compose([
            transforms.ToTensor(),
        ])
    return transform


def get_model_instance_segmentation(num_classes):
    # load an instance segmentation model
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(
        pretrained=False)

    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # now get the number of input features for the mask classifier
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    # and replace the mask predictor with a new one
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask,
                                                       hidden_layer,
                                                       num_classes)

    return model


if __name__ == "__main__":
    _logger = mkExpDir(args)

    # train on the GPU or on the CPU, if a GPU is not available
    gpuid = str(args.gpuid)
    if isinstance(args.gpuid, list):
        gpuid = ""
        for i in args.gpuid:
            gpuid = gpuid + str(i) + ", "
    os.environ["CUDA_VISIBLE_DEVICES"] = gpuid
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        torch.device('cpu')

    num_classes = 21
    model = get_model_instance_segmentation(num_classes)
    model.to(device)

    model.load_state_dict(torch.load(args.checkpoint))
    model.to(device)
    dataset = TinyDataset('../', get_transform(), mode="test")
    testLoader = torch.utils.data.DataLoader(
                    dataset, batch_size=args.batch_size,
                    shuffle=False, num_workers=4,
                    collate_fn=utils.collate_fn)

    # For inference
    model.eval()
    with torch.no_grad():
        coco_results = []
        for data in testLoader:
            images, imageid = data
            images = list(image.to(device) for image in images)

            predictions = model(images)
            predictions = [{k: v.to(torch.device("cpu")) for k, v in t.items()}
                           for t in predictions]
            output = {imageid: prediction for imageid, prediction
                      in zip(imageid, predictions)}

            for original_id, prediction in output.items():
                if len(prediction) == 0:
                    continue

                scores = prediction["scores"]
                labels = prediction["labels"]
                masks = prediction["masks"]

                masks = masks > args.mask_threshold

                scores = prediction["scores"].tolist()
                labels = prediction["labels"].tolist()

                rles = [
                    mask_util.encode(np.array(mask[0, :, :, np.newaxis],
                                     dtype=np.uint8, order="F"))[0]
                    for mask in masks
                ]
                for rle in rles:
                    rle["counts"] = rle["counts"].decode("utf-8")

                coco_results.extend(
                    [
                        {
                            "image_id": original_id,
                            "score": scores[k],
                            "category_id": labels[k],
                            "segmentation": rle,
                        }
                        for k, rle in enumerate(rles)
                    ]
                )

    out_file = os.path.join(args.save_dir, args.outjson)
    with open(out_file, 'w') as json_file:
        json.dump(coco_results, json_file)
