import torch
import torch.nn as nn
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torchvision import transforms
import torch.distributed as dist

import os
import numpy as np
import argparse
from PIL import Image

from detection.engine import train_one_epoch, evaluate
import detection.utils as utils
import detection.transforms as T

from tinyDataset import TinyDataset
from utils import mkExpDir
from options import args


def get_transform(train):
    transforms = []
    transforms.append(T.ToTensor())
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))

    if args.norm:
        transforms.append(T.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225]))
    return T.Compose(transforms)


def get_model_instance_segmentation(num_classes):
    # load an instance segmentation model
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(
        pretrained=False,
        pretrained_backbone=True)

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
    # make save_dir
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

    num_classes = args.epochs
    # use our dataset and defined transformations
    dataset = TinyDataset('../data', get_transform(train=True), mode="train")
    dataset_test = TinyDataset('../data',
                               get_transform(train=False), mode="train")

    # split the dataset in train and test set
    indices = torch.randperm(len(dataset)).tolist()
    if args.split_val:
        dataset = torch.utils.data.Subset(dataset, indices[:-args.split_val])
        dataset_test = torch.utils.data.Subset(dataset_test,
                                               indices[-args.split_val:])
    else:
        dataset = torch.utils.data.Subset(dataset, indices[:])
        dataset_test = torch.utils.data.Subset(dataset_test, indices[:])

    # define training and validation data loaders
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True, num_workers=4,
        collate_fn=utils.collate_fn)

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=args.batch_size, shuffle=False, num_workers=4,
        collate_fn=utils.collate_fn)

    # 0 is background
    num_classes = 21
    model = get_model_instance_segmentation(num_classes)
    if args.checkpoint:
        model.load_state_dict(torch.load(args.checkpoint))
    model.to(device)

    # construct an optimizer
    for p in model.parameters():
        p.requires_grad = True
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=args.lr_rate,
                                momentum=0.9, weight_decay=0.0005)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                   step_size=args.step_size,
                                                   gamma=args.gamma)

    num_epochs = args.epochs

    for epoch in range(num_epochs):
        _logger.info("Epoch " + str(epoch))
        # train for one epoch, printing every 10 iterations
        log = train_one_epoch(model, optimizer, data_loader,
                              device, epoch, print_freq=args.print_freq)

        _logger.info(log)
        # update the learning rate
        lr_scheduler.step()

        # evaluate on the test dataset
        coco_evaluator = evaluate(model, data_loader_test, device=device)
        _logger.info("IoU metric: segm ")
        _logger.info("Average Precision (AP) " +
                     "@[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = " +
                     str(coco_evaluator.coco_eval['segm'].stats[0]))

        if epoch % args.save_every == 0:
            _logger.info('saving the model...')
            path = args.save_dir.strip('/') + '/model/model_'
            path = path + str(epoch).zfill(5) + '.pth'
        # store the model
        torch.save(model.state_dict(), path)
