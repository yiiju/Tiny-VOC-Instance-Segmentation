import os
import glob
import torch
from PIL import Image
from pycocotools.coco import COCO


class TinyDataset(object):
    def __init__(self, root, transforms, mode):
        self.root = root
        self.transforms = transforms
        self.mode = mode
        if mode == "train":
            self.annojson = os.path.join(root, "pascal_train.json")
            self.annococo = COCO(self.annojson)
        elif mode == "test":
            self.annojson = os.path.join(root, "test.json")
            self.annococo = COCO(self.annojson)

    def __getitem__(self, idx):
        imgid = list(self.annococo.imgs.keys())[idx]
        img_info = self.annococo.loadImgs(ids=imgid)
        if self.mode == "train":
            # load images
            img_path = os.path.join(self.root, "train",
                                    img_info[0]['file_name'])
            img = Image.open(img_path).convert("RGB")

            # get mask
            annids = self.annococo.getAnnIds(imgIds=imgid)
            anns = self.annococo.loadAnns(annids)
            num_objs = len(annids)
            boxes = []
            labels = []
            iscrowd = []
            masks = []
            for i in range(len(annids)):
                bbox = anns[i]['bbox']
                xmin = bbox[0]
                ymin = bbox[1]
                width = bbox[2]
                height = bbox[3]
                xmax = xmin + width
                ymax = ymin + height
                boxes.append([xmin, ymin, xmax, ymax])
                labels.append(anns[i]['category_id'])
                iscrowd.append(anns[i]['iscrowd'])
                masks.append(self.annococo.annToMask(anns[i]))

            # convert everything into a torch.Tensor
            boxes = torch.as_tensor(boxes, dtype=torch.float32)
            masks = torch.as_tensor(masks, dtype=torch.uint8)
            iscrowd = torch.as_tensor(iscrowd, dtype=torch.uint8)

            image_id = torch.tensor([imgid])
            area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])

            target = {}
            target["boxes"] = boxes
            target["labels"] = torch.as_tensor(labels, dtype=torch.int64)
            target["image_id"] = image_id
            target["area"] = area
            target["iscrowd"] = iscrowd
            target["masks"] = masks

            if self.transforms is not None:
                img, target = self.transforms(img, target)
        else:
            img_path = os.path.join(self.root, "test",
                                    img_info[0]['file_name'])
            img = Image.open(img_path).convert("RGB")
            target = imgid
            if self.transforms is not None:
                img = self.transforms(img)

        return img, target

    def __len__(self):
        return len(self.annococo.imgs.keys())
