# Tiny-VOC-Instance-Segmentation

An instance segmentation task in Tiny PASCAL VOC dataset.

## Hardware
Ubuntu 18.04 LTS

Intel(R) Xeon(R) Silver 4210 CPU @ 2.20GHz

1x GeForce RTX 2080 Ti

## Set Up
### Install Dependency
All requirements is detailed in requirements.txt.

    $ pip install -r requirements.txt

### Unzip dataset
Download dataset from [Tiny PASCAL VOC dataset](https://drive.google.com/drive/folders/1txQEFaxWd9iKBqZFN6fKD8U7alxOfLlo?usp=sharing).

Execute the following instruction
    
    $ sh data.sh

 - Create data/ directory and unzip dataset into data/.

### Using pytorch_maskrcnn

Into [pytorch_maskrcnn](/pytorch_maskrcnn) directory.

The model is Mask R-CNN with backbone ResNet50 and pretrain on ImageNet.

The details of training and test are written below.

The helper functions in [pytorch_maskrcnn/detection](/pytorch_maskrcnn/detection) directory are copy from [pytorch/vision/references/detection](https://github.com/pytorch/vision/tree/master/references/detection)

```
└── pytorch_maskrcnn
    ├── detection ─ the helper functions from pytorch
    ├── test.py
    ├── train.py
    ├── options.py - Setting of ArgumentParser
    ├── utils.py - Store the log in command line
    ├── tinyDataset.py - Custom Dataset for Tiny PASCAL VOC dataset
    ├── train.sh
    └── test.sh
```

### Coding Style
Use PEP8 guidelines.

    $ pycodestyle *.py

## Dataset - Tiny PASCAL VOC Dataset
The data directory is structured as:
```
└── data 
    ├── test ─ 1,000 test images
    ├── train ─ 1,349 training images
    ├── pascal_train.json ─ training annotations in json format
    └── test.json - test information in json format
```

## Train
Train in PyTorch. (The root is in pytorch_maskrcnn)

    $ sh tain.sh

Argument
 - `--gpuid` the ids of gpus to use
 - `--epochs` the path to store the checkpoints and config setting
 - `--batch_size` the batch size
 - `--print_freq` the frequence of print batch metric
 - `--save_dir` the directory to store checkpoints
 - `--save_every` the period of save checkpoints
 - `--log_file_name` the name of log file

## Inference
Test in PyTorch. (The root is in pytorch_maskrcnn)

    $ sh test.sh

Argument
 - `--test` set the test mode
 - `--gpuid` the ids of gpus to use
 - `--batch_size` the batch size
 - `--checkpoint` the path of load checkpoint model
 - `--outjson` the name of the result json file
 - `--save_dir` the directory to store args.txt