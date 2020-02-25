"""
Mask R-CNN
Train on the toy Balloon dataset and implement color splash effect.
Copyright (c) 2018 Matterport, Inc.
Licensed under the MIT License (see LICENSE for details)
Written by Waleed Abdulla
------------------------------------------------------------
Usage: import the module (see Jupyter notebooks for examples), or run from
       the command line as such:
    # Train a new model starting from pre-trained COCO weights
    python3 balloon.py train --dataset=/path/to/balloon/dataset --weights=coco
    # Resume training a model that you had trained earlier
    python3 balloon.py train --dataset=/path/to/balloon/dataset --weights=last
    # Train a new model starting from ImageNet weights
    python3 balloon.py train --dataset=/path/to/balloon/dataset --weights=imagenet
    # Apply color splash to an image
    python3 balloon.py splash --weights=/path/to/weights/file.h5 --image=<URL or path to file>
    # Apply color splash to video using the last weights you trained
    python3 balloon.py splash --weights=last --video=<URL or path to file>
"""

import os
import sys
import glob
import json
import datetime
import numpy as np
import skimage.draw
import cv2

# Root directory of the project
sys.path.append("../../")
ROOT_DIR = os.path.abspath("../../")
sys.path.extend([os.path.join(root, name) for root, dirs, _ in os.walk("../") for name in dirs])

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn.config import Config
from mrcnn import model as modellib, utils

# Path to trained weights file
COCO_WEIGHTS_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")

# Directory to save logs and model checkpoints, if not provided
# through the command line argument --logs
DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "endovislogs")

############################################################
#  Configurations
############################################################


class EndovisConfig(Config):
    """Configuration for training on the toy  dataset.
    Derives from the base Config class and overrides some values.
    """
    # Give the configuration a recognizable name
    NAME = "instruments"

    # We use a GPU with 12GB memory, which can fit two images.
    # Adjust down if you use a smaller GPU.
    IMAGES_PER_GPU = 2

    # Number of classes (including background)
    NUM_CLASSES = 1 + 1  # Background + balloon

    # Number of training steps per epoch
    STEPS_PER_EPOCH = 100

    # Skip detections with < 90% confidence
    DETECTION_MIN_CONFIDENCE = 0.9


############################################################
#  Dataset
############################################################

class EndovisDataset(utils.Dataset):

    def load_dataset(self, dataset_dir):
        """Load a subset of the Balloon dataset.
        dataset_dir: Root directory of the dataset.
        subset: Subset to load: train or val
        """
        # Add classes. We have only one class to add.
        self.add_class("instruments", 1, "instruments")

        # Train or validation dataset?
        imgs = []
        img_id = []
        with open(dataset_dir, 'r') as f:
            lines = f.readlines()
            for line in lines:
                str_list = line.split(' ')
                imgs.append(str_list[0])
                img_id.append(str_list[1])
        print('************Train Image Num is: ',len(imgs),'***********')
        i = 0
        for img in imgs:
            # Add images
            image_path = img
            image = skimage.io.imread(image_path)
            height, width = image.shape[:2]
            # print(height,width)

            # mask_o = skimage.io.imread(image_path.replace('images', 'instrument_instances'))
            # print(mask_o.shape)

            label_list = glob.glob(image_path.replace('images', 'amodel')[:-4]+'*.png')
            polygons = len(label_list)
            # print(polygons)

            self.add_image(
                "instruments",
                image_id=img_id[i],  # use file name as a unique image id
                path=image_path,
                width=width, height=height,
                mask_o = label_list,
                polygons = polygons)
            i+=1

    def load_mask(self, image_id):
        """Generate instance masks for an image.
       Returns:
        masks: A bool array of shape [height, width, instance count] with
            one mask per instance.
        class_ids: a 1D array of class IDs of the instance masks.
        """
        # If not a balloon dataset image, delegate to parent class.
        info = self.image_info[image_id]
        n = info["polygons"]
        # print(n)
        image_info = self.image_info[image_id]
        if image_info["source"] != "instruments":
            return super(self.__class__, self).load_mask(image_id)

        # [height, width, instance_count]
        if n ==0:
            mask = np.zeros([info["height"], info["width"],1],dtype=np.uint8)
        else:
            mask = np.zeros([info["height"], info["width"], n],
                        dtype=np.uint8)

            mask_o = info['mask_o']
            # print(mask_o.shape)
            i=0
            for mask_i in mask_o:
                # mask_n = np.zeros([info["height"], info["width"]], dtype=np.uint8)
                # mask_n[mask_o==i+1] = 1
                mask_n = cv2.imread(mask_i)
                gray_n = cv2.cvtColor(mask_n, cv2.COLOR_BGR2GRAY)
                mask[:,:,i] = gray_n
                i+=1
            # print(mask[:,:,0])

        # Return mask, and array of class IDs of each instance. Since we have
        # one class ID only, we return an array of 1s
        return mask.astype(np.bool), np.ones([mask.shape[-1]], dtype=np.int32)

    def image_reference(self, image_id):
        """Return the path of the image."""
        info = self.image_info[image_id]
        if info["source"] == "instruments":
            return info["path"]
        else:
            super(self.__class__, self).image_reference(image_id)

def train(model):
    """Train the model."""
    # Training dataset.
    dataset_train = EndovisDataset()
    dataset_train.load_dataset(dataset)
    dataset_train.prepare()

    # Validation dataset
    dataset_val = EndovisDataset()
    dataset_val.load_dataset(val_data)
    dataset_val.prepare()


    # *** This training schedule is an example. Update to your needs ***
    # Since we're using a very small dataset, and starting from
    # COCO trained weights, we don't need to train too long. Also,
    # no need to train all layers, just the heads should do it.
    print("Training network heads")
    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE,
                epochs=60,
                layers='heads')

############################################################
#  Training
############################################################

if __name__ == '__main__':
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Train Mask R-CNN to detect balloons.')
    parser.add_argument('--logs', required=False,
                        default=DEFAULT_LOGS_DIR,
                        metavar="/path/to/logs/",
                        help='Logs and checkpoints directory (default=logs/)')
    args = parser.parse_args()

    # For Train
    command = 'train'  # help="'train' or 'splash'"
    # weights = 'last'   # help="Path to weights .h5 file or 'coco'"
    weights = 'coco'
    # weights = '/home/srh/Mask_RCNN/endovislogs/instruments20190911T2015/mask_rcnn_instruments_0030.h5'

    dataset = 'train.txt'
    val_data = 'val.txt'

    # Validate arguments
    assert dataset, "Argument --dataset is required for training"

    print("Weights: ", weights)
    print("Dataset: ", dataset)
    print("Logs: ", args.logs)

    # Configurations
    config = EndovisConfig()

    # Create model
    model = modellib.MaskRCNN(mode="training", config=config,
                                  model_dir=args.logs)

    # Select weights file to load
    if weights.lower() == "coco":
        weights_path = COCO_WEIGHTS_PATH
        # Download weights file
        if not os.path.exists(weights_path):
            utils.download_trained_weights(weights_path)
    elif weights.lower() == "last":
        # Find last trained weights
        weights_path = model.find_last()
    else:
        weights_path = weights

    # Load weights
    if weights.lower() == "coco":
        # Exclude the last layers because they require a matching
        # number of classes
        model.load_weights(weights_path, by_name=True, exclude=[
            "mrcnn_class_logits", "mrcnn_bbox_fc",
            "mrcnn_bbox", "mrcnn_mask"])
    else:
        model.load_weights(weights_path, by_name=True)
    print("********Loading weights ", weights_path,'********')
    
    # Train or evaluate
    train(model)
    
