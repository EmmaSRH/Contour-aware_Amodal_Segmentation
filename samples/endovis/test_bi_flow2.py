import os
import sys
import json
import datetime
import numpy as np
import skimage.draw
import glob
from imageio import imread

# Root directory of the project
ROOT_DIR = os.path.abspath("../../")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn.config import Config
from mrcnn import model as modellib, utils
# from evaluation.test.test_instance_dice import TestDiceCalculation as get_dice_score
from evaluation.dice_calculations import compute_dice_coefficient, compute_dice_coefficient_per_instance


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


def color_splash(image, mask):
    """Apply color splash effect.
    image: RGB image [height, width, 3]
    mask: instance segmentation mask [height, width, instance count]
    Returns result image.
    """
    # for show
    # # Make a grayscale copy of the image. The grayscale copy still
    # # has 3 RGB channels, though.
    gray = skimage.color.gray2rgb(skimage.color.rgb2gray(image)) * 255
    # We're treating all instances as one, so collapse the mask into one layer
    mask = (np.sum(mask, -1, keepdims=True) >= 1)
    mask_bi = np.zeros((mask.shape[0],mask.shape[1])).astype(np.uint8)
    mask_bi[mask[:, :, 0] != False] = 1

    # Copy color pixels from the original color image where mask is set
    if mask.shape[0] > 0:
        splash = np.where(mask, image, gray).astype(np.uint8)
    else:
        splash = gray
    return splash, mask_bi

    # # for submit
    # if mask.shape[-1] > 0:
    #     # We're treating all instances as one, so collapse the mask into one layer
    #
    #     mask = (np.sum(mask, -1, keepdims=True) >= 1)
    #     splash = np.zeros((mask.shape[0],mask.shape[1])).astype(np.uint8)
    #     splash[mask[:,:,0] != False ] = 1
    # else:
    #     splash = np.zeros((mask.shape[0],mask.shape[1])).astype(np.uint8)
    # return splash

def detect_and_color_splash(model, image_path=None, video_path=None, out_path=None):
    assert image_path or video_path
    out_path = out_path

    # Run model detection and generate the color splash effect
    print("Running on {}".format(image_path))
    # Read image
    image = skimage.io.imread(image_path)
    # Detect objects
    r = model.detect([image], verbose=1)[0]

    #--------- for show -----------#
    splash,mask_bi = color_splash(image, r['masks'])

    # get dice score
    x_path = image_path.replace('raw', 'instrument_instances').replace('img', 'gt')
    gt = imread(x_path)
    gt[gt < 0.5] = 0
    gt[gt >= 0.5] = 1
    dice = compute_dice_coefficient(gt, mask_bi)
    print(dice)
    with open('60_flow_dice_sorce.txt', 'a+') as f:
        f.write(x_path.split('/')[-1] + ':' + str(dice) + '\n')

    # Save output
    file_name = out_path+image_path.split('/')[-1]
    skimage.io.imsave(file_name, splash)

    # #--------- for submit -----------#
    # _, mask_bi = color_splash(image, r['masks'])
    # # Save output
    # file_name = out_path + '60_bi_result/' + image_path.split('/')[-1]
    # skimage.io.imsave(file_name, mask_bi)
    

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
    parser.add_argument('--input_path', required=False,
                        default='/data/srh/Training/image/',
                        help='where to load input data')
    parser.add_argument('--output_path', required=False,
                        default='50_flow_bi_results/',
                        help='where to load output data')
    args = parser.parse_args()

    # For test
    command = 'splash'
    weights = '/home/srh/Mask_RCNN/flow_endovislogs/instruments20190911T2015/mask_rcnn_instruments_0050.h5'

    imgs = glob.glob(args.input_path+'*.png')

    # for submit
    imgs = glob.glob(args.input_path+'*/*/*/raw.png') #Stage_1/Prokto/1/1500/raw.png

    # Configurations
    class InferenceConfig(EndovisConfig):
        # Set batch size to 1 since we'll be running inference on
        # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
        GPU_COUNT = 1
        IMAGES_PER_GPU = 1
    config = InferenceConfig()
    # config.display()

    # Create model
    model = modellib.MaskRCNN(mode="inference", config=config,
                                  model_dir=args.logs)
    # Load weights
    model.load_weights(weights, by_name=True)
    print("********Loading weights ", weights,'********')
    
    # Train or evaluate
    for img in imgs:
        detect_and_color_splash(model, image_path=img, out_path=args.output_path)
    # detect_and_color_splash(model, image_path="/home/srh/Training/imgs/Prokto_2_84000_raw.png")

