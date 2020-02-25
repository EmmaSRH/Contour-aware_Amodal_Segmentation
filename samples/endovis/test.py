import os
import sys
import numpy as np
import glob
from imageio import imread
from imageio import imsave
from evaluate import testEachimage_for_test
# Root directory of the project
ROOT_DIR = os.path.abspath("../../")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn.config import Config
from mrcnn import model as modellib, utils

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

def detect_and_color_splash(model, image_path=None, out_path=None):
    out_path = out_path

    # Run model detection and generate the color splash effect
    print("Running on {}".format(image_path))
    # Read image
    image = imread(image_path)
    # Detect objects
    r = model.detect([image], verbose=1)[0]

    # Save output
    file_name1 = image_path.split('/')[-1][:-4]  # img id
    file_name2 = image_path.split('/')[-3]  # instrument id

    out_ = out_path + '/' + file_name2 + '/'

    # if not os._exists(out_):
    #     os.makedirs(out_)

    file_name = out_ + file_name1

    # if r['masks'].shape[-1] > 0:
    #     # We're treating all instances as one, so collapse the mask into one layer
    #     splash = np.zeros((r['masks'].shape[0], r['masks'].shape[1])).astype(np.uint8)
    #     for i in range(r['masks'].shape[-1]):
    #         splash[r['masks'][:, :, i] != False] = i + 1
    # else:
    #     splash = np.zeros((r['masks'].shape[0], r['masks'].shape[1])).astype(np.uint8)


    # draw output img
    pre_mask = []
    if r['masks'].shape[-1] > 0:
        for i in range(r['masks'].shape[-1]):
            splash = np.zeros((r['masks'].shape[0], r['masks'].shape[1])).astype(np.uint8)
            mask = r['masks'][:, :, i]
            splash[mask != False] = 255
            # pre_mask.append(splash)
            imsave(file_name + '_ins_' + str(i) + '.png', splash)
    else:
        splash = np.zeros((r['masks'].shape[0], r['masks'].shape[1])).astype(np.uint8)
        # pre_mask.append(splash)
        imsave(file_name + '_ins_0.png', splash)

    # gt_path = '/data/srh/Amodel_Data/'+image_path.split('/')[-4]+'/'
    # file_name1 = img.split('/')[-1][:-4]  # img id
    # file_name2 = img.split('/')[-3]  # instrument id
    # gt_list = glob.glob(gt_path + file_name2 + '/amodel/' + file_name1 + '*.png')
    # mAP, _ = testEachimage_for_test(pre_mask, gt_list)


    # return mAP



############################################################
#  Test
############################################################

if __name__ == '__main__':

    imgs_train = glob.glob('/data/srh/Amodel_Data/train/*/images/*.jpg')
    print('----------imgs_train lens is: ', len(imgs_train), '------------')
    imgs_test = glob.glob('/data/srh/Amodel_Data/test/*/images/*.png')
    print('----------imgs_test lens is: ', len(imgs_test), '------------')



    weights = os.path.join(ROOT_DIR, 'endovislogs/instruments20200210T2237/mask_rcnn_instruments_0050.h5')

    # Configurations
    class InferenceConfig(EndovisConfig):
        GPU_COUNT = 1
        IMAGES_PER_GPU = 1
    config = InferenceConfig()

    # Create model
    model = modellib.MaskRCNN(mode="inference", config=config,model_dir=ROOT_DIR)
    # Load weights
    model.load_weights(weights, by_name=True)

    # Train or evaluate
    all_train_ap, all_test_ap = 0, 0
    for img in imgs_train:
        detect_and_color_splash(model, image_path=img, out_path=ROOT_DIR+'/output_2')
        # all_train_ap += mAP

    for img in imgs_test:
        # if img.split('/')[-3] == 'instrument9':
            # break
        mAP = detect_and_color_splash(model, image_path=img, out_path=ROOT_DIR+'/output_2')
        # all_test_ap += mAP

    # train_ap = all_train_ap/len(imgs_train)
    # test_num = len(imgs_test)-300
    # test_ap = all_test_ap/test_num

    # with open('ap_for_each_weight.txt','a') as f:
    #     f.writelines('weight45: '+train_ap +' '+ test_ap + '\n')





