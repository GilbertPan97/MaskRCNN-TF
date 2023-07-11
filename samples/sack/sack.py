"""
Mask R-CNN
Train on the toy Balloon datasets and implement color splash effect.

Copyright (c) 2018 Fanuc, Inc.
Licensed under the MIT License (see LICENSE for details)
Written by Jiabin Pan

------------------------------------------------------------

Usage: import the module (see Jupyter notebooks for examples), or run from
       the command line as such:

    # Train a new model starting from pre-trained COCO weights
    python3 balloon.py train --datasets=/path/to/balloon/datasets --weights=coco

    # Resume training a model that you had trained earlier
    python3 balloon.py train --datasets=/path/to/balloon/datasets --weights=last

    # Train a new model starting from ImageNet weights
    python3 balloon.py train --datasets=/path/to/balloon/datasets --weights=imagenet

    # Apply color splash to an image
    python3 balloon.py splash --weights=/path/to/weights/file.h5 --image=<URL or path to file>

    # Apply color splash to video using the last weights you trained
    python3 balloon.py splash --weights=last --video=<URL or path to file>
"""

import os
import sys
import json
import datetime
import numpy as np
import skimage.draw
import random
from pycocotools.coco import COCO
from mrcnn.model import log
from mrcnn import visualize

import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings(action='ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Set cuda visible devices
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # gpu:0 (cpu always visiable)

# Root directory of the project
ROOT_DIR = os.path.abspath("../../")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn.config import Config
from mrcnn import model as modellib, utils

# Main function work mode("train", "evaluate" or "test")
WORK_MODE = "train"
TRAIN_WITH_COCO = True     # train with last model -> False

# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "models")    # last trained model dir and trained output dir

# Path to relate path
# COCO_DATAS_PATH = os.path.join(ROOT_DIR, "sack", "dataset1")
COCO_DATAS_DIR = os.path.join(ROOT_DIR, "datasets", "sack", "1.2_coco")

# Path to coco pre-trained weights file
COCO_WEIGHTS_PATH = os.path.join(ROOT_DIR, "coco", "mask_rcnn_coco.h5")

# Directory to save logs and model checkpoints, if not provided
# through the command line argument --logs
DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "logs")


############################################################
#  Configurations
############################################################

class SackConfig(Config):
    """Configuration for training on the toy  datasets.
    Derives from the base Config class and overrides some values.
    """
    # Give the configuration a recognizable name
    NAME = "sack"

    # We use a GPU with 12GB memory, which can fit two images.
    # Adjust down if you use a smaller GPU.
    GPU_COUNT = 1
    IMAGES_PER_GPU = 2

    # train epochs
    EPOCHS = 36

    # Number of classes (including background)
    NUM_CLASSES = 1 + 1  # Background + sack

    # Number of training steps per epoch
    STEPS_PER_EPOCH = 100

    # Skip detections with < 90% confidence
    DETECTION_MIN_CONFIDENCE = 0.9


############################################################
#  Dataset class generator
############################################################

class SackDataset(utils.Dataset):

    def load_sack(self, dataset_dir, subset):
        """Load a subset of the Sack datasets.
        dataset_dir: Root directory of the datasets.
        subset: Subset to load: train or val
        """
        # Add classes. We have only one class to add.
        self.add_class("sack", 1, "sack")

        # Define images and annotation dirs
        annos_dir = os.path.join(dataset_dir, "annotations")
        imgs_dir = os.path.join(dataset_dir, "images")

        # Train or validation datasets?
        assert subset in ["train", "val"]
        used_anno = os.path.join(annos_dir, subset+".json")

        # Load coco annotations (coco2.0)
        coco = COCO(annotation_file=used_anno)

        # get all image index info (ids+1=img_id)
        ids = list(sorted(coco.imgs.keys()))

        # get all coco class labels
        coco_classes = dict([(v["id"], v["name"]) for k, v in coco.cats.items()])

        for img_id in ids:
            # get annotation idx according to img idx
            ann_ids = coco.getAnnIds(imgIds=img_id)

            # get all annotation info according to annotations idx
            targets = coco.loadAnns(ann_ids)

            # get image file path
            img_info = coco.loadImgs(img_id)
            img_path = os.path.join(imgs_dir, img_info[0]['file_name'])

            # reshape polygons
            polygons = []
            for target in targets:
                # get category name
                cat_name = coco_classes[target["category_id"]]

                # shape polygons (map segmentation to all_points_x and all_points_y)
                segs = target['segmentation']
                all_points_x, all_points_y = segs[0][0::2], segs[0][1::2]
                polygon = ({'name': 'polygon',
                            'all_points_x': all_points_x,
                            'all_points_y': all_points_y})
                polygons.append(polygon)

            self.add_image(
                "sack",
                image_id=img_info[0]['file_name'],  # use file name as a unique image id
                path=img_path,
                width=img_info[0]['width'], height=img_info[0]['height'],
                polygons=polygons)

    def load_mask(self, image_id):
        """Generate instance masks for an image.
       Returns:
        masks: A bool array of shape [height, width, instance count] with
            one mask per instance.
        class_ids: a 1D array of class IDs of the instance masks.
        """
        # If not a balloon datasets image, delegate to parent class.
        image_info = self.image_info[image_id]
        if image_info["source"] != "sack":
            return super(self.__class__, self).load_mask(image_id)

        # Convert polygons to a bitmap mask of shape
        # [height, width, instance_count]
        info = self.image_info[image_id]
        mask = np.zeros([info["height"], info["width"], len(info["polygons"])],
                        dtype=np.uint8)
        for i, p in enumerate(info["polygons"]):
            # Get indexes of pixels inside the polygon and set them to 1
            rr, cc = skimage.draw.polygon(p['all_points_y'], p['all_points_x'])
            mask[rr, cc, i] = 1

        # Return mask, and array of class IDs of each instance. Since we have
        # one class ID only, we return an array of 1s
        return mask.astype(np.bool), np.ones([mask.shape[-1]], dtype=np.int32)

    def image_reference(self, image_id):
        """Return the path of the image."""
        info = self.image_info[image_id]
        if info["source"] == "balloon":
            return info["path"]
        else:
            super(self.__class__, self).image_reference(image_id)


############################################################
#  Train, evaluate and test model
############################################################

def train(model_path, train_config, dataset_dir):
    """Train the model."""
    print("==================================================")
    print("Loading weights from: ", model_path)
    print("Loading coco datasets from: ", dataset_dir)
    print("==================================================")
    # Create model
    model = modellib.MaskRCNN(mode="training", config=train_config,
                              model_dir=DEFAULT_LOGS_DIR)

    # Download weights file
    if not os.path.exists(model_path):
        utils.download_trained_weights(model_path)

    # Exclude the last layers because they require a matching number of classes
    model.load_weights(model_path, by_name=True, exclude=[
        "mrcnn_class_logits", "mrcnn_bbox_fc",
        "mrcnn_bbox", "mrcnn_mask"])

    # Training datasets.
    dataset_train = SackDataset()
    dataset_train.load_sack(dataset_dir, "train")
    dataset_train.prepare()

    # Validation datasets
    dataset_val = SackDataset()
    dataset_val.load_sack(dataset_dir, "val")
    dataset_val.prepare()

    # Train the head branches
    # Passing layers="heads" freezes all layers except the head
    # layers. You can also pass a regular expression to select
    # which layers to train by name pattern.
    print("Training network heads")
    model.train(dataset_train, dataset_val,
                learning_rate=train_config.LEARNING_RATE,
                epochs=train_config.EPOCHS,
                layers='heads')

    # Fine tune all layers
    # Passing layers="all" trains all layers. You can also
    # pass a regular expression to select which layers to
    # train by name pattern.
    model.train(dataset_train, dataset_val,
                learning_rate=train_config.LEARNING_RATE / 10,
                epochs=train_config.EPOCHS,
                layers="all")

    # Save weights
    # Typically not needed because callbacks save after every epoch
    # Uncomment to save manually
    tmp_path = os.path.join(MODEL_DIR, "mask_rcnn_sack.h5")
    if os.path.exists(tmp_path):
        new_model_name = "mask_rcnn_sack_{:%Y%m%dT%H%M%S}.h5".format(datetime.datetime.now())
        model_path_out = os.path.join(MODEL_DIR, new_model_name)
        model.keras_model.save_weights(model_path_out)
    else:
        model.keras_model.save_weights(tmp_path)


def evaluate(model_path, inference_config, dataset_dir):
    """Evaluate the model."""
    # Recreate the model in inference mode
    model = modellib.MaskRCNN(mode="inference", config=inference_config,
                              model_dir=MODEL_DIR)

    # Load trained weights
    print("==================================================")
    print("Loading evaluate model weights from: ", model_path)
    print("==================================================")
    model.load_weights(model_path, by_name=True)

    # get validation datasets
    dataset_val = SackDataset()
    dataset_val.load_sack(dataset_dir, "val")
    dataset_val.prepare()

    # Evaluation
    # Compute VOC-Style mAP @ IoU=0.5
    # Running on 5 images. Increase for better accuracy.
    image_ids = np.random.choice(dataset_val.image_ids, 5)
    APs = []
    for image_id in image_ids:
        # Load image and ground truth data
        image, image_meta, gt_class_id, gt_bbox, gt_mask = \
            modellib.load_image_gt(dataset_val, inference_config, image_id)
        molded_images = np.expand_dims(modellib.mold_image(image, inference_config), 0)
        # Run object detection
        results = model.detect([image], verbose=0)
        r = results[0]
        # Compute AP
        AP, precisions, recalls, overlaps = \
            utils.compute_ap(gt_bbox, gt_class_id, gt_mask,
                             r["rois"], r["class_ids"], r["scores"], r['masks'])
        APs.append(AP)

    print("mAP: ", np.mean(APs))


def test(model_path, inference_config, dataset_dir):
    """Test the model."""
    # Recreate the model in inference mode
    model = modellib.MaskRCNN(mode="inference", config=inference_config,
                              model_dir=MODEL_DIR)

    # Load trained weights
    print("==================================================")
    print("Loading test model weights from: ", model_path)
    print("==================================================")
    model.load_weights(model_path, by_name=True)

    # get validation datasets
    dataset_val = SackDataset()
    dataset_val.load_sack(dataset_dir, "val")
    dataset_val.prepare()

    # get random image data
    image_id = random.choice(dataset_val.image_ids)
    original_image, image_meta, gt_class_id, gt_bbox, gt_mask = \
        modellib.load_image_gt(dataset_val, inference_config, image_id)

    # log original image annotation info
    log("original_image", original_image)
    log("image_meta", image_meta)
    log("gt_class_id", gt_class_id)
    log("gt_bbox", gt_bbox)
    log("gt_mask", gt_mask)

    # visualize annotation info (image with mask)
    # visualize.display_instances(original_image, gt_bbox, gt_mask, gt_class_id,
    #                             dataset_val.class_names, figsize=(8, 8))

    # get model inference result
    results = model.detect([original_image], verbose=1)

    r = results[0]
    visualize.display_instances(original_image, r['rois'], r['masks'], r['class_ids'],
                                dataset_val.class_names, r['scores'])


############################################################
#  Main function
############################################################
if __name__ == '__main__':

    command = WORK_MODE
    assert command, "Main function work mode is:" + command

    print("Weights: ", COCO_WEIGHTS_PATH)
    print("Dataset: ", COCO_DATAS_DIR)
    print("Logs: ", DEFAULT_LOGS_DIR)

    # Configurations ({train}: config->SackConfig() or {evaluate, test}: config->InferenceConfig())
    if command == "train":
        config = SackConfig()
    else:
        class InferenceConfig(SackConfig):
            # Set batch size to 1 since we'll be running inference on
            # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
            GPU_COUNT = 1
            IMAGES_PER_GPU = 1
            USE_MINI_MASK = False
        config = InferenceConfig()
    config.display()

    # view training device
    from tensorflow.python.client import device_lib
    print(device_lib.list_local_devices())

    # get newest modified weight file from MODEL_DIR
    lists = os.listdir(MODEL_DIR)
    lists.sort(key=lambda x: os.path.getmtime(MODEL_DIR + '\\' + x))
    model_new_path = []
    try:
        model_new_path = os.path.join(MODEL_DIR, lists[-1])    # newest modified model file path
    except:
        warnings.warn('Model weight file not exist in MODEL_DIR.')

    # Train or evaluate
    if command == "train":
        if TRAIN_WITH_COCO:
            train(COCO_WEIGHTS_PATH, config, COCO_DATAS_DIR)
        else:
            # Get path to newest saved weight
            train(model_new_path, config, COCO_DATAS_DIR)
    elif command == "evaluate":
        evaluate(model_new_path, config, COCO_DATAS_DIR)
    else:
        test(model_new_path, config, COCO_DATAS_DIR)





