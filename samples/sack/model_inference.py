import os
import sys
import json
import time
import cv2
import natsort
import datetime
import numpy as np
import skimage.draw
import matplotlib.pyplot as plt
from matplotlib import patches,  lines
from mrcnn import visualize

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

# images inference dir
INFERENCE_DIR = os.path.join(ROOT_DIR, "inference")

# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "models")    # last trained model dir and trained output dir

# Path to load model
MODEL_PATH = os.path.join(ROOT_DIR, "models", "mask_rcnn_sack_20230116T160948.h5")


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

    # Number of classes (including background)
    NUM_CLASSES = 1 + 1  # Background + balloon

    # Number of training steps per epoch
    STEPS_PER_EPOCH = 100

    # Skip detections with < 90% confidence
    DETECTION_MIN_CONFIDENCE = 0.9


def color_splash_and_save(image, boxes, masks, class_ids, class_names, scores, colors=None, save_path=None):
    """Apply color splash effect.
    image: RGB image [height, width, 3]
    mask: instance segmentation mask [height, width, instance count]

    Returns result image.
    """
    # Number of instances
    N = boxes.shape[0]
    if not N:
        print("\n*** No instances to display *** \n")
    else:
        assert boxes.shape[0] == masks.shape[-1] == class_ids.shape[0]

    # If no axis is passed, create one and automatically call show()
    _, ax = plt.subplots(1, figsize=(16, 16))

    # Generate random colors
    colors = colors or visualize.random_colors(N)

    # Show area outside image boundaries.
    height, width = image.shape[:2]
    ax.set_ylim(height + 10, -10)
    ax.set_xlim(-10, width + 10)
    ax.axis('off')
    ax.set_title("Predict image")

    masked_image = image.astype(np.uint32).copy()
    for i in range(N):
        color = colors[i]

        # Bounding box
        if not np.any(boxes[i]):
            # Skip this instance. Has no bbox. Likely lost in image cropping.
            continue
        y1, x1, y2, x2 = boxes[i]

        p = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2,
                            alpha=0.7, linestyle="dashed",
                            edgecolor=color, facecolor='none')
        ax.add_patch(p)

        # Label
        class_id = class_ids[i]
        score = scores[i] if scores is not None else None
        label = class_names[class_id]
        caption = "{} {:.3f}".format(label, score) if score else label

        ax.text(x1, y1 + 8, caption,
                color='w', size=11, backgroundcolor="none")

        # Mask
        mask = masks[:, :, i]
        masked_image = visualize.apply_mask(masked_image, mask, color)

        # Mask Polygon
        # Pad to ensure proper polygons for masks that touch image edges.
        padded_mask = np.zeros(
            (mask.shape[0] + 2, mask.shape[1] + 2), dtype=np.uint8)
        padded_mask[1:-1, 1:-1] = mask
        contours = visualize.find_contours(padded_mask, 0.5)
        for verts in contours:
            # Subtract the padding and flip (y, x) to (x, y)
            verts = np.fliplr(verts) - 1
            p = visualize.Polygon(verts, facecolor="none", edgecolor=color)
            ax.add_patch(p)
    ax.imshow(masked_image.astype(np.uint8))

    # save predict image (just save one result)
    if not save_path:
        print("Result save path not know.")
    plt.savefig(save_path)

    # close all figures
    plt.close('all')


def detect_and_color_splash(model, config, inf_dir):
    assert inf_dir

    ori_imgs_dir = os.path.join(inf_dir, "images")
    out_result_dir = os.path.join(inf_dir, "result")

    if not os.path.exists(inf_dir):
        os.makedirs(inf_dir)
    if not os.path.exists(out_result_dir):
        os.makedirs(out_result_dir)

    if inf_dir:
        # get dir images name list
        imgs_list = os.listdir(ori_imgs_dir)
        imgs_list = natsort.natsorted(imgs_list)

        for img_name in imgs_list:
            # Run model detection and generate the color splash effect
            print("Running on image: {}.".format(img_name))

            # get image path
            img_path = os.path.join(ori_imgs_dir, img_name)

            # Read image
            ori_image = skimage.io.imread(img_path)

            # image_in, window, scale, padding, crop = utils.resize_image(
            #     ori_image,
            #     min_dim=config.IMAGE_MIN_DIM,
            #     min_scale=config.IMAGE_MIN_SCALE,
            #     max_dim=config.IMAGE_MAX_DIM,
            #     mode=config.IMAGE_RESIZE_MODE)

            # Detect objects
            start_time = time.time()
            results = model.detect([ori_image], verbose=1)
            r = results[0]
            end_time = time.time()
            print("Inference time: {}s".format(end_time-start_time))

            # Save output
            class_names = ["background", "sack"]
            img_path_out = os.path.join(out_result_dir, img_name)
            color_splash_and_save(ori_image, r['rois'], r['masks'], r['class_ids'],
                                  class_names, scores=r['scores'], save_path=img_path_out)

            # visualize.display_instances(ori_image, r['rois'], r['masks'], r['class_ids'], "sack", r['scores'])

        print("Saved predict image to: {}", out_result_dir)


if __name__ == '__main__':

    class InferenceConfig(SackConfig):
        # Set batch size to 1 since we'll be running inference on
        # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
        GPU_COUNT = 1
        IMAGES_PER_GPU = 1
    config = InferenceConfig()

    model = modellib.MaskRCNN(mode="inference", config=config,
                              model_dir=MODEL_DIR)

    # Load trained weights
    print("==================================================")
    print("Loading test model weights from: ", MODEL_PATH)
    print("==================================================")
    model.load_weights(MODEL_PATH, by_name=True)

    detect_and_color_splash(model, config, inf_dir=INFERENCE_DIR)
