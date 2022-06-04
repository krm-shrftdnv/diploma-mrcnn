import random

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
from numpy import expand_dims
from numpy import mean

import mrcnn.model as modellib
from mrcnn import utils
from mrcnn import visualize
from mrcnn.model import load_image_gt, MaskRCNN
from mrcnn.model import log
from mrcnn.model import mold_image
from mrcnn.utils import compute_ap
from src.cnn import OfficeObjectDataset, InferenceConfig

# MODEL_PATH = 'mask_rcnn_train_0010.h5'
MODEL_PATH = 'models/train20220604T1650/mask_rcnn_train_0001.h5'


def plot_actual_vs_predicted(dataset, model, cfg, n_images=5):
    for i in range(n_images):
        image = dataset.load_image(i)
        mask, _ = dataset.load_mask(i)
        scaled_image = mold_image([image], cfg)
        sample = expand_dims(scaled_image, 0)
        yhat = model.detect(sample, verbose=0)[0]
        plt.subplot(n_images, 2, i * 2 + 1)
        plt.imshow(image)
        plt.title('Actual')

        for j in range(mask.shape[2]):
            plt.imshow(mask[:, :, j], cmap='gray', alpha=0.3)

        plt.subplot(n_images, 2, i * 2 + 2)
        plt.imshow(image)
        plt.title('Predicted')
        ax = plt.gca()
        for box in yhat['rois']:
            y1, x1, y2, x2 = box
            width, height = x2 - x1, y2 - y1
            rect = Rectangle((x1, y1), width, height, fill=False, color='red')
            ax.add_patch(rect)
    # show the figure
    plt.show()


# calculate the mAP for a model on a given dataset
def evaluate_model(dataset, model, cfg):
    APs = list()
    for image_id in dataset.image_ids:
        # load image, bounding boxes and masks for the image id
        image, image_meta, gt_class_id, gt_bbox, gt_mask = load_image_gt(dataset, cfg, image_id, use_mini_mask=False)
        # convert pixel values (e.g. center)
        scaled_image = mold_image(image, cfg)
        # convert image into one sample
        sample = expand_dims(scaled_image, 0)
        # make prediction
        yhat = model.detect(sample, verbose=0)
        # extract results for first sample
        r = yhat[0]
        # calculate statistics, including AP
        AP, _, _, _ = compute_ap(gt_bbox, gt_class_id, gt_mask, r["rois"], r["class_ids"], r["scores"], r['masks'])
        # store
        APs.append(AP)
    # calculate the mean AP across all images
    mAP = mean(APs)
    return mAP


def get_ax(rows=1, cols=1, size=16):
    _, ax = plt.subplots(rows, cols, figsize=(size * cols, size * rows))
    return ax


# load the test dataset
test_set = OfficeObjectDataset(train_part=0.8)
test_set.load_dataset('dataset', is_train=False)
test_set.prepare()
print('Test: %d' % len(test_set.image_ids))

cfg = InferenceConfig()
cfg.display()
model = MaskRCNN(mode='inference', model_dir='models/', config=cfg)

model.load_weights(MODEL_PATH, by_name=True)

image_id = random.choice(test_set.image_ids)
image, image_meta, gt_class_id, gt_bbox, gt_mask = \
    modellib.load_image_gt(test_set, cfg, image_id, use_mini_mask=False)
info = test_set.image_info[image_id]
print("image ID: {}.{} ({}) {}".format(info["source"], info["id"], image_id,
                                       test_set.image_reference(image_id)))
print("Original image shape: ", modellib.parse_image_meta(image_meta[np.newaxis, ...])["original_image_shape"][0])

# Run object detection
results = model.detect_molded(np.expand_dims(image, 0), np.expand_dims(image_meta, 0), verbose=1)

# Display results
r = results[0]
log("gt_class_id", gt_class_id)
log("gt_bbox", gt_bbox)
log("gt_mask", gt_mask)

# Compute AP over range 0.5 to 0.95 and print it
utils.compute_ap_range(gt_bbox, gt_class_id, gt_mask,
                       r['rois'], r['class_ids'], r['scores'], r['masks'],
                       verbose=1)

visualize.display_differences(
    image,
    gt_bbox, gt_class_id, gt_mask,
    r['rois'], r['class_ids'], r['scores'], r['masks'],
    test_set.class_names,
    show_box=False, show_mask=False,
    iou_threshold=0.1, score_threshold=0.1)
