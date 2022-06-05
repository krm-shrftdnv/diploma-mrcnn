from os import listdir
from xml.etree import ElementTree

import numpy as np
import skimage
from PIL import Image
from numpy import asarray
from numpy import zeros

from mrcnn.config import Config
from mrcnn.utils import Dataset


class TrainConfig(Config):
    NAME = "train"
    NUM_CLASSES = 1 + 2
    GPU_COUNT = 1
    STEPS_PER_EPOCH = 10
    IMAGES_PER_GPU = 1


class InferenceConfig(Config):
    NAME = "inference"
    NUM_CLASSES = 1 + 2
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1


def get_image_sizes(path):
    im = Image.open(path)
    return im.width, im.height


def extract_boxes_polygon(filename, obj_type):
    root = ElementTree.parse(filename)
    boxes = list()
    for obj in root.findall('.//object'):
        if obj.find('type').text == obj_type:
            box = list()
            for point in obj.find('bndbox').findall('point'):
                x = int(point.find('x').text)
                y = int(point.find('y').text)
                box.append((x, y))
            boxes.append(box)
    path = root.find('path').text
    width, height = get_image_sizes(path)
    return boxes, width, height


class OfficeObjectDataset(Dataset):
    DOOR_OBJ_TYPE = 'door'
    PLATE_OBJ_TYPE = 'plate'

    OBJ_TYPES = [
        DOOR_OBJ_TYPE,
        PLATE_OBJ_TYPE,
    ]

    def __init__(self, train_part, class_map=None):
        super().__init__(class_map=class_map)
        self.train_part = train_part

    def load_dataset(self, dataset_dir, is_train=True):
        for id, obj_type in enumerate(self.OBJ_TYPES):
            self.add_class('dataset', id + 1, obj_type)
        images_dir = dataset_dir + '/images/'
        annotations_dir = dataset_dir + '/annotations/'
        train_part_size = int(self.train_part * len(listdir(images_dir)))
        for filename in listdir(images_dir):
            image_id = filename[:-4]
            if is_train and int(image_id) >= train_part_size:
                continue
            if not is_train and int(image_id) < train_part_size:
                continue
            img_path = images_dir + filename
            ann_path = annotations_dir + image_id + '.xml'
            self.add_image('dataset', image_id=image_id, path=img_path, annotation=ann_path)

    def load_mask(self, image_id):
        info = self.image_info[image_id]
        filepath = info['annotation']
        class_ids = list()
        total_boxes = []
        width = 0
        height = 0
        for obj_type in self.OBJ_TYPES:
            boxes, width, height = extract_boxes_polygon(filepath, obj_type)
            wrapped_boxes = map(lambda b: {'box': b, 'obj_type': obj_type}, boxes)
            total_boxes.extend(wrapped_boxes)
        mask = zeros([height, width, len(total_boxes)], dtype='uint8')
        for i, box in enumerate(total_boxes):
            xs = []
            ys = []
            for point in box['box']:
                xs.append(point[0])
                ys.append(point[1])
            rr, cc = skimage.draw.polygon(ys, xs)
            mask[rr, cc, i] = 1
            class_ids.append(self.class_names.index(total_boxes[i]['obj_type']))
        return mask.astype(np.bool), asarray(class_ids, dtype='int32')

    def image_reference(self, image_id):
        info = self.image_info[image_id]
        return info['path']
