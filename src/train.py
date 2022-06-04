from mrcnn.model import MaskRCNN

from src.cnn import OfficeObjectDataset, TrainConfig

EPOCHS = 10

train_set = OfficeObjectDataset(train_part=0.8)
train_set.load_dataset('dataset', is_train=True)
train_set.prepare()
print('Train: %d' % len(train_set.image_ids))
test_set = OfficeObjectDataset(train_part=0.8)
test_set.load_dataset('dataset', is_train=False)
test_set.prepare()
print('Test: %d' % len(test_set.image_ids))

cfg = TrainConfig()
model = MaskRCNN(mode='training', model_dir='models/', config=cfg)

model_path = 'mask_rcnn_coco.h5'
model.load_weights(model_path, by_name=True,
                   exclude=["mrcnn_class_logits", "mrcnn_bbox_fc", "mrcnn_bbox", "mrcnn_mask"])
# train weights (output layers or 'heads')
model.train(train_set, test_set, learning_rate=cfg.LEARNING_RATE, epochs=EPOCHS, layers='heads')
