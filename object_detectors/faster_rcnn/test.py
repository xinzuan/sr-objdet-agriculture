import os
from detectron2.data.datasets import register_coco_instances
from detectron2.engine import DefaultTrainer
from detectron2 import model_zoo
from detectron2.data.datasets import register_coco_instances
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.utils.visualizer import Visualizer
from detectron2.config import get_cfg
from detectron2.engine.defaults import DefaultPredictor
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader

dataset_root = '../../agml-base/apple_detection_drone_brazil'
register_coco_instances('apple_detection_drone_brazil_train', {}, f'{dataset_root}/train/annotations.json', f'{dataset_root}/train/images')
register_coco_instances('apple_detection_drone_brazil_val', {}, f'{dataset_root}/valid/annotations.json', f'{dataset_root}/valid/images')
register_coco_instances('apple_detection_drone_brazil_test', {}, f'{dataset_root}/test/annotations.json', f'{dataset_root}/test/images')

cfg = get_cfg()
cfg.merge_from_file('configs/COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml')
cfg.DATASETS.TRAIN = ('apple_detection_drone_brazil_train',)
cfg.DATASETS.TEST = ('apple_detection_drone_brazil_val',)
cfg.DATALOADER.NUM_WORKERS = 2
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url('COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml')  # Let training initialize from model zoo
cfg.SOLVER.IMS_PER_BATCH = 8  # This is the real "batch size" commonly known to deep learning people
cfg.SOLVER.BASE_LR = 0.00025  # pick a good LR
cfg.SOLVER.MAX_ITER = 2500    # 300 iterations seems good enough for this toy dataset; you will need to train longer for a practical dataset
cfg.SOLVER.STEPS = []        # do not decay learning rate
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 256   # The "RoIHead batch size". 128 is faster, and good enough for this toy dataset (default: 512)
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # only has one class (ballon). (see https://detectron2.readthedocs.io/tutorials/datasets.html#update-the-config-for-new-datasets)
# NOTE: this config means the number of classes, but a few popular unofficial tutorials incorrect uses num_classes+1 here.
cfg.MODEL.WEIGHTS = 'runs/train/r101fpn3x/model_final.pth'  # path to the model we just trained
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7   # set a custom testing threshold

predictor = DefaultPredictor(cfg)
evaluator = COCOEvaluator('apple_detection_drone_brazil_test', output_dir='runs/eval/r101fpn3x')
val_loader = build_detection_test_loader(cfg, 'apple_det_test')
print(inference_on_dataset(predictor.model, val_loader, evaluator))