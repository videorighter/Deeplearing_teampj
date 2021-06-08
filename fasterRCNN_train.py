# -*- coding: utf-8 -*-
'''
## Train & Validation Faster-RCNN model
videorighter@ds.seoultech.ac.kr
https://github.com/videorighter
'''

from detectron2.engine import DefaultPredictor, DefaultTrainer
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer, ColorMode
from detectron2.utils.logger import setup_logger
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader
import os, cv2
import time
import utils
import shutil
import argparse

setup_logger()





def train(args):
    if args.init_train:
        if os.path.isdir('/home/videorighter/Deeplearning_teampj/output'):
            try:
                shutil.rmtree('/home/videorighter/Deeplearning_teampj/output')
            except OSError as e:
                print(f"Error: {e.strerror}")

        if os.path.isdir('/home/videorighter/Deeplearning_teampj/val_result'):
            try:
                shutil.rmtree('/home/videorighter/Deeplearning_teampj/val_result')
            except OSError as e:
                print(f"Error: {e.strerror}")

        if os.path.isdir('/home/videorighter/Deeplearning_teampj/output_val'):
            try:
                shutil.rmtree('/home/videorighter/Deeplearning_teampj/output_val')
            except OSError as e:
                print(f"Error: {e.strerror}")

    start = time.time()

    # training model
    cfg = get_cfg()
    cfg.merge_from_file(
        "detectron2/configs/COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml")
    cfg.DATASETS.TRAIN = ("food_train",)
    cfg.DATASETS.TEST = ("food_val",)
    cfg.DATALOADER.NUM_WORKERS = args.num_workers
    cfg.MODEL.MASK_ON = args.mask_on
    cfg.SOLVER.IMS_PER_BATCH = args.ims_per_batch
    cfg.SOLVER.BASE_LR = args.lr
    cfg.SOLVER.MAX_ITER = args.max_iter
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = args.batch_per_image
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = args.num_classes  # 1 classes ('food')

    # make output directory and save output files
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    trainer = DefaultTrainer(cfg)

    # 이전에 학습시킨 pth파일로 resume할 것인지 여부
    trainer.resume_or_load(resume=True)
    trainer.train()

    print("Training running time: ", time.time() - start)

    return cfg, trainer


def validation(cfg, trainer, args):

    start = time.time()
    ################################# model validation ####################################
    cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
    predictor = DefaultPredictor(cfg)

    ############################# validation print ##################################
    train_metadata, val_metadata = utils.metadata("/home/videorighter/Deeplearning_teampj/data/annotations/")

    dataset_dicts_val = utils.get_facelip_dtcs('val',
                                               "/home/videorighter/Deeplearning_teampj/data/annotations/val")
    threshold = args.threshold

    for d in dataset_dicts_val:
        im = cv2.imread(d["file_name"])
        outputs = predictor(im)
        filtered_predictions = utils.filter_predictions_from_outputs(outputs,
                                                                     threshold=threshold,
                                                                     verbose=False)
        v = Visualizer(im[:, :, ::-1],
                       metadata=val_metadata,
                       scale=1)
        v = v.draw_instance_predictions(filtered_predictions)
        img = v.get_image()[:, :, ::-1]
        if not os.path.isdir('val_result'):
            os.mkdir('val_result')
        cv2.imwrite(os.path.join("/home/videorighter/Deeplearning_teampj/val_result", os.path.split(d["file_name"])[1]),
                    img)

    ############################# validation score ###################################

    evaluator = COCOEvaluator("food_val", cfg, False, output_dir="./output_val/")
    val_loader = build_detection_test_loader(cfg, "food_val")
    print(inference_on_dataset(trainer.model, val_loader, evaluator))
    print("Validation running time: ", time.time() - start)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--init_train', type=bool, default=False, help='Initiate train')
    parser.add_argument('--threshold', type=float, default=0.7, help='Result threshold')
    parser.add_argument('--num_workers', type=int, default=2, help='Number of workers')
    parser.add_argument('--mask_on', type=bool, default=False, help='Mask on True/False')
    parser.add_argument('--ims_per_batch', type=int, default=2, help='cfg.SOLVER.IMS_PER_BATCH')
    parser.add_argument('--lr', type=float, default=0.0001, help='cfg.SOLVER.BASE_LR')
    parser.add_argument('--max_iter', type=int, default=4000, help='cfg.SOLVER.MAX_ITER')
    parser.add_argument('--batch_per_image', type=int, default=128, help='cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE')
    parser.add_argument('--num_classes', type=int, default=1, help='cfg.MODEL.ROI_HEADS.NUM_CLASSES')

    args = parser.parse_args()

    cfg, trainer = train(args)
    validation(cfg, trainer, args)


if __name__ == '__main__':
    main()