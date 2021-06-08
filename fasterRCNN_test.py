# -*- coding: utf-8 -*-

'''
## Test Faster-RCNN model
videorighter@ds.seoultech.ac.kr
https://github.com/videorighter
'''

from PIL import ImageFile
import codecs
import time
import os, json, cv2, fnmatch
from detectron2.engine import DefaultPredictor, DefaultTrainer
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer, ColorMode
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.utils.logger import setup_logger
import utils
import shutil
from sklearn.metrics import confusion_matrix, classification_report
import glob
import numpy as np

setup_logger()

load_path = "/home/videorighter/Deeplearning_teampj/data/annotations/test"

start = time.time()

for d in ["train", "val"]:
    DatasetCatalog.register(
        "food_" + d,
        lambda d=d: utils.get_facelip_dtcs(d, "/home/videorighter/Deeplearning_teampj/data/annotations/" + d))
    MetadataCatalog.get("food_" + d).set(thing_classes=["food"])
train_food_metadata = MetadataCatalog.get("food_train")
val_food_metadata = MetadataCatalog.get("food_val")

cfg = get_cfg()
cfg.merge_from_file(
    "/home/videorighter/Deeplearning_teampj/detectron2/configs/COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml")
cfg.DATASETS.TRAIN = ("food_train",)
cfg.DATASETS.TEST = ("food_val",)
cfg.DATALOADER.NUM_WORKERS = 2
cfg.MODEL.MASK_ON = False
cfg.SOLVER.IMS_PER_BATCH = 2
cfg.SOLVER.BASE_LR = 0.0001
cfg.SOLVER.MAX_ITER = 4000
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # 1 classes ('food')

trainer = DefaultTrainer(cfg)
trainer.resume_or_load(resume=True)
trainer.train()

cfg.MODEL.WEIGHTS = "/home/videorighter/Deeplearning_teampj/output/model_final.pth"
predictor = DefaultPredictor(cfg)

file_list = os.listdir(load_path)

threshold = np.arange(0.81, 1.00, 0.01)
print(threshold)

if not os.path.isdir('/home/videorighter/Deeplearning_teampj/figs'):
    os.mkdir('/home/videorighter/Deeplearning_teampj/figs')

for i, point in enumerate(threshold):
    print(point)
    result_dict = {}
    pred = []
    # nonfood = 0 / food = 1
    gt = []

    if not os.path.isdir(f'/home/videorighter/Deeplearning_teampj/result_{i}'):
        os.mkdir(f'/home/videorighter/Deeplearning_teampj/result_{i}')

    for file in file_list:
        tmp_dict = {}
        im = cv2.imread(os.path.join(load_path, file))
        outputs = predictor(im)
        v = Visualizer(im[:, :, ::-1],
                       metadata=val_food_metadata,
                       scale=1.2)
        v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
        img = v.get_image()[:, :, ::-1]

        classes = outputs["instances"].pred_classes.to('cpu').numpy()
        scores = outputs["instances"].scores.to('cpu').numpy()
        boxes = outputs["instances"].pred_boxes.to('cpu').tensor.numpy()

        tmp_dict["file_name"] = os.path.splitext(os.path.basename(file))[0]
        tmp_dict["file_path"] = file
        tmp_dict["classes"] = classes.tolist()
        tmp_dict["scores"] = scores.tolist()
        tmp_dict["boexs"] = boxes.tolist()

        for j, n in enumerate(tmp_dict["scores"]):
            if n > point:
                pred.append(int(1))
                if not os.path.isdir(f'/home/videorighter/Deeplearning_teampj/result_{i}/food'):
                    os.mkdir(f'/home/videorighter/Deeplearning_teampj/result_{i}/food')
                cv2.imwrite(os.path.join(f'/home/videorighter/Deeplearning_teampj/result_{i}/food', file), img)
                break
            elif j+1 == len(tmp_dict["scores"]):
                result_dict[os.path.splitext(os.path.basename(file))[0]] = tmp_dict
                pred.append(int(0))
                if not os.path.isdir(f'/home/videorighter/Deeplearning_teampj/result_{i}/nonfood'):
                    os.mkdir(f'/home/videorighter/Deeplearning_teampj/result_{i}/nonfood')
                cv2.imwrite(os.path.join(f'/home/videorighter/Deeplearning_teampj/result_{i}/nonfood', file), img)

        gt.append(int(file.split('_')[0]))

    # store data(serialize)
    json.dump(result_dict, codecs.open(f"result_{i}.json", "w", encoding="utf-8"), separators=(',', ':'), indent=4)
    print("Classified images: ", len(result_dict))
    report = classification_report(gt, pred, target_names=['nonfood', 'food'])
    print(report)
    cf = confusion_matrix(gt, pred)
    print(cf)

    utils.draw_plot(gt, pred, cf, i - 1)
