'''
videorighter@ds.seoultech.ac.kr
util functions for detectron2
'''

from detectron2.structures import BoxMode
from detectron2.data import MetadataCatalog, DatasetCatalog
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve
import seaborn as sn


def get_facelip_dtcs(d, json_dir):
    json_file = json_dir + "/" + d + ".json"
    with open(json_file) as f:
        imgs_anns = json.load(f)
    dataset_dicts = []
    for z in range(len(imgs_anns['images'])):
        record = {}
        record['file_name'] = json_dir + "/" + imgs_anns['images'][z]['id'] + '.png'
        record['image_id'] = imgs_anns['images'][z]['id']
        record['width'] = imgs_anns['images'][z]['width']
        record['height'] = imgs_anns['images'][z]['height']
        anno_list = []
        for i in range(len(imgs_anns['annotations'])):
            anno = {}
            if imgs_anns['images'][z]['id'] == imgs_anns['annotations'][i]['image_id']:
                anno['bbox'] = imgs_anns['annotations'][i]['bbox'].copy()  # check
                anno['bbox_mode'] = BoxMode.XYWH_ABS
                anno['segmentation'] = []
                anno['category_id'] = imgs_anns['annotations'][i]['category_id']  # check
                anno_list.append(anno)
        record['annotations'] = anno_list
        dataset_dicts.append(record)
    return dataset_dicts


def filter_predictions_from_outputs(outputs,
                                    threshold=0.7,
                                    verbose=True):
    predictions = outputs["instances"].to("cpu")

    if verbose:
        print(list(predictions.get_fields()))

    # Reference: https://github.com/facebookresearch/detectron2/blob/7f06f5383421b847d299b8edf480a71e2af66e63/detectron2/structures/instances.py#L27
    #
    #   Indexing: ``instances[indices]`` will apply the indexing on all the fields
    #   and returns a new :class:`Instances`.
    #   Typically, ``indices`` is a integer vector of indices,
    #   or a binary mask of length ``num_instances``

    indices = [i
               for (i, s) in enumerate(predictions.scores)
               if s >= threshold
               ]

    filtered_predictions = predictions[indices]

    return filtered_predictions


def draw_plot(y_test, y_pred, cm, num):
    group_counts = ["{0:0.0f}".format(value) for value in cm.flatten()]
    group_percentage = ["{0:.2%}".format(value) for value in cm.flatten() / np.sum(cm)]
    labels = [f"{v1}\n{v2}" for v1, v2 in zip(group_counts, group_percentage)]
    labels = np.asarray(labels).reshape(2, 2)
    df_cm = pd.DataFrame(cm, index=['T_nonfood', 'T_food'], columns=['P_nonfood', 'P_food'])
    fig = plt.figure(figsize=(20, 8))
    fig.suptitle('Faster-RCNN classifier', fontsize=23)

    ax2 = fig.add_subplot(121)
    fpr, tpr, thresholds = roc_curve(y_test, y_pred)
    ax2.plot(fpr, tpr)
    ax2.set_xlim([0.0, 1.0])
    ax2.set_ylim([0.0, 1.0])
    ax2.set_title('ROC curve for classifier', fontsize=20)
    ax2.set_xlabel('False Positive Rate (1 - Specificity)', fontsize=20)
    ax2.set_ylabel('True Positive Rate (Sensitivity)', fontsize=20)
    ax2.tick_params(axis='both', labelsize=15)
    ax2.grid(True)

    ax3 = fig.add_subplot(122)
    ax3.set_title('Confusion matrix for classifier', fontsize=20)
    ax3.tick_params(axis='both', labelsize=15)
    sn.heatmap(df_cm, annot=labels, fmt='', annot_kws={"size": 20}, cmap='Reds')

    plt.savefig(f"/home/videorighter/Deeplearning_teampj/figs/fasterRCNN_{num + 1}.png")
    plt.show()


def metadata(annotation_path):  # must have train and val directory
    for d in ["train", "val"]:
        DatasetCatalog.register("food_" + d, lambda d=d: utils.get_facelip_dtcs(d, annotation_path + d))
        MetadataCatalog.get("food_" + d).set(thing_classes=["food"])
    train_metadata = MetadataCatalog.get("food_train")
    val_metadata = MetadataCatalog.get("food_val")

    return train_metadata, val_metadata