# Deeplearing_teampj
====================================================================================================
## The search filter using Object detection task [Faster R-CNN]


## How to use
```
# training faster R-CNN

$ python fasterRCNN_train.py --init_train [BOOL] --threshold [FLOAT] --num_workers [INTEGER] --mask_on [BOOL] --ims_per_batch [INTEGER] --lr [FLOAT] --max_iter [INTEGER] --batch_per_image [INTEGER] --num_classes [INTEGER]

# metric: AP(Average Precision)
```
output/
- last_checkpoint
- model_final.pth

output_val/
- coco_instances_results.json

val_result/
- visualize validation images

```
$ python fasterRCNN_test.py
```
>result_{i}/
>>food/ <br>
>>>classify food <br>

>>nonfood/ <br>
>>>classify nonfood

