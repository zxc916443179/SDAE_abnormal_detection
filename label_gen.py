import cv2
import tensorflow as tf
import mylib
import pickle
import os
import numpy as np

tf.flags.DEFINE_string('path', 'none', 'path to your dataset and groundtruth is yourpath_gt')
tf.flags.DEFINE_string('dst_path', 'none', 'dest of saving dataset')

flags = tf.flags.FLAGS

dataDest = str(flags.dst_path)
mylib.createDirectory(dataDest)
print ("Destination directory to store processed dataset: " + dataDest)


appDataset15 = open(os.path.join(dataDest,"appearance15.p") ,'wb')
label15 = open(os.path.join(dataDest, "label15.p"), 'wb')
paths = flags.path.split(',')
for path in paths:
    imgs = os.listdir(path)
    gt_path = path + '_gt'
    gts = os.listdir(gt_path)
    if '.DS_Store' in imgs:
        imgs.remove('.DS_Store')
        gts.remove('.DS_Store')

    for i in range(len(imgs)):
        obs_img_path = os.path.join(path, imgs[i])
        image = cv2.imread(obs_img_path, 0)
        gt = cv2.imread(os.path.join(gt_path, gts[i]), 0)
        img_resized = cv2.resize(image, (240, 165), interpolation=cv2.INTER_LINEAR)
        gt_resized = cv2.resize(gt, (240, 165), interpolation=cv2.INTER_LINEAR)

        img_slides = mylib.sliding_window(img_resized, stepSize=15, windowSize=(15, 15))
        gt_slides = mylib.sliding_window(gt_resized, stepSize=15, windowSize=(15, 15))
        for (x, y, slide) in img_slides:
            (x, y, gt_slide) = next(gt_slides)
            gt_slide = gt_slide.flatten()
            if mylib.zero_fraction(gt_slide) != 1:
                pickle.dump(1, label15)
            else :
                pickle.dump(0, label15)
            slide = slide.flatten()
            slide = cv2.normalize(slide.astype(float), slide.astype(float), alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
            pickle.dump(slide, appDataset15)