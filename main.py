"""
Runnable tracker file that implements OSVOS tracker, Kalman filtering and IoU scores.
"""
import os
import sys
from typing import Sequence
root_folder = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.abspath(root_folder))
os.chdir(root_folder)
import numpy as np
import cv2
import glob
from PIL import Image
from osvos_train_test import train_and_test_osvos
from osvos_IoU_score import mean_iou_score
import matplotlib.pyplot as plt

def main(seq):
    # User defined parameters
    seq_name = seq         # Change to train and test other data sets. Should be the name of the folder containing the images.
    gpu_id = 0                      # Change according to your GPU id.
    train_model = True             # Change to train/not train the model. If set to False, you need pre-trained model.
    max_training_iters = 500       # Change this according to the model name if using the pretrained models
    train_img_name = '00000.jpg'    # Change to train with a different frame
    annot_img_name = '00000.png'    # This should be the same as the 'train_img_name'. Extensions of file should used accordingly.
    show_per_frame_iou = False       # Set this to True to show IoU score of every frame, False to show just mean IoU score.

    boundary_snapping_result_path = os.path.join('DAVIS', 'Results', 'Segmentations', '480p', 'OSVOS', seq_name)
    og_img_path = os.path.join('DAVIS', 'JPEGImages', '480p', seq_name)

    annotation_filenames = glob.glob(os.path.join('DAVIS', 'Annotations', '480p', seq_name, '*.png'))
    annotation_filenames.sort()
    annotation_imgs = [cv2.imread(img,0) for img in annotation_filenames]

    og_img_filenames = glob.glob(os.path.join(og_img_path, '*.jpg'))
    og_img_filenames.sort()
    og_img = [cv2.imread(img) for img in og_img_filenames]
    
    result_filenames = glob.glob(os.path.join(boundary_snapping_result_path, '*.png'))
   
    # Train and test or just test, depending on the value of "train_model"
    if len(result_filenames)==0:
        train_and_test_osvos(seq_name, gpu_id, boundary_snapping_result_path, train_model, max_training_iters, train_img_name, annot_img_name)
        result_filenames = glob.glob(os.path.join(boundary_snapping_result_path, '*.png'))
        result_filenames.sort()
        result_imgs = [cv2.imread(img,0) for img in result_filenames]
    else:
        result_filenames.sort()
        result_imgs = [cv2.imread(img,0) for img in result_filenames]
    
    # Find contours
    nameCounter = 0
    for img_p, frame in zip(result_imgs,og_img):
        segmentationMask=img_p
        contours, hierarchy =cv2.findContours(segmentationMask,1, 2)
        newSegmentationMask=np.zeros_like(segmentationMask,dtype=np.uint8)
        cv2.drawContours(newSegmentationMask, contours,-1,(255,255,255),thickness=cv2.FILLED)
        
        # Write new segmentation mask to file
        if not os.path.exists(boundary_snapping_result_path):
            os.makedirs(boundary_snapping_result_path)
        cv2.imwrite(os.path.join(boundary_snapping_result_path, os.path.basename(result_filenames[nameCounter])), newSegmentationMask)
        nameCounter += 1

    boundary_snapping_result_path = glob.glob(os.path.join(boundary_snapping_result_path, '*.png'))

    if len(boundary_snapping_result_path)==0:
        print("Results not found!")
    else:
        boundary_snapping_result_path.sort()
        boundary_snapping_result_path = [cv2.imread(img,0) for img in boundary_snapping_result_path]
        print("mean IOU with contour snapping for ", seq, "is: ", mean_iou_score(annotation_imgs,boundary_snapping_result_path,show_per_frame_iou))

if __name__ == '__main__':
    sequences = ['car-shadow', 'parkour', 'horsejump-high', 'flamingo', 'tennis', 'paragliding-launch', 'goat', 'bear', 'drift-chicane', 'bus', 'surf', 'swing', 'breakdance', 'car-roundabout', 'dance-twirl', 'drift-straight', 'breakdance-flare', 'lucia', 'hike', 'hockey', 'bmx-bumps', 'mallard-water', 'car-turn', 'libby', 'elephant', 'dance-jump', 'dog-agility', 'camel', 'horsejump-low', 'rollerblade', 'blackswan', 'motocross-jump', 'motocross-bumps', 'soapbox', 'drift-turn', 'scooter-black', 'bmx-trees', 'rhino', 'mallard-fly', 'kite-walk', 'paragliding', 'scooter-gray', 'soccerball', 'stroller', 'kite-surf']
    youtube_seqs = ['boat', 'cows', 'dog', 'motorbike', 'train']
    for seq in sequences:
        main(seq)