"""
This is an IoU Score implementation between the segmentation result from the OSVOS and the annotations,
performed on data from DAVIS 2016 dataset.
"""
import numpy as np
import os
import sys
import glob
import cv2
root_folder = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.abspath(root_folder))
os.chdir(root_folder)
import math

def mean_iou_score(annotation_imgs, result_imgs,show_per_frame_iou):
    i = 1
    iou_score_sum = 0
    for a_img, r_img in zip(annotation_imgs, result_imgs):
        intersection = np.logical_and(a_img, r_img)
        union = np.logical_or(a_img, r_img)
        try:
            iou_score = np.sum(intersection) / np.sum(union)
        except:
            pass
        # print(iou_score)
        if show_per_frame_iou:
            print("Frame : "+str(i-1)+" score : "+str(iou_score))
        if not math.isnan(iou_score):
            iou_score_sum += iou_score
            i += 1
    return iou_score_sum/i

def recall_iou_score(annotation_imgs, result_imgs,show_per_frame_iou):
    i = 1
    iou_score_sum = 0
    for a_img, r_img in zip(annotation_imgs, result_imgs):
        intersection = np.logical_and(a_img, r_img)
        union = np.logical_or(a_img, r_img)
        try:
            iou_score = np.sum(intersection) / np.sum(union)
        except:
            pass
        # print(iou_score)
        if show_per_frame_iou:
            print("Frame : "+str(i-1)+" score : "+str(iou_score))
        if not math.isnan(iou_score):
            iou_score_sum += iou_score
            i += 1
    return iou_score_sum/i

if __name__ == '__main__':
    sequences = ['car-shadow', 'parkour', 'tennis', 'paragliding-launch', 'drift-chicane', 'bus', 'surf', 'swing', 'breakdance', 'car-roundabout', 'dance-twirl', 'drift-straight', 'breakdance-flare', 'lucia', 'hike', 'hockey', 'bmx-bumps', 'mallard-water', 'car-turn', 'libby', 'dance-jump', 'rollerblade', 'motocross-jump', 'motocross-bumps', 'soapbox', 'drift-turn', 'scooter-black', 'bmx-trees', 'mallard-fly', 'kite-walk', 'paragliding', 'scooter-gray', 'soccerball', 'stroller', 'kite-surf']
    avg1 = 0
    avg2 = 0
    count=len(sequences)
    for seq_name in sequences:
        result_path_no_BS = os.path.join('DAVIS', 'Results', 'Segmentations', '480p', 'OSVOS', seq_name)
        result_path_BS = os.path.join('DAVIS', 'boundary_snappingResults', 'Segmentations', '480p', 'OSVOS', seq_name)
        og_img_path = os.path.join('DAVIS', 'JPEGImages', '480p', seq_name)
        annotation_filenames = glob.glob(os.path.join('DAVIS', 'Annotations', '480p', seq_name, '*.png'))
        annotation_filenames.sort()
        annotation_imgs = [cv2.imread(img,0) for img in annotation_filenames]
        og_img_filenames = glob.glob(os.path.join(og_img_path, '*.jpg'))
        og_img_filenames.sort()
        og_img = [cv2.imread(img) for img in og_img_filenames]
        
        result_no_BS_filenames = glob.glob(os.path.join(result_path_no_BS, '*.png'))
        result_BS_filenames = glob.glob(os.path.join(result_path_BS, '*.png'))

        result_no_BS_filenames.sort()
        result_no_BS_imgs = [cv2.imread(img,0) for img in result_no_BS_filenames]

        if len(result_no_BS_filenames)==0:
            print("result_no_BS_filenames not found!")
        else:
            score = mean_iou_score(annotation_imgs,result_no_BS_imgs,False)
            avg1+=score
            print("IoU Score(s) without BS : ", seq_name, score)
        if len(result_BS_filenames)==0:
            print("Results not found!")
        else:
            result_BS_filenames.sort()
            result_BS_filenames = [cv2.imread(img,0) for img in result_BS_filenames]
            score = mean_iou_score(annotation_imgs,result_BS_filenames,False)
            avg2+=score
            
    print("avg 1", avg1/count)
    print("avg 2", avg2/count)