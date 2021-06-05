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

def mean_temporal_consistency_score(result_imgs):
    i = 1
    iou_score_sum = 0
    prev_frame = result_imgs[0]
    old_points = None
    for frame_no in range(1, len(result_imgs)-1):
        occ_flow = cv2.calcOpticalFlowFarneback(prev_frame, result_imgs[frame_no], old_points,
                                       0.5, 3, 15, 3, 5, 1.2, 0)
        mag, ang = cv2.cartToPolar(occ_flow[...,0], occ_flow[...,1])
        hsv2 = cv2.normalize(mag,None,0,255,cv2.NORM_MINMAX)
        intersection = np.logical_and(hsv2, prev_frame)
        union = np.logical_or(hsv2, prev_frame)
        try:
            iou_score = np.sum(intersection) / np.sum(union)
        except:
            pass
        if not math.isnan(iou_score):
            iou_score_sum += iou_score
            i += 1
        prev_frame = result_imgs[frame_no]
        old_points = occ_flow
    return iou_score_sum/i

if __name__ == '__main__':
    sequences = ['car-shadow', 'parkour', 'tennis', 'paragliding-launch', 'drift-chicane', 'bus', 'surf', 'swing', 'breakdance', 'car-roundabout', 'dance-twirl', 'drift-straight', 'breakdance-flare', 'lucia', 'hike', 'hockey', 'bmx-bumps', 'mallard-water', 'car-turn', 'libby', 'dance-jump', 'rollerblade', 'motocross-jump', 'motocross-bumps', 'soapbox', 'drift-turn', 'scooter-black', 'bmx-trees', 'mallard-fly', 'kite-walk', 'paragliding', 'scooter-gray', 'soccerball', 'stroller', 'kite-surf']
    temporal_consistency = 0
    count=len(sequences)
    for seq_name in sequences:
        result_path_BS = os.path.join('DAVIS', 'Results', 'Segmentations', '480p', 'OSVOS', seq_name)
        og_img_path = os.path.join('DAVIS', 'JPEGImages', '480p', seq_name)
        annotation_filenames = glob.glob(os.path.join('DAVIS', 'Annotations', '480p', seq_name, '*.png'))
        annotation_filenames.sort()
        annotation_imgs = [cv2.imread(img,0) for img in annotation_filenames]
        og_img_filenames = glob.glob(os.path.join(og_img_path, '*.jpg'))
        og_img_filenames.sort()
        og_img = [cv2.imread(img) for img in og_img_filenames]
        
        result_BS_filenames = glob.glob(os.path.join(result_path_BS, '*.png'))
        if len(result_BS_filenames)==0:
            print("Results not found!")
            count-=1
        else:
            result_BS_filenames.sort()
            result_BS_filenames = [cv2.imread(img,0) for img in result_BS_filenames]
            score = mean_temporal_consistency_score(result_BS_filenames)
            temporal_consistency+=score
    print("Mean temporal_consistency score: \n", temporal_consistency/count)