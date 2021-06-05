"""
This is an IoU Score implementation between the segmentation result from the OSVOS and the annotations,
performed on data from DAVIS 2016 dataset.
"""
import numpy as np
import os
import sys
import glob
import cv2
from sklearn.metrics import accuracy_score, recall_score

root_folder = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.abspath(root_folder))
os.chdir(root_folder)

def getIndex(contours):
    ind = -1
    len_max = -1
    for i in range(len(contours)):
        if len_max < len(contours[i]):
            len_max = len(contours[i])
            ind = i
    return ind

# annotation_imgs
def contour_score(result_imgs, annotated_imgs):
    i=1
    accuracy_score_contour=0
    recall_score_contour=0
    for img_r, img_a in zip(result_imgs, annotated_imgs):

        img_black = cv2.imread("black.png", 1)
        img_an = cv2.cvtColor(img_a, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(img_an,(25,25),0) # apply blur for contour
        ret, binary = cv2.threshold(blur,25,255,cv2.THRESH_BINARY) # apply threshold to blur image
        contours, hierarchy = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) # find countour
        obj_index = getIndex(contours)
        contour_img_a = cv2.drawContours(img_black, contours, obj_index, (255,255,255), 3) # draw coutour on original image
        contour_img_a = np.array(contour_img_a).reshape((1,np.size(contour_img_a)))[0]
        
        img_black = cv2.imread("black.png", 1)
        img_rn = cv2.cvtColor(img_r, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(img_rn,(25,25),0) # apply blur for contour
        ret, binary = cv2.threshold(blur,25,255,cv2.THRESH_BINARY) # apply threshold to blur image
        contours, hierarchy = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) # find countour
        obj_index = getIndex(contours)
        contour_img_r = cv2.drawContours(img_black, contours, obj_index, (255,255,255), 3) # draw coutour on original image
        contour_img_r = np.array(contour_img_r).reshape((1,np.size(contour_img_r)))[0]
        if np.sum(contour_img_r)!=0 and np.sum(contour_img_a)!=0:
            x = recall_score(contour_img_a, contour_img_r, pos_label=255, zero_division=0)
            if x==0:
                continue
            recall_score_contour += x
            accuracy_score_contour += accuracy_score(contour_img_a, contour_img_r)
            i+=1
    return [recall_score_contour/i, accuracy_score_contour/i]

if __name__ == '__main__':
    try:
        sequences = ['car-shadow', 'parkour', 'tennis', 'paragliding-launch', 'drift-chicane', 'bus', 'surf', 'swing', 'breakdance', 'car-roundabout', 'dance-twirl', 'drift-straight', 'breakdance-flare', 'lucia', 'hike', 'hockey', 'bmx-bumps', 'mallard-water', 'car-turn', 'libby', 'dance-jump', 'rollerblade', 'motocross-jump', 'motocross-bumps', 'soapbox', 'drift-turn', 'scooter-black', 'bmx-trees', 'mallard-fly', 'kite-walk', 'paragliding', 'scooter-gray', 'soccerball', 'stroller', 'kite-surf']
        x = 0
        y = 0
        count1=len(sequences)
        count2=len(sequences)
        for seq_name in sequences:
            result_path_BS = os.path.join('DAVIS', 'Results', 'Segmentations', '480p', 'OSVOS', seq_name)
            annotation_filenames = glob.glob(os.path.join('DAVIS', 'Annotations', '480p', seq_name, '*.png'))
            annotation_filenames.sort()
            annotation_imgs = [cv2.imread(img,1) for img in annotation_filenames]
            result_BS_filenames = glob.glob(os.path.join(result_path_BS, '*.png'))
            if len(result_BS_filenames)==0:
                print("Results not found!")
                count1-=1
                count2-=1
            else:
                result_BS_filenames.sort()
                result_BS_imgs = [cv2.imread(img,1) for img in result_BS_filenames]
                scores = contour_score(result_BS_imgs, annotation_imgs)
                x+=scores[0]
                y+=scores[1]
                print(scores, seq_name)
        print("accuracy contour = {}".format(y/(count1)))
        print("recall contour = {}".format(x/(count2)))
    except KeyboardInterrupt:
        exit()