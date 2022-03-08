import numpy as np
import cv2
import skimage
from skimage import transform, measure, feature
from skimage.transform import rescale
from skimage.measure import label, find_contours

def predict_image(img: np.ndarray):
    img1 = np.float32(img[:,:,0])
    template = img1[719:770,2842:2897]
    corr_skimage = skimage.feature.match_template(img1, template, pad_input=True)
    lbl, n = skimage.measure.label(corr_skimage >= 0.525, connectivity=2,
                                   return_num=True)
    city_centers = np.int64([np.round(np.mean(np.argwhere(lbl == i),
                                              axis=0)) for i in range(1, n + 1)])

    HLS = cv2.cvtColor(img[110:2500,50:3800], cv2.COLOR_RGB2HLS)

    HUE = HLS[:, :, 0]
    LIGHT = HLS[:, :, 1]
    SAT = HLS[:, :, 2]

    green = (HUE > 35) & (HUE < 61)
    blue = (HUE > 17) & (HUE < 30)  & (LIGHT < 90) & (SAT < 235)
    yellow = (HUE > 92) & (HUE < 100) & (SAT >137)
    black = (LIGHT <40)&(SAT < 50)
    red = (HUE > 120) & (HUE <127) & (SAT > 0) & (LIGHT < 120)

    kernel = np.ones((15,15),np.uint8)
    kernel1= np.ones((20,20),np.uint8)

    green_closing = cv2.morphologyEx(green.astype(np.uint8), cv2.MORPH_CLOSE, kernel)
    green_opening = cv2.morphologyEx(green_closing, cv2.MORPH_OPEN, kernel1)

    blue_closing = cv2.morphologyEx(blue.astype(np.uint8), cv2.MORPH_CLOSE, kernel)
    blue_opening = cv2.morphologyEx(blue_closing, cv2.MORPH_OPEN, kernel1)

    yellow_closing = cv2.morphologyEx(yellow.astype(np.uint8), cv2.MORPH_CLOSE, kernel)
    yellow_opening = cv2.morphologyEx(yellow_closing, cv2.MORPH_OPEN, kernel1)

    black_closing = cv2.morphologyEx(black.astype(np.uint8), cv2.MORPH_CLOSE, kernel)
    black_opening = cv2.morphologyEx(black_closing, cv2.MORPH_OPEN, kernel1)

    red_closing = cv2.morphologyEx(red.astype(np.uint8), cv2.MORPH_CLOSE, kernel)
    red_opening = cv2.morphologyEx(red_closing, cv2.MORPH_OPEN, kernel1)


    green_contours, green_hierarchy = cv2.findContours(green_opening, cv2.RETR_TREE,
                                            cv2.CHAIN_APPROX_SIMPLE)

    blue_contours, blue_hierarchy = cv2.findContours(blue_opening, cv2.RETR_TREE,
                                            cv2.CHAIN_APPROX_SIMPLE)

    yellow_contours, yellow_hierarchy = cv2.findContours(yellow_opening, cv2.RETR_TREE,
                                            cv2.CHAIN_APPROX_SIMPLE)

    black_contours, black_hierarchy = cv2.findContours(black_opening, cv2.RETR_TREE,
                                            cv2.CHAIN_APPROX_SIMPLE)

    red_contours, red_hierarchy = cv2.findContours(red_opening, cv2.RETR_TREE,
                                            cv2.CHAIN_APPROX_SIMPLE)

    def score(contours):
        score = 0
        count = 0
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 4000:
                if area < 7000:
                    count +=1
                    score += 1
                elif area < 13000:
                    count +=2
                    score += 2
                elif area < 22000:
                    count +=3
                    score += 4
                elif area < 29000:
                    count +=4
                    score += 7
                elif area < 37000:
                    count +=6
                    score += 15
                elif area< 48000:
                    count +=8
                    score += 21
        return count, score

    green_count, green_score =score(green_contours)
    blue_count, blue_score =score(blue_contours)
    yellow_count, yellow_score =score(yellow_contours)
    black_count, black_score =score(black_contours)
    red_count, red_score =score(red_contours)
    n_trains = {'blue': blue_count, 'green': green_count, 'black': black_count, 'yellow': yellow_count, 'red': red_count}
    scores = {'blue': blue_score, 'green': green_score, 'black': black_score, 'yellow': yellow_score, 'red': red_score}
    return city_centers, n_trains, scores