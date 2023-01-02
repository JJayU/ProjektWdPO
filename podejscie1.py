import json
from pathlib import Path
from typing import Dict

import click
import cv2
from tqdm import tqdm

import numpy as np ###################################################### !!!!!!!!!! ##################################

def empty_callback(emm):
    pass

cv2.namedWindow('Progi')
cv2.createTrackbar('HMinBar', 'Progi', 163, 180, empty_callback)
cv2.createTrackbar('SMinBar', 'Progi', 46, 255, empty_callback)
cv2.createTrackbar('VMinBar', 'Progi', 0, 255, empty_callback)
cv2.createTrackbar('HMaxBar', 'Progi', 169, 180, empty_callback)
cv2.createTrackbar('SMaxBar', 'Progi', 175, 255, empty_callback)
cv2.createTrackbar('VMaxBar', 'Progi', 255, 255, empty_callback)
#cv2.createTrackbar('p1', 'Progi', 0, 255, empty_callback)
#cv2.createTrackbar('p2', 'Progi', 90, 255, empty_callback)
#cv2.createTrackbar('p3', 'Progi', 0, 255, empty_callback)
#cv2.createTrackbar('p4', 'Progi', 1000, 10000, empty_callback)

cv2.namedWindow('Kolor', cv2.WINDOW_NORMAL)
cv2.namedWindow('Filtr', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Kolor', 800, 800)
cv2.resizeWindow('Filtr', 800, 800)

def detect(img_path: str) -> Dict[str, int]:
    """Object detection function, according to the project description, to implement.

    Parameters
    ----------
    img_path : str
        Path to processed image.

    Returns
    -------
    Dict[str, int]
        Dictionary with quantity of each object.
    """
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)

    #TODO: Implement detection method.
    
    while True:         
        h_min = cv2.getTrackbarPos('HMinBar', 'Progi')
        s_min = cv2.getTrackbarPos('SMinBar', 'Progi')
        v_min = cv2.getTrackbarPos('VMinBar', 'Progi')
        h_max = cv2.getTrackbarPos('HMaxBar', 'Progi')
        s_max = cv2.getTrackbarPos('SMaxBar', 'Progi')
        v_max = cv2.getTrackbarPos('VMaxBar', 'Progi')
        #p1 = cv2.getTrackbarPos('p1', 'Progi')
        #p2 = cv2.getTrackbarPos('p2', 'Progi')
        #p3 = cv2.getTrackbarPos('p3', 'Progi')
        #p4 = cv2.getTrackbarPos('p4', 'Progi')
        
        img_temp = cv2.imread(img_path, cv2.IMREAD_COLOR)
        img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        wynik_hsv = cv2.inRange(img_hsv, (h_min,s_min,v_min), (h_max,s_max,v_max))
        # 143,78,0   180,255,255 126
        # 126,50,0   173,255,255  <- fioletowy
        wynik_hsv = cv2.medianBlur(wynik_hsv, 15)

        #wynik_hsv = cv2.erode(wynik_hsv, np.ones((5, 5), np.uint8))

        # Setup SimpleBlobDetector parameters.
        params = cv2.SimpleBlobDetector_Params()

        # Change thresholds
        params.minThreshold = 0
        params.maxThreshold = 90

        params.filterByColor = 1
        params.blobColor = 255

        # Filter by Area.
        params.filterByArea = True
        params.minArea = 100 # 9 
        params.maxArea = 50000 # 3200

        # Filter by Circularity
        params.filterByCircularity = False
        params.minCircularity = 0

        # Filter by Convexity
        params.filterByConvexity = False
        params.minConvexity = 0.87

        # Filter by Inertia
        params.filterByInertia = False
        params.minInertiaRatio = 0.01

        detector = cv2.SimpleBlobDetector_create(params)
        keypoints = detector.detect(wynik_hsv)
        #print(len(keypoints))
        purple = len(keypoints)
        im_with_keypoints = cv2.drawKeypoints(img_temp, keypoints, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)


        cv2.imshow('Kolor', im_with_keypoints)
        cv2.imshow('Filtr', wynik_hsv)
        
        key_code = cv2.waitKey(10)  # MUSI BYĆ KIEDY JEST imshow()!!!
        
        #break
        
        if key_code == 27:
            # escape key pressed
            break
    
    red = 0
    yellow = 0
    green = 0
    #purple = 0

    return {'red': red, 'yellow': yellow, 'green': green, 'purple': purple}

#def porownaj()

@click.command()
@click.option('-p', '--data_path', help='Path to data directory', type=click.Path(exists=True, file_okay=False, path_type=Path), required=True)
@click.option('-o', '--output_file_path', help='Path to output file', type=click.Path(dir_okay=False, path_type=Path), required=True)
def main(data_path: Path, output_file_path: Path):
    img_list = data_path.glob('*.jpg')

    results = {}
    
    #######333
    f = open('tablica_cukierkow.json')
    results_control = json.load(f)
    #print(results_control)
    ##########
    
    suma = 0 #########################
    suma_kontrolna = 0 #################################3

    for img_path in tqdm(sorted(img_list)):
        fruits = detect(str(img_path))
        results[img_path.name] = fruits
        ##########3
        suma += fruits['purple']          ###########################3
        suma_kontrolna += results_control[img_path.name]['purple']
        #print(fruits['purple'])
    
    ###########3    
    print('Jest:', suma)
    print('Powinno:', suma_kontrolna)
    ###############

    with open(output_file_path, 'w') as ofp:
        json.dump(results, ofp)


if __name__ == '__main__':
    main()
