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
cv2.createTrackbar('p1', 'Progi', 102, 255, empty_callback)

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
        
        prog = cv2.getTrackbarPos('p1', 'Progi')

        img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        img_szary = cv2.cvtColor(img_hsv, cv2.COLOR_BGR2GRAY)
        res, thresh = cv2.threshold(img_szary, prog, 255, cv2.THRESH_BINARY)

        thresh = cv2.medianBlur(thresh, 13)
        
        distance_filter = np.zeros(thresh.shape, dtype=np.uint8)
        
        distance_filter = cv2.distanceTransform(thresh, cv2.DIST_C, 5)
        cv2.normalize(distance_filter, distance_filter, 0, 255, cv2.NORM_MINMAX)
        distance_filter = distance_filter.astype(np.uint8)     

        #testowe = cv2.adaptiveThreshold(distance_filter, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 25, 2)
        kernel = np.ones((3, 3), np.uint8)
        dilated = cv2.dilate(distance_filter, kernel)

        srodki = dilated - distance_filter
        res, srodki = cv2.threshold(srodki, 0, 255, cv2.THRESH_BINARY)

        srodki = cv2.medianBlur(srodki, 9)
        kernel = np.ones((9, 9), np.uint8)
        srodki = cv2.erode(srodki, kernel, iterations=2)
        
        kontury = cv2.Canny(srodki, 30, 150, 3)
        kontury = cv2.dilate(kontury, (3,3), iterations=2)
        #kontury = cv2.erode(kontury, (5,5), iterations = 2)


        
        (cnt, hierarchy) = cv2.findContours(kontury, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        #print(len(cnt))

        #bez_tla = np.zeros(img_hsv.shape, dtype=img_hsv.dtype)
        #bez_tla = cv2.bitwise_or(img, bez_tla, mask=thresh)

        cv2.imshow('Kolor', img)
        cv2.imshow('Filtr', srodki)
        
        #break
        
        key_code = cv2.waitKey(10)
        if key_code == 27:
            # escape key pressed
            break
    
    red = 0
    yellow = 0
    green = 0
    purple = len(cnt)

    return {'red': red, 'yellow': yellow, 'green': green, 'purple': purple}

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
        print(str(fruits['purple']) + ' / ' + str(results_control[img_path.name]['purple'] + results_control[img_path.name]['red'] + results_control[img_path.name]['yellow'] + results_control[img_path.name]['green']))
    
    ###########3    
    print('Jest:', suma)
    print('Powinno:', suma_kontrolna)
    ###############

    with open(output_file_path, 'w') as ofp:
        json.dump(results, ofp)


if __name__ == '__main__':
    main()
