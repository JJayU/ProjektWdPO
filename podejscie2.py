import json
from pathlib import Path
from typing import Dict

import click
import cv2
from tqdm import tqdm

import numpy as np ###################################################### !!!!!!!!!! ##################################

def empty_callback(emm):
    pass

#################################################################
cv2.namedWindow('Progi')
cv2.createTrackbar('HMinBar', 'Progi', 163, 180, empty_callback)
cv2.createTrackbar('SMinBar', 'Progi', 46, 255, empty_callback)
cv2.createTrackbar('VMinBar', 'Progi', 0, 255, empty_callback)
cv2.createTrackbar('HMaxBar', 'Progi', 168, 180, empty_callback)
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
##################################################################

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
        
        odc_szar = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        odc_szar = cv2.GaussianBlur(odc_szar, (11,11), 0)
        canny = cv2.Canny(odc_szar, 30, 150, 3)
        cv2.imshow('Canny', canny)
        
        #break
        
        key_code = cv2.waitKey(10)
        if key_code == 27:
            # escape key pressed
            break
    
    red = 0
    yellow = 0
    green = 0
    purple = 0

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
