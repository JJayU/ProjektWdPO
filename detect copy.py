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
cv2.createTrackbar('p1', 'Progi', 108, 255, empty_callback)
cv2.createTrackbar('p2', 'Progi', 28, 255, empty_callback)
cv2.createTrackbar('p3', 'Progi', 165, 255, empty_callback)
cv2.createTrackbar('p4', 'Progi', 255, 255, empty_callback)

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
        
        prog1 = cv2.getTrackbarPos('p1', 'Progi')
        prog2 = cv2.getTrackbarPos('p2', 'Progi')
        prog3 = cv2.getTrackbarPos('p3', 'Progi')
        prog4 = cv2.getTrackbarPos('p4', 'Progi')

        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        # Czerwone #
        filtr_czerwone = cv2.inRange(hsv, np.array([175, 50, 100]), np.array([185, 255, 255]))
        filtr_czerwone = cv2.blur(filtr_czerwone, (9,9))
        ret, filtr_czerwone = cv2.threshold(filtr_czerwone, 80, 255, cv2.THRESH_BINARY)

        detector_czerwone_parametry = cv2.SimpleBlobDetector_Params()
        detector_czerwone_parametry.blobColor = 255
        detector_czerwone_parametry.minArea = 280
        detector_czerwone_parametry.maxArea = 100000
        detector_czerwone_parametry.filterByCircularity = False
        detector_czerwone_parametry.filterByConvexity = False
        detector_czerwone_parametry.filterByInertia = False

        detector_czerwone = cv2.SimpleBlobDetector_create(detector_czerwone_parametry)
        wykryte_czerwone = detector_czerwone.detect(filtr_czerwone)

        # Fioletowe #
        filtr_fioletowe = cv2.inRange(hsv, np.array([108, 28, 20]), np.array([165, 255, 255]))
        filtr_fioletowe = cv2.blur(filtr_fioletowe, (9,9))
        ret, filtr_fioletowe = cv2.threshold(filtr_fioletowe, 80, 255, cv2.THRESH_BINARY)

        detector_fioletowe_parametry = cv2.SimpleBlobDetector_Params()
        detector_fioletowe_parametry.blobColor = 255
        detector_fioletowe_parametry.minArea = 200
        detector_fioletowe_parametry.maxArea = 600000
        detector_fioletowe_parametry.filterByCircularity = False
        detector_fioletowe_parametry.filterByConvexity = False
        detector_fioletowe_parametry.filterByInertia = False

        detector_fioletowe = cv2.SimpleBlobDetector_create(detector_fioletowe_parametry)
        wykryte_fioletowe = detector_fioletowe.detect(filtr_fioletowe)

        # Zielone #
        filtr_zielone = cv2.inRange(hsv, np.array([34, 210, 50]), np.array([99, 255, 255]))
        filtr_zielone = cv2.blur(filtr_zielone, (9,9))
        ret, filtr_zielone = cv2.threshold(filtr_zielone, 80, 255, cv2.THRESH_BINARY)

        detector_zielone_parametry = cv2.SimpleBlobDetector_Params()
        detector_zielone_parametry.blobColor = 255
        detector_zielone_parametry.minArea = 200
        detector_zielone_parametry.maxArea = 500000
        detector_zielone_parametry.filterByCircularity = False
        detector_zielone_parametry.filterByConvexity = False
        detector_zielone_parametry.filterByInertia = False

        detector_zielone = cv2.SimpleBlobDetector_create(detector_zielone_parametry)
        wykryte_zielone = detector_zielone.detect(filtr_zielone)

        # Zolte #
        filtr_zolte = cv2.inRange(hsv, np.array([17, 213, 50]), np.array([23, 255, 255]))
        filtr_zolte = cv2.blur(filtr_zolte, (9,9))
        ret, filtr_zolte = cv2.threshold(filtr_zolte, 80, 255, cv2.THRESH_BINARY)

        detector_zolte_parametry = cv2.SimpleBlobDetector_Params()
        detector_zolte_parametry.blobColor = 255
        detector_zolte_parametry.minArea = 150
        detector_zolte_parametry.maxArea = 150000
        detector_zolte_parametry.filterByCircularity = False
        detector_zolte_parametry.filterByConvexity = False
        detector_zolte_parametry.filterByInertia = False

        detector_zolte = cv2.SimpleBlobDetector_create(detector_zolte_parametry)
        wykryte_zolte = detector_zolte.detect(filtr_zolte)

        im_with_keypoints = cv2.drawKeypoints(img, wykryte_fioletowe, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        cv2.imshow('Kolor', im_with_keypoints)
        cv2.imshow('Filtr', filtr_fioletowe)

        break

        print(len(wykryte_fioletowe))
        
        key_code = cv2.waitKey(10)
        if key_code == 27:
            # escape key pressed
            break
    
    red = len(wykryte_czerwone)
    yellow = len(wykryte_zolte)
    green = len(wykryte_zielone)
    purple = len(wykryte_fioletowe)

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
    
    suma_czerwone = 0 #########################
    suma_kontrolna_czerwone = 0 #################################3
    suma_zielone = 0
    suma_kontrolna_zielone = 0
    suma_zolte = 0        ###########################3
    suma_kontrolna_zolte = 0
    suma_fioletowe = 0
    suma_kontrolna_fioletowe = 0
    MARPE = 0

    for img_path in tqdm(sorted(img_list)):
        fruits = detect(str(img_path))
        results[img_path.name] = fruits
        ##########3
        suma_czerwone += fruits['red']          ###########################3
        suma_kontrolna_czerwone += results_control[img_path.name]['red']
        suma_zielone += fruits['green']          ###########################3
        suma_kontrolna_zielone += results_control[img_path.name]['green']
        suma_zolte += fruits['yellow']          ###########################3
        suma_kontrolna_zolte += results_control[img_path.name]['yellow']
        suma_fioletowe += fruits['purple']          ###########################3
        suma_kontrolna_fioletowe += results_control[img_path.name]['purple']
        MARPE += (abs(fruits['red']-results_control[img_path.name]['red'])+abs(fruits['green']-results_control[img_path.name]['green'])+abs(fruits['yellow']-results_control[img_path.name]['yellow'])+abs(fruits['purple']-results_control[img_path.name]['purple']))/(results_control[img_path.name]['yellow']+results_control[img_path.name]['red']+results_control[img_path.name]['green']+results_control[img_path.name]['purple'])
    
    ###########3    
    print('Czerwone:\nJest:', suma_czerwone)
    print('Powinno:', suma_kontrolna_czerwone)
    print(str((abs(suma_czerwone-suma_kontrolna_czerwone)/suma_kontrolna_czerwone) * 100), ' %')
    print('Zielone:\nJest:', suma_zielone)
    print('Powinno:', suma_kontrolna_zielone)
    print(str((abs(suma_zielone-suma_kontrolna_zielone)/suma_kontrolna_zielone) * 100), ' %')
    print('Zolte:\nJest:', suma_zolte)
    print('Powinno:', suma_kontrolna_zolte)
    print(str((abs(suma_zolte-suma_kontrolna_zolte)/suma_kontrolna_zolte) * 100), ' %')
    print('Fioletowe:\nJest:', suma_fioletowe)
    print('Powinno:', suma_kontrolna_fioletowe)
    print(str((abs(suma_fioletowe-suma_kontrolna_fioletowe)/suma_kontrolna_fioletowe) * 100), ' %')

    MARPE = 100/40 * MARPE
    print('MARPE:', MARPE)
    ###############

    with open(output_file_path, 'w') as ofp:
        json.dump(results, ofp)


if __name__ == '__main__':
    main()
