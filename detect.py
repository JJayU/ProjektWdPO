import json
from pathlib import Path
from typing import Dict

import click
import cv2
from tqdm import tqdm

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
    
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Czerwone #
    filtr_czerwone = cv2.inRange(hsv, (175, 50, 100), (185, 255, 255))
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
    filtr_fioletowe = cv2.inRange(hsv, (108, 28, 20), (165, 255, 255))
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
    filtr_zielone = cv2.inRange(hsv, (34, 210, 50), (99, 255, 255))
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
    filtr_zolte = cv2.inRange(hsv, (17, 213, 50), (23, 255, 255))
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

    for img_path in tqdm(sorted(img_list)):
        fruits = detect(str(img_path))
        results[img_path.name] = fruits

    with open(output_file_path, 'w') as ofp:
        json.dump(results, ofp)

if __name__ == '__main__':
    main()
