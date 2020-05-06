import os
import cv2
import json
import tqdm
import pytesseract
import multiprocessing
from pytesseract import image_to_data, Output


pytesseract.pytesseract.tesseract_cmd = r'C:/Program Files/Tesseract-OCR/tesseract.exe'


def read_image(image_path):
    return cv2.imread(image_path)


def image_edge_detection(image):
    return cv2.Canny(image, 100, 200)


def image_ocr(image):
    return image_to_data(image, output_type=Output.DATAFRAME)[['left', 'top', 'width', 'height', 'text']]


def remove_nan_entries(image_ocr_dataframe):
    return image_ocr_dataframe.dropna().reset_index(drop=True)


def draw_bounding_box(image_ocr_data_tuple):
    image, ocr_data = image_ocr_data_tuple

    def draw_box(row):
        cv2.rectangle(image, (row['left'], row['top']), (row['left'] + row['width'], row['top'] + row['height']),\
                      (0, 0, 255), 2)

    ocr_data.apply(lambda row: draw_box(row), axis=1)
    return image


def save_image(image_image_path):
    image, image_path = image_image_path
    cv2.imwrite(image_path, image)
    return None


if __name__ == '__main__':

    config = json.loads(open('config.json').read())
    image_list = [os.path.join(config['original'], image) for image in os.listdir(config['original'])]
    image_path_list = [os.path.join(config['processed'], image) for image in os.listdir(config['original'])]

    print('Reading images from directory ')
    image_read_pool = multiprocessing.Pool(processes=multiprocessing.cpu_count())
    image_list = list(tqdm.tqdm(image_read_pool.map(read_image, image_list), total=len(image_list)))
    print()

    image_list_copy = image_list.copy()

    # print('Preproccessing images by detecting edge')
    # image_read_pool = multiprocessing.Pool(processes=multiprocessing.cpu_count())
    # image_list = list(tqdm.tqdm(image_read_pool.map(image_edge_detection, image_list), total=len(image_list)))

    print('Getting text and box information from ocr ')
    image_read_pool = multiprocessing.Pool(processes=multiprocessing.cpu_count())
    image_list = list(tqdm.tqdm(image_read_pool.map(image_ocr, image_list), total=len(image_list)))
    print()

    print('Removing null entries from ocr data ')
    image_read_pool = multiprocessing.Pool(processes=multiprocessing.cpu_count())
    image_list = list(tqdm.tqdm(image_read_pool.map(remove_nan_entries, image_list), total=len(image_list)))
    print()

    print('Drawing bounding on identified text ')
    image_read_pool = multiprocessing.Pool(processes=multiprocessing.cpu_count())
    image_list = list(tqdm.tqdm(image_read_pool.map(draw_bounding_box, zip(image_list_copy, image_list)),\
                                total=len(image_list)))
    print()

    print('Saving images ')
    image_read_pool = multiprocessing.Pool(processes=multiprocessing.cpu_count())
    image_list = list(tqdm.tqdm(image_read_pool.map(save_image, zip(image_list, image_path_list)),\
                                total=len(image_list)))
    print()
