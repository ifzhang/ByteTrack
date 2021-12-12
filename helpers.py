import csv
import os
import pathlib
import pickle
import sqlite3
import time
import urllib.request
from typing import Union

import cv2
import numpy as np

import GLOBAL_VARIABLES as GV


class Helpers(object):
    """
    This class contains some general purposes helper functions like create database, is connected to internest, ans so on.

    Args:
        object (class): [description]
    """

    #####################################################################
    # GENERAL HELPER FUNCTIONS
    #####################################################################

    @staticmethod
    def get_txt_file_lines(path):
        """
            This function takes path of a .txt file and
            returns the list of lines in that file
        """

        with open(path) as f:
            sources = [line.rstrip() for line in f]
        return sources

    @staticmethod
    def write_pickle(obj, path):
        with open(path, 'wb') as handle:
            pickle.dump(obj, handle, protocol=pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def read_pickle(path: Union[str, pathlib.Path]):
        """Reads a pickle file

        Args:
            path (str): Path to the pickle file

        Returns:
            object: The object stored in the file. If file is empty or doesn't exist, returns None
        """
        try:
            with open(path, 'rb') as handle:
                obj = pickle.load(handle)
        except (EOFError, FileNotFoundError):
            obj = None
        return obj

    @staticmethod
    def read_multiple_pickle_files(path: str) -> list:
        """Reads multiple pickle files and returns them in a list

        Args:
            path (str): Path to directory containing the .pickle files

        Returns:
            list: List of objects loaded from the files
        """
        object_list = []
        for file in pathlib.Path(path).glob('**/*.pickle'):
            obj = Helpers.read_pickle(file)
            if obj:
                object_list.append(obj)
        return object_list

    @staticmethod
    def draw_polygon(frame, polygons, type='ENTRY'):

        canvas = np.zeros_like(frame)

        for polygon in polygons:
            xx, yy = polygon.exterior.coords.xy
            points = []
            for x, y in zip(xx, yy):
                points.append((int(x), int(y)))

            cv2.fillPoly(canvas, np.array([points]), GV.POLYGON_COLORS[type])

        cv2.addWeighted(frame, 0.8, canvas, 0.2, 0, frame)

    @staticmethod
    def write_dict_header_to_csv(dict_keys, save_location):

        directory_path = save_location.rsplit('/', 1)[0] + '/'
        if not os.path.exists(directory_path):
            os.makedirs(directory_path)

        with open(save_location, newline='', mode='w') as csv_file:
            csv_writer = csv.DictWriter(csv_file, fieldnames=list(dict_keys))
            csv_writer.writeheader()

    @staticmethod
    def write_dict_to_csv_row(row_data, save_location):
        header_data = list(row_data.keys())
        if not os.path.exists(save_location):
            Helpers.write_dict_header_to_csv(header_data, save_location)

        with open(save_location, newline='', mode='a') as csv_file:
            csv_writer = csv.DictWriter(csv_file, fieldnames=header_data)
            csv_writer.writerow(row_data)

    @staticmethod
    def write_csv_row(header_data: list, row_data: list, save_location: str):
        """Writes a csv row and handles the opening and closing of file.


        Args:
            header_data (list): Data to be written as header. Written only when file is initialized
            row_data (list): Data to be appended to row
            save_location (str): Location to be saved
        """
        if not os.path.exists(save_location):
            directory_path = save_location.rsplit('/', 1)[0] + '/'
            if not os.path.exists(directory_path):
                os.makedirs(directory_path)
            with open(save_location, newline='', mode='w') as csv_file:
                csv_writer = csv.writer(csv_file)
                csv_writer.writerow(header_data)

        with open(save_location, newline='', mode='a') as csv_file:
            csv_writer = csv.writer(csv_file)
            csv_writer.writerow(row_data)

    #####################################################################
    # COORDINATE CONVERSION FUNCTIONS
    #####################################################################

    @staticmethod
    def _xcycwh_to_tlwh(bbox_cxcywh):

        bbox_cxcywh[:, 0] = bbox_cxcywh[:, 0] - bbox_cxcywh[:, 2] / 2.
        bbox_cxcywh[:, 1] = bbox_cxcywh[:, 1] - bbox_cxcywh[:, 3] / 2.
        return bbox_cxcywh

    @staticmethod
    def _cxcywh_to_txtybxby(bbox_cxcywh, width, height):
        xc, yc, w, h = bbox_cxcywh
        tx = max(int(xc - w / 2), 0)
        bx = min(int(xc + w / 2), width - 1)
        ty = max(int(yc - h / 2), 0)
        by = min(int(yc + h / 2), height - 1)
        return tx, ty, bx, by

    @staticmethod
    def _tlwh_to_txtybxby(bbox_tlwh, width, height):

        x, y, w, h = bbox_tlwh
        tx = max(int(x), 0)
        bx = min(int(x + w), width - 1)
        ty = max(int(y), 0)
        by = min(int(y + h), height - 1)
        return tx, ty, bx, by

    @staticmethod
    def _txtybxby_to_cxcywh(x1, y1, x2, y2):

        cx = (x1 + x2) / 2
        w = x2 - x1
        cy = (y1 + y2) / 2
        h = y2 - y1
        return cx, cy, w, h

    #####################################################################
    # POLYGONS RELATED FUNCTIONS
    #####################################################################

    @staticmethod
    def get_bottom_center(bbox):
        return np.array([(bbox[2] - bbox[0]) // 2 + bbox[0], bbox[3]])

    @staticmethod
    def toworld(centers, F):

        imagepoint = [centers[0], centers[1], 1]
        worldpoint = np.array(np.dot(F, imagepoint))
        scalar = worldpoint[2]
        xworld = int(worldpoint[0] / scalar)
        yworld = int(worldpoint[1] / scalar)

        return (xworld, yworld)

    #####################################################################
    # Image Related Utilities
    #####################################################################

    @staticmethod
    def clip_crop(frame, detection):
        return frame[int(detection.bbox[1]):int(detection.bbox[3]), int(detection.bbox[0]):int(detection.bbox[2])]

    @staticmethod
    def save_crops_from_bboxes(detections, frame):
        for det in detections.Detections:
            crop = Helpers.clip_crop(frame, det)
            cv2.imwrite(f'./saved_crops/{GV.CROP_SAVE_NUMBER}.jpg', crop)
            GV.CROP_SAVE_NUMBER += 1

    # @staticmethod
    # def load_vino_model(model_path):
    #     model_path = osp.abspath(model_path)
    #     model_description_path = model_path
    #     model_weights_path = osp.splitext(model_path)[0] + ".bin"
    #     assert osp.isfile(model_description_path), \
    #     assert osp.isfile(model_weights_path), \
    #         "Model weights are not found at '%s'" % (model_weights_path)
    #     model = IENetwork(model_description_path, model_weights_path)
    #     return model
    #####################################################################
    # Database Related Utilities
    #####################################################################

    @staticmethod
    def push_data_to_db(db_path, data, key):

        conn = sqlite3.connect(db_path)
        curs = conn.cursor()

        curs.execute('insert into requests (Data, Key) values(?, ?)', [str(data), key])

        conn.commit()
        conn.close()

    @staticmethod
    def clear_db(db_path):

        conn = sqlite3.connect(db_path)
        curs = conn.cursor()

        curs.execute('DELETE from requests')

        conn.commit()
        conn.close()

    @staticmethod
    def create_connection(db_file):
        """ create a database connection to a SQLite database """
        conn = None
        try:
            conn = sqlite3.connect(db_file)
            conn.execute('''CREATE TABLE `requests` (
                    `Row_ID`	INTEGER NOT NULL PRIMARY KEY AUTOINCREMENT,
                    `Data`	TEXT NOT NULL,
                    `Key`	TEXT
                );''')
            print(sqlite3.version)
        except sqlite3.Error as e:
            print(e)
        finally:
            if conn:
                conn.close()

    @staticmethod
    def create_database(video_id, path='./data_db/'):
        """ create a database connection to an SQLite database """
        conn = None
        path = './data_db/' + video_id + '.db'

        if os.path.exists(path):
            print('--------- Database Already Exists ---------')
            return path

        try:
            conn = sqlite3.connect(path)
            conn.execute('''CREATE TABLE `requests` (
                    `Row_ID`	INTEGER NOT NULL PRIMARY KEY AUTOINCREMENT,
                    `Data`	TEXT NOT NULL,
                    `Key`	TEXT
                );''')
        except Exception as e:
            print(e)

        if conn:
            conn.close()

        print('--------- Database Created Successfully ---------')
        return path

    @staticmethod
    def pop_data_from_db(db_path):
        """
            To get a row from DB to post. Deletes that row from the db as well
        """

        data = key = None
        conn = sqlite3.connect(db_path)

        try:
            curs = conn.execute('SELECT Row_ID, Data, Key FROM requests order by Row_ID limit 1')
            result = curs.fetchone()

            if type(result).__name__ != 'NoneType':

                row_id, data, key = result[0], result[1], result[2]
                conn.execute("DELETE from requests where Row_ID = ?;", [row_id])
                conn.commit()
                conn.close()

                return data, key, True
        except Exception as e:
            print(e)
            return data, key, False

        return data, key, False

    @staticmethod
    def is_connected_to_internet(host='http://google.com'):
        time.sleep(1)
        try:
            urllib.request.urlopen(host)
            return True
        except Exception as e:
            print("Error during is_connected_to_internet")
            print(e)
            return False
