import sys

sys.path.append('./External_Model_Files/YOLOV5-6')

import models.experimental as YoloV6Experimental
import numpy as np
import torch
from utils import datasets, general, torch_utils

from Interfaces.ModelClassInterface import ModelClassInterface
from Result_Modules.Results import Results


class YoloV5_6(ModelClassInterface):
    """
    A Model Class file. The format in this file is to be
    Globally followed in every model class file.
    It:
        - Recieves its arguments and sets self parameters
        - Loads the model
        - Preprocesses the image if required
        - Infers on the input
        - Returns two results:
            > The standard one
            > One required for the current scenario

    Args:
        ModelClassInterface (class): [description]
    """

    def __init__(self, args):

        # Setting up all the hyperparemeters from args
        self.model_path = args.weights
        self.device = args.device
        self.img_size = args.img_size
        self.conf_thresh = args.conf_thresh
        self.iou_thresh = args.iou_thresh
        self.agnostic_nms = args.agnostic_nms
        self.augment = args.augment
        self.classes = args.classes
        self.filter_persons = args.filter_persons

        self.model = None
        self.names = None

    def set_model(self):

        print("==========       Loading YOLOV5s6 Model...               ==========")

        device = torch_utils.select_device(self.device)
        half = device.type != 'cpu'  # half precision only supported on CUDA

        weights = self.model_path
        w = str(weights[0] if isinstance(weights, list) else weights)
        model = torch.jit.load(w) if 'torchscript' in w else YoloV6Experimental.attempt_load(weights, map_location=device)
        # model = YoloV6Experimental.attempt_load(self.model_path, map_location=device)  # load FP32 model
        self.img_size = general.check_img_size(self.img_size, s=64)  # check img_size
        if half:
            model.half()  # to FP16

        # Get names and colors
        names = model.module.names if hasattr(model, 'module') else model.names

        self.model = model
        self.names = names
        self.device = device
        self.half = half

        print("==========       YOLOV5s6 Model Loaded Successfully!     ==========")

    def preprocess(self, img0):
        # Padded resize
        img = datasets.letterbox(img0, new_shape=self.img_size, stride=64, auto=True)[0]

        # Convert
        # img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
        img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        img = np.ascontiguousarray(img)

        return img

    def infer(self, frame):
        """
            This function take a frame and infers on that single image
            Returns an object detection results object.
        """

        with torch.no_grad():

            # Getting Object Detection Results module for population
            results = Results.get_objectDetection_results()

            im0 = frame.copy()

            image = self.preprocess(im0)

            img = torch.from_numpy(image).to(self.device)

            # img = img.float()  # uint8 to fp16/32
            img = img.half() if self.half else img.float()
            img /= 255.0  # 0 - 255 to 0.0 - 1.0
            if len(img.shape) == 3:
                img = img[None]

            # img = torch.cat([img, img, img, img, img, img, img, img], 0)

            # Inference
            # t1 = torch_utils.time_synchronized()
            pred = self.model(img, augment=self.augment)[0]
            # pred = self.model(torch.zeros(1, 3, *self.img_size))

            # t2 = torch_utils.time_synchronized()
            # print(t2-t1)

            pred = general.non_max_suppression(pred, self.conf_thresh, self.iou_thresh, classes=self.classes, agnostic=self.agnostic_nms)
            # print(len(pred[0]))
            # Process detections
            for i, det in enumerate(pred):  # detections per image

                # gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  #  normalization gain whwh
                if det is not None and len(det):
                    # Rescale boxes from img_size to im0 size
                    det[:, :4] = general.scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                    # Print results
                    # for c in det[:, -1].unique():
                    #     n = (det[:, -1] == c).sum()  # detections per class

                    for *xyxy, conf, cls in reversed(det):
                        # print(len(det))
                        if self.filter_persons:
                            if "person" == self.names[int(cls)]:
                                new_xyxy = np.array([i.cpu().item() for i in xyxy])
                                new_conf = conf.cpu().item()
                                class_name = "person"
                                results.add_detection(new_xyxy, new_conf, class_name)
                        else:
                            new_xyxy = np.array([i.cpu().item() for i in xyxy])
                            new_conf = conf.cpu().item()
                            class_name = self.names[int(cls)]
                            results.add_detection(new_xyxy, new_conf, class_name)

            return results

    # BATCH PROCESSING STILL NEEDS TO BE UPDATED ACCORDING TO THE LATEST REPO

    def create_batch(self, img_list):
        """
            This function take an image list and does the necessary
            preprocessing to pass it to the model.
        """

        list_of_tensors = []

        for frame in img_list:

            im0 = frame.copy()
            image = self.preprocess(im0)

            if self.device == "cpu":
                img = torch.from_numpy(image).to("cpu")
            else:
                img = torch.from_numpy(image).to("cuda")

            img = img.float()  # uint8 to fp16/32
            img /= 255.0  # 0 - 255 to 0.0 - 1.0
            if img.ndimension() == 3:
                img = img.unsqueeze(0)

            list_of_tensors.append(img)

        return torch.cat(list_of_tensors, axis=0)

    def infer_batch(self, img_list):
        """
            A function to infer on a batch of images.
            Takes a list of images and infers.
            Returns a list of ObjectDetection class objects.
        """

        with torch.no_grad():

            im0 = img_list[0].copy()
            img = self.create_batch(img_list, self.img_size)

            # Inference
            # t1 = torch_utils.time_synchronized()
            pred = self.model(img)[0]
            # t2 = torch_utils.time_synchronized()

            pred = general.non_max_suppression(pred, self.conf_thresh, self.iou_thresh, classes=self.classes, agnostic=self.agnostic_nms)

            results_list = []

            # Process detections
            for i, det in enumerate(pred):  # detections per image

                # Getting Object Detection Results module for population
                results = Results.get_objectDetection_results()

                # gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  #  normalization gain whwh
                if det is not None and len(det):
                    # Rescale boxes from img_size to im0 size
                    det[:, :4] = general.scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                    # Print results
                    # for c in det[:, -1].unique():
                    #     n = (det[:, -1] == c).sum()  # detections per class

                    for *xyxy, conf, cls in det:

                        if self.filter_persons:
                            if "person" == self.names[int(cls)]:
                                new_xyxy = np.array([i.cpu().item() for i in xyxy])
                                new_conf = conf.cpu().item()
                                class_name = "person"
                                results.add_detection(new_xyxy, new_conf, class_name)
                        else:
                            new_xyxy = np.array([i.cpu().item() for i in xyxy])
                            new_conf = conf.cpu().item()
                            class_name = self.names[int(cls)]
                            results.add_detection(new_xyxy, new_conf, class_name)

                results_list.append(results)

        return results_list
