import logging

from Model_Classes.deep_sort import DeepSort
from Model_Classes.Norfair import Norfair
from Model_Classes.YoloV5_6 import YoloV5_6


class Models:
    """
    This is the main AI models class. The responsibilites of this
    class are as follows:
        - Import all model_class files
        - To load all of the models as per args
        - To contain all the loaded models in its variables and make
          easy usage
        - Provision of statistics regarding every model in one place
    """

    def __init__(self, args, stream_parameters):

        self.args = args

        # Models
        self.PERSON_DETECTOR = None
        self.OBJECT_TRACKER = None
        self.ATTRIBUTE_CLASSIFIER = None
        self.FACE_DETECTOR = None

        self.models_logger = logging.getLogger(__name__)

    def set_initial_models(self, stream_parameters):

        # Traversing through the model setting arguments and setting
        # models which are required to be set globally

        for k in self.args.MODELS.__dict__:
            # import pdb; pdb.set_trace()
            if (not self.PERSON_DETECTOR and (self.args.MODELS.__dict__['use_yolo_v5'] or self.args.MODELS.__dict__['use_yolo_v6'] or stream_parameters['run_intrusion_detection'])):
                self.PERSON_DETECTOR = YoloV5_6(self.args.YOLOV5)
                self.PERSON_DETECTOR.set_model()
                self.models_logger.info('Person detector model (YoloV5) is imported')
            # else:
            #     self.models_logger.error('Person detector model cannot be imported')
            if (not self.OBJECT_TRACKER and (self.args.MODELS.__dict__['use_deepsort'] or self.args.MODELS.__dict__['use_norfair'] or stream_parameters['run_intrusion_detection'])):

                if self.args.MODELS.__dict__['use_norfair']:
                    self.OBJECT_TRACKER = Norfair(self.args.NORFAIR)
                    self.models_logger.info('Object tracker model (Norfair) model is imported')
                if self.args.MODELS.__dict__['use_deepsort']:
                    self.models_logger.info('Object tracker model (Deepsort) model is imported')
                    self.OBJECT_TRACKER = DeepSort(self.args.DEEPSORT)
                self.OBJECT_TRACKER.set_model()
            # else:
            #     self.models_logger.error('Object tracker model cannot be imported')
