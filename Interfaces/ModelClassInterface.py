from abc import ABC, abstractmethod


class ModelClassInterface(ABC):

    @abstractmethod
    def set_model(self):
        """Set your model using the metadata, in this function"""
        pass

    @abstractmethod
    def infer(self, frame):
        """Do inference on your image in this function and return results class"""
        pass
