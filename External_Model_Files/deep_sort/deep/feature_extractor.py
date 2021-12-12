import cv2
import numpy as np
import torch
import torchvision.transforms as transforms

# from .nn_model import Net
from .nn_model import Net


class Extractor(object):
    """
    This class is a feature extractor used by the deepsort model.

    Args:
        object (class): [description]
    """

    def __init__(self, model_path, use_cuda=True):
        self.net = Net(reid=True)
        self.device = "cuda" if torch.cuda.is_available() and use_cuda else "cpu"
        state_dict = torch.load(model_path, map_location=lambda storage, loc: storage)[
            'net_dict']
        self.net.load_state_dict(state_dict)
        # print("Loading weights from {}... Done!".format(model_path))
        self.net.to(self.device)
        self.size = (3, 64, 128)
        self.norm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

        # self.norm = transforms.Compose([transforms.ToTensor(),
        #                                 transforms.Normalize((0.5,), (0.5,))
        #                                 ])

    def _preprocess(self, im_crops):
        """
        TODO:
            1. to float with scale from 0 to 1
            2. resize to (64, 128) as Market1501 dataset did
            3. concatenate to a numpy array
            3. to torch Tensor
            4. normalize
        """
        def _resize(im, size):
            return cv2.resize(im.astype(np.float32) / 255., (64, 128))

        im_batch = torch.cat([self.norm(_resize(im, self.size)).unsqueeze(0) for im in im_crops], dim=0).float()
        return im_batch

    def __call__(self, im_crops):
        im_batch = self._preprocess(im_crops)
        with torch.no_grad():
            im_batch = im_batch.to(self.device)

            # im_batch = torch.cat((im_batch, im_batch, im_batch, im_batch, im_batch, im_batch,
            # im_batch, im_batch, im_batch, im_batch, im_batch, im_batch, im_batch, im_batch, im_batch, im_batch, im_batch, im_batch, im_batch, im_batch, im_batch, im_batch, im_batch, im_batch, im_batch, im_batch,
            # im_batch, im_batch, im_batch, im_batch, im_batch, im_batch, im_batch, im_batch, im_batch, im_batch, im_batch, im_batch, im_batch, im_batch, im_batch, im_batch, im_batch, im_batch, im_batch, im_batch, im_batch, im_batch, im_batch, im_batch, im_batch, im_batch, im_batch,
            # im_batch, im_batch, im_batch, im_batch, im_batch, im_batch, im_batch, im_batch, im_batch, im_batch, im_batch), axis = 0)
            features = self.net(im_batch)

            # GV.MODEL_TIMES['deepsort_extractor'] += (t2 - t1)

        # cache list
        # [0 frame features, 1 frame features .... 40]

        return features.cpu().numpy()


if __name__ == "__main__":
    img = cv2.imread("demo.png")[:, :, (2, 1, 0)]
    # img = cv2.imread("demo.png")
    print(img.shape)
    extr = Extractor("checkpoint/ckpt.t7")
    feature = extr(img)
    print(feature.shape)
