import cv2
import numpy as np
import torch
from torch.autograd import Variable
import torch.nn.functional as F
import torch.nn as nn
import pickle
import os
from torch.nn.modules import CrossMapLRN2d as SpatialCrossMapLRN
#from torch.legacy.nn import SpatialCrossMapLRN as SpatialCrossMapLRNOld
from torch.autograd import Function, Variable
from torch.nn import Module


def clip_boxes(boxes, im_shape):
    """
    Clip boxes to image boundaries.
    """
    boxes = np.asarray(boxes)
    if boxes.shape[0] == 0:
        return boxes
    boxes = np.copy(boxes)
    # x1 >= 0
    boxes[:, 0::4] = np.maximum(np.minimum(boxes[:, 0::4], im_shape[1] - 1), 0)
    # y1 >= 0
    boxes[:, 1::4] = np.maximum(np.minimum(boxes[:, 1::4], im_shape[0] - 1), 0)
    # x2 < im_shape[1]
    boxes[:, 2::4] = np.maximum(np.minimum(boxes[:, 2::4], im_shape[1] - 1), 0)
    # y2 < im_shape[0]
    boxes[:, 3::4] = np.maximum(np.minimum(boxes[:, 3::4], im_shape[0] - 1), 0)
    return boxes


def load_net(fname, net, prefix='', load_state_dict=False):
    import h5py
    with h5py.File(fname, mode='r') as h5f:
        h5f_is_module = True
        for k in h5f.keys():
            if not str(k).startswith('module.'):
                h5f_is_module = False
                break
        if prefix == '' and not isinstance(net, nn.DataParallel) and h5f_is_module:
            prefix = 'module.'

        for k, v in net.state_dict().items():
            k = prefix + k
            if k in h5f:
                param = torch.from_numpy(np.asarray(h5f[k]))
                if v.size() != param.size():
                    print('Inconsistent shape: {}, {}'.format(v.size(), param.size()))
                else:
                    v.copy_(param)
            else:
                print.warning('No layer: {}'.format(k))

        epoch = h5f.attrs['epoch'] if 'epoch' in h5f.attrs else -1

        if not load_state_dict:
            if 'learning_rates' in h5f.attrs:
                lr = h5f.attrs['learning_rates']
            else:
                lr = h5f.attrs.get('lr', -1)
                lr = np.asarray([lr] if lr > 0 else [], dtype=np.float)

            return epoch, lr

        state_file = fname + '.optimizer_state.pk'
        if os.path.isfile(state_file):
            with open(state_file, 'rb') as f:
                state_dicts = pickle.load(f)
                if not isinstance(state_dicts, list):
                    state_dicts = [state_dicts]
        else:
            state_dicts = None
        return epoch, state_dicts


# class SpatialCrossMapLRNFunc(Function):

#     def __init__(self, size, alpha=1e-4, beta=0.75, k=1):
#         self.size = size
#         self.alpha = alpha
#         self.beta = beta
#         self.k = k

#     def forward(self, input):
#         self.save_for_backward(input)
#         self.lrn = SpatialCrossMapLRNOld(self.size, self.alpha, self.beta, self.k)
#         self.lrn.type(input.type())
#         return self.lrn.forward(input)

#     def backward(self, grad_output):
#         input, = self.saved_tensors
#         return self.lrn.backward(input, grad_output)


# # use this one instead
# class SpatialCrossMapLRN(Module):
#     def __init__(self, size, alpha=1e-4, beta=0.75, k=1):
#         super(SpatialCrossMapLRN, self).__init__()
#         self.size = size
#         self.alpha = alpha
#         self.beta = beta
#         self.k = k

#     def forward(self, input):
#         return SpatialCrossMapLRNFunc(self.size, self.alpha, self.beta, self.k)(input)


class Inception(nn.Module):
    def __init__(self, in_planes, n1x1, n3x3red, n3x3, n5x5red, n5x5, pool_planes):
        super(Inception, self).__init__()
        # 1x1 conv branch
        self.b1 = nn.Sequential(
            nn.Conv2d(in_planes, n1x1, kernel_size=1),
            nn.ReLU(True),
        )

        # 1x1 conv -> 3x3 conv branch
        self.b2 = nn.Sequential(
            nn.Conv2d(in_planes, n3x3red, kernel_size=1),
            nn.ReLU(True),
            nn.Conv2d(n3x3red, n3x3, kernel_size=3, padding=1),
            nn.ReLU(True),
        )

        # 1x1 conv -> 5x5 conv branch
        self.b3 = nn.Sequential(
            nn.Conv2d(in_planes, n5x5red, kernel_size=1),
            nn.ReLU(True),

            nn.Conv2d(n5x5red, n5x5, kernel_size=5, padding=2),
            nn.ReLU(True),
        )

        # 3x3 pool -> 1x1 conv branch
        self.b4 = nn.Sequential(
            nn.MaxPool2d(3, stride=1, padding=1),

            nn.Conv2d(in_planes, pool_planes, kernel_size=1),
            nn.ReLU(True),
        )

    def forward(self, x):
        y1 = self.b1(x)
        y2 = self.b2(x)
        y3 = self.b3(x)
        y4 = self.b4(x)
        return torch.cat([y1,y2,y3,y4], 1)


class GoogLeNet(nn.Module):

    output_channels = 832

    def __init__(self):
        super(GoogLeNet, self).__init__()
        self.pre_layers = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
            nn.ReLU(True),

            nn.MaxPool2d(3, stride=2, ceil_mode=True),
            SpatialCrossMapLRN(5),

            nn.Conv2d(64, 64, 1),
            nn.ReLU(True),

            nn.Conv2d(64, 192, 3, padding=1),
            nn.ReLU(True),

            SpatialCrossMapLRN(5),
            nn.MaxPool2d(3, stride=2, ceil_mode=True),
        )

        self.a3 = Inception(192,  64,  96, 128, 16, 32, 32)
        self.b3 = Inception(256, 128, 128, 192, 32, 96, 64)

        self.maxpool = nn.MaxPool2d(3, stride=2, ceil_mode=True)

        self.a4 = Inception(480, 192,  96, 208, 16,  48,  64)
        self.b4 = Inception(512, 160, 112, 224, 24,  64,  64)
        self.c4 = Inception(512, 128, 128, 256, 24,  64,  64)
        self.d4 = Inception(512, 112, 144, 288, 32,  64,  64)
        self.e4 = Inception(528, 256, 160, 320, 32, 128, 128)

    def forward(self, x):
        out = self.pre_layers(x)
        out = self.a3(out)
        out = self.b3(out)
        out = self.maxpool(out)
        out = self.a4(out)
        out = self.b4(out)
        out = self.c4(out)
        out = self.d4(out)
        out = self.e4(out)

        return out


class Model(nn.Module):
    def __init__(self, n_parts=8):
        super(Model, self).__init__()
        self.n_parts = n_parts

        self.feat_conv = GoogLeNet()
        self.conv_input_feat = nn.Conv2d(self.feat_conv.output_channels, 512, 1)

        # part net
        self.conv_att = nn.Conv2d(512, self.n_parts, 1)

        for i in range(self.n_parts):
            setattr(self, 'linear_feature{}'.format(i+1), nn.Linear(512, 64))

    def forward(self, x):
        feature = self.feat_conv(x)
        feature = self.conv_input_feat(feature)

        att_weights = torch.sigmoid(self.conv_att(feature))

        linear_feautres = []
        for i in range(self.n_parts):
            masked_feature = feature * torch.unsqueeze(att_weights[:, i], 1)
            pooled_feature = F.avg_pool2d(masked_feature, masked_feature.size()[2:4])
            linear_feautres.append(
                getattr(self, 'linear_feature{}'.format(i+1))(pooled_feature.view(pooled_feature.size(0), -1))
            )

        concat_features = torch.cat(linear_feautres, 1)
        normed_feature = concat_features / torch.clamp(torch.norm(concat_features, 2, 1, keepdim=True), min=1e-6)

        return normed_feature


def load_reid_model(ckpt):
    model = Model(n_parts=8)
    model.inp_size = (80, 160)
    load_net(ckpt, model)
    print('Load ReID model from {}'.format(ckpt))

    model = model.cuda()
    model.eval()
    return model


def im_preprocess(image):
    image = np.asarray(image, np.float32)
    image -= np.array([104, 117, 123], dtype=np.float32).reshape(1, 1, -1)
    image = image.transpose((2, 0, 1))
    return image


def extract_image_patches(image, bboxes):
    bboxes = np.round(bboxes).astype(np.int)
    bboxes = clip_boxes(bboxes, image.shape)
    patches = [image[box[1]:box[3], box[0]:box[2]] for box in bboxes]
    return patches


def extract_reid_features(reid_model, image, tlbrs):
    if len(tlbrs) == 0:
        return torch.FloatTensor()

    patches = extract_image_patches(image, tlbrs)
    patches = np.asarray([im_preprocess(cv2.resize(p, reid_model.inp_size)) for p in patches], dtype=np.float32)

    with torch.no_grad():
        im_var = Variable(torch.from_numpy(patches))
        im_var = im_var.cuda()
        features = reid_model(im_var).data
    return features