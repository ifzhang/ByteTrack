import os
import os.path as osp
from dataclasses import dataclass
from typing import List, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import models, transforms


@dataclass
class SceneContextBundle:
    """
    Container for per-frame scene context.
    """

    road_mask: np.ndarray
    obstacle_distance: np.ndarray
    depth_map: np.ndarray
    intrinsics: np.ndarray


def _load_torchvision_segmenter(
    model_name: str, device: torch.device
) -> Tuple[torch.nn.Module, transforms.Compose]:
    model_name = model_name.lower()
    if model_name == "deeplabv3_resnet50":
        weights = models.segmentation.DeepLabV3_ResNet50_Weights.DEFAULT
        model = models.segmentation.deeplabv3_resnet50(weights=weights)
    elif model_name == "deeplabv3_resnet101":
        weights = models.segmentation.DeepLabV3_ResNet101_Weights.DEFAULT
        model = models.segmentation.deeplabv3_resnet101(weights=weights)
    elif model_name == "lraspp_mobilenet_v3_large":
        weights = models.segmentation.LRASPP_MobileNet_V3_Large_Weights.DEFAULT
        model = models.segmentation.lraspp_mobilenet_v3_large(weights=weights)
    else:
        raise ValueError(f"Unsupported segmentation model: {model_name}")

    model.to(device).eval()
    return model, weights.transforms()


def _load_midas_depth_model(
    model_type: str, device: torch.device, checkpoint_path: Optional[str] = None
) -> Tuple[torch.nn.Module, transforms.Compose]:
    midas = torch.hub.load("intel-isl/MiDaS", model_type)
    midas.to(device).eval()
    midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
    if model_type in {"DPT_Large", "DPT_Hybrid", "DPT_BEiT_L_512"}:
        transform = midas_transforms.dpt_transform
    else:
        transform = midas_transforms.small_transform

    if checkpoint_path and osp.isfile(checkpoint_path):
        state = torch.load(checkpoint_path, map_location=device)
        if "state_dict" in state:
            state = state["state_dict"]
        midas.load_state_dict(state)

    return midas, transform


class SceneContextProcessor:
    """
    Helper class that loads semantic segmentation and depth models and produces
    per-frame context used by the tracker.
    """

    def __init__(
        self,
        device: torch.device,
        segmenter_model: str = "deeplabv3_resnet50",
        segmenter_path: Optional[str] = None,
        depth_model_type: str = "DPT_Hybrid",
        depth_model_path: Optional[str] = None,
        road_overlap_thresh: float = 0.5,
        road_class_ids: Optional[List[int]] = None,
    ) -> None:
        self.device = device
        self.road_overlap_thresh = road_overlap_thresh
        self.road_class_ids = road_class_ids or [7, 8, 12]
        self.segmenter, self.segmenter_transform = self._load_segmenter(
            segmenter_model, segmenter_path
        )
        self.depth_model, self.depth_transform = _load_midas_depth_model(
            depth_model_type, device, depth_model_path
        )
        self._intrinsics_cache: Optional[np.ndarray] = None
        self._last_depth_map: Optional[np.ndarray] = None

    def _load_segmenter(
        self, model_name: str, checkpoint_path: Optional[str]
    ) -> Tuple[torch.nn.Module, transforms.Compose]:
        if checkpoint_path and osp.isfile(checkpoint_path):
            model = torch.jit.load(checkpoint_path, map_location=self.device)
            model.eval()
            transform = transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225],
                    ),
                ]
            )
            return model, transform
        return _load_torchvision_segmenter(model_name, self.device)

    def _compute_intrinsics(self, width: int, height: int) -> np.ndarray:
        if self._intrinsics_cache is None:
            focal = float(max(width, height))
            cx, cy = width / 2.0, height / 2.0
            self._intrinsics_cache = np.array(
                [[focal, 0.0, cx], [0.0, focal, cy], [0.0, 0.0, 1.0]],
                dtype=np.float32,
            )
        return self._intrinsics_cache

    def _run_segmentation(self, frame_bgr: np.ndarray) -> np.ndarray:
        image_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(image_rgb)
        input_tensor = self.segmenter_transform(pil_image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            output = self.segmenter(input_tensor)["out"][0]
            probs = torch.softmax(output, dim=0)
            valid_ids = [i for i in self.road_class_ids if i < probs.shape[0]]
            if valid_ids:
                selected = probs[valid_ids, :, :].sum(dim=0)
            else:
                selected = torch.zeros_like(probs[0])
        road_mask = (selected > self.road_overlap_thresh).cpu().numpy()
        return road_mask.astype(np.uint8)

    def _run_depth(self, frame_bgr: np.ndarray) -> np.ndarray:
        image_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        input_batch = self.depth_transform(image_rgb).to(self.device)
        with torch.no_grad():
            prediction = self.depth_model(input_batch)
            prediction = F.interpolate(
                prediction.unsqueeze(1),
                size=image_rgb.shape[:2],
                mode="bicubic",
                align_corners=False,
            ).squeeze()
        return prediction.cpu().numpy()

    def process_frame(
        self, frame_bgr: np.ndarray, frame_id: int, depth_stride: int = 1
    ) -> SceneContextBundle:
        height, width = frame_bgr.shape[:2]
        intrinsics = self._compute_intrinsics(width, height)

        road_mask = self._run_segmentation(frame_bgr)
        obstacle_distance = cv2.distanceTransform(
            road_mask, distanceType=cv2.DIST_L2, maskSize=3
        )

        if depth_stride < 1:
            depth_stride = 1
        if frame_id % depth_stride == 0 or self._last_depth_map is None:
            self._last_depth_map = self._run_depth(frame_bgr)
        depth_map = self._last_depth_map

        return SceneContextBundle(
            road_mask=road_mask,
            obstacle_distance=obstacle_distance,
            depth_map=depth_map,
            intrinsics=intrinsics,
        )
