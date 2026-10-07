"""Pure-numpy fallback for cython_bbox.bbox_overlaps.

`cython_bbox` is a Cython extension, so installing it requires a working C++
toolchain (MSVC on Windows, build-essential elsewhere). `yolox/tracker/matching.py`
only uses a single function from it (`bbox_overlaps`), so a numpy implementation
with identical semantics lets inference-only users run the tracker without any
compiler.

Same MIT license as the rest of this repository.
"""

import numpy as np


def bbox_overlaps(boxes, query_boxes):
    """IoU between two sets of [x1, y1, x2, y2] boxes.

    Keeps the original cython_bbox semantics, including its +1 pixel convention
    for widths/heights and areas.

    :param boxes: (N, 4) float array
    :param query_boxes: (K, 4) float array
    :return: (N, K) IoU matrix
    """
    boxes = np.ascontiguousarray(boxes, dtype=np.float64)
    query_boxes = np.ascontiguousarray(query_boxes, dtype=np.float64)

    N = boxes.shape[0]
    K = query_boxes.shape[0]
    overlaps = np.zeros((N, K), dtype=np.float64)

    for k in range(K):
        box_area = (
            (query_boxes[k, 2] - query_boxes[k, 0] + 1)
            * (query_boxes[k, 3] - query_boxes[k, 1] + 1)
        )
        for n in range(N):
            iw = (
                min(boxes[n, 2], query_boxes[k, 2])
                - max(boxes[n, 0], query_boxes[k, 0])
                + 1
            )
            if iw > 0:
                ih = (
                    min(boxes[n, 3], query_boxes[k, 3])
                    - max(boxes[n, 1], query_boxes[k, 1])
                    + 1
                )
                if ih > 0:
                    ua = (
                        (boxes[n, 2] - boxes[n, 0] + 1)
                        * (boxes[n, 3] - boxes[n, 1] + 1)
                        + box_area
                        - iw * ih
                    )
                    overlaps[n, k] = iw * ih / ua
    return overlaps
