import numpy as np
from itertools import product

def generate_anchors(feature_size, image_size, scales, aspect_ratios):
    """
    Generate anchor boxes for object detection.
    """
    # Write code here
    stride = image_size/feature_size
    anchors = []
    for i in range(feature_size):
        for j in range(feature_size):
            cx = ((j + 0.5)*stride)
            cy = ((i + 0.5)*stride) 
            for scale, ar in product(scales, aspect_ratios):
                w = scale * np.sqrt(ar)
                h = scale / np.sqrt(ar)
                anchors.append([cx - w/2, cy - h/2, cx + w/2, cy + h/2])

    return anchors