"""
I don't to save the windows as png. They are all already in /data_c/coco/...
I just need the bbox (x_start, y_start, x_end, y_end)
I can get the window by cutting it out from the coco data 

I don't need to safe the mask as png. I can save it as RLE, which is saving a lot of disk space.

json structure:
{
    "<image_id>_<proposal_id>": [[x_start, y_start, x_end, y_end], RLE_binary_mask, iou],
    ...
}

Upon dataloading:
Input: image_id, proposal_id
1. Load the image by <image_id>
2. Cut out window with [x_start, y_start, x_end, y_end]
3. get binary_mask with decode(RLE_binary_mask)
4. simply read IoU
 

"""
import json
import os
import numpy

from pycocotools import coco
from alchemy.utils import mask as mask_util

import numpy
import numpy as np

from config import ANNOTATION_FILE_FORMAT, ANNOTATION_TYPE, MY_RESULTS_FILE_NAME


def binaryMaskIOU(mask1, mask2):   # From the question.
    mask1_area = np.count_nonzero(mask1 == 1)
    mask2_area = np.count_nonzero(mask2 == 1)
    intersection = np.count_nonzero(np.logical_and( mask1==1,  mask2==1 ))
    try:
        iou = float(intersection)/(mask1_area+mask2_area-intersection)
    except ZeroDivisionError:
        return 0
    return iou



def count_occourences(array):
    unique, counts = numpy.unique(array, return_counts=True)
    return dict(zip(unique, counts))

class MaskExtractor():

    ANNOTATION_FILE_PATH = ANNOTATION_FILE_FORMAT % ANNOTATION_TYPE
    RESULT_FILE_PATH = "/data2/8hirsch/att_mask_extraction/%s" % MY_RESULTS_FILE_NAME

    COCO = coco.COCO(ANNOTATION_FILE_PATH)
    CATEGORY_IDS = COCO.getCatIds()  # get all categories

    results = {}

    def __init__(self, image_id):
        self.image_id = image_id

        anns_ids = self.COCO.getAnnIds(imgIds=image_id, catIds=self.CATEGORY_IDS, iscrowd=None)
        anns = self.COCO.loadAnns(anns_ids)
        self.invalid = False
        if not anns:
            self.invalid = True
            return
        anns_img = np.zeros_like(self.COCO.annToMask(anns[0]))
        for ann in anns:
            # TODO this causes an error sometimes
            anns_img = np.maximum(anns_img, self.COCO.annToMask(ann))

        self.global_ground_truth_mask = anns_img.astype(np.uint8)

    def local_gt_mask(self, x_begin, x_end, y_begin, y_end):
        return self.global_ground_truth_mask[x_begin: x_end, y_begin: y_end]

    def add_entry(self, proposal_id, x_begin, x_end, y_begin, y_end, local_proposal_mask):
        """Adds an entry for given proposal to the results dict.

        Args:
            x_begin (int): Where the window starts on x_axis
            x_end (int): Where the window ends on x_axis
            y_begin (int): Where the window starts on y_axis
            y_end (int): Where the window ends on y_axis
            proposal_id (int): id of proposal
            local_proposal_mask (dict): a lre-encoded mask 
        """
        if self.invalid:
            return
        key = format(self.image_id, '012d') + "_" + format(proposal_id, '06d')

        local_gt_mask = self.global_ground_truth_mask[x_begin: x_end, y_begin: y_end]
        # rle_local_gt_mask = mask_util.encode(local_gt_mask)
        rle_local_proposal_mask = mask_util.encode(local_proposal_mask)
        # iou = mask_util.iou(rle_local_proposal_mask, rle_local_gt_mask, [1])
        iou = binaryMaskIOU(local_gt_mask, local_proposal_mask)
        bbox = [x_begin, x_end, y_begin, y_end]
        self.results[key] = {
            "bbox": bbox, 
            "mask": rle_local_proposal_mask,
            "iou": iou
        }

    @classmethod
    def save_results_file(cls):
        with open(cls.RESULT_FILE_PATH, 'w') as filepath:
            json.dump(cls.results, filepath)
