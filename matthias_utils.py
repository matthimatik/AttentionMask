import json
import os
from PIL import Image
import numpy
from pycocotools import coco

import numpy
import numpy as np

from PIL import Image

from config import ANNOTATION_FILE_FORMAT, ANNOTATION_TYPE, RESULTS_FOLDER


def count_occourences(array):
    unique, counts = numpy.unique(array, return_counts=True)
    return dict(zip(unique, counts))


def union_mask(mask1, mask2):
    union_mask = np.zeros_like(mask1)
    union_mask[mask1 + mask2 > 0] = 1
    return union_mask


def intersection_mask(mask1, mask2):
    return mask1 * mask2


def calc_iou(mask1, mask2):
    intersection_count = np.count_nonzero(intersection_mask(mask1, mask2))
    union_count = np.count_nonzero(union_mask(mask1, mask2))

    try:
        iou = float(intersection_count) / float(union_count)
    except ZeroDivisionError as e:
        iou = 0
    finally:
        return iou


class MaskExtractor():

    ANNOTATION_FILE_PATH = ANNOTATION_FILE_FORMAT % ANNOTATION_TYPE
    RESULT_FILES_PATH = "/data_c/8hirsch/att_mask_extraction/%s/" % RESULTS_FOLDER

    GLOBAL_RGB_IMAGE_STR = "global_rgb"
    GLOBAL_GROUND_TRUTH_MASK_IMAGE_STR = "global_ground_truth_mask"
    LOCAL_RGB_IMAGE_STR = "local_rgb"
    LOCAL_MASK_IMAGE_STR = "local_mask"
    LOCAL_GROUND_TRUTH_MASK_IMAGE_STR = "local_ground_truth_mask"

    COCO = coco.COCO(ANNOTATION_FILE_PATH)
    CATEGORY_IDS = COCO.getCatIds()  # get all categories

    IOU_FILE_PATH = os.path.join(RESULT_FILES_PATH, "ground_truth_ious.json")
    IOU_DICT = {}

    def __init__(self, global_rgb_image, image_id):
        """[summary]

        Args:
            global_rgb_image (np.ndarray): [description]
            global_ground_truth_mask (np.ndarray): [description]
            image_id (int): [description]
        """

        # TODO convert to uint8 array
        self.global_rgb_image = global_rgb_image
        self.width = global_rgb_image.shape[0]
        self.height = global_rgb_image.shape[1]
        self.image_id = image_id

        self.path = os.path.join(self.RESULT_FILES_PATH, format(image_id, '012d'))
        try:
            os.mkdir(self.path)
        except OSError:
            if not os.path.isdir(self.path):
                raise

        anns_ids = self.COCO.getAnnIds(imgIds=image_id, catIds=self.CATEGORY_IDS, iscrowd=None)
        anns = self.COCO.loadAnns(anns_ids)
        anns_img = np.zeros((self.width, self.height))
        for ann in anns:
            # TODO this causes an error sometimes
            anns_img = np.maximum(anns_img, self.COCO.annToMask(ann))

        self.global_ground_truth_mask_1 = anns_img.astype(np.uint8)
        self.global_ground_truth_mask_255 = np.zeros_like(self.global_ground_truth_mask_1)
        self.global_ground_truth_mask_255[self.global_ground_truth_mask_1 > 0] = 255


        self.save_global_rgb_image()
        self.save_global_ground_truth_mask_image()

    def save_iou(self, proposal_number, global_proposal_mask):
        iou = calc_iou(global_proposal_mask, self.global_ground_truth_mask_1)
        key = format(self.image_id, '012d') + "_" + format(proposal_number, '06d')
        self.IOU_DICT[key] = iou

    @classmethod
    def finalize_iou_file(cls):
        with open(cls.IOU_FILE_PATH, 'w') as filepath:
            json.dump(cls.IOU_DICT, filepath)

    def save_global_rgb_image(self):
        # TODO name it so it will be listed on top
        path = os.path.join(self.path, self.GLOBAL_RGB_IMAGE_STR + ".png")
        save_array_as_image(self.global_rgb_image, path)

    def save_global_ground_truth_mask_image(self):
        path = os.path.join(self.path, self.GLOBAL_GROUND_TRUTH_MASK_IMAGE_STR + ".png")
        save_array_as_image(self.global_ground_truth_mask_255, path)

    def save_local_rgb_image(self, xb, xe, yb, ye, proposal_number):
        local_rgb_image = self.global_rgb_image[xb:xe, yb:ye]

        path = self.build_image_path(proposal_number, self.LOCAL_RGB_IMAGE_STR)
        save_array_as_image(local_rgb_image, path)

    def save_local_ground_truth_mask_image(self, xb, xe, yb, ye, proposal_number):
        local_ground_truth_mask_image = self.global_ground_truth_mask_255[xb:xe, yb:ye]
        self.local_ground_truth_mask_image = local_ground_truth_mask_image

        path = self.build_image_path(proposal_number, self.LOCAL_GROUND_TRUTH_MASK_IMAGE_STR)
        save_array_as_image(local_ground_truth_mask_image, path)

    def save_local_mask_image(self, mask, proposal_number):
        mask = mask.astype(numpy.uint8)
        mask[mask > 0] = 255
        self.local_mask_image = mask

        path = self.build_image_path(proposal_number, self.LOCAL_MASK_IMAGE_STR)
        save_array_as_image(mask, path)

    def build_image_path(self, proposal_number, image_type):
        return os.path.join(self.path, format(proposal_number, '06d') + image_type + ".png")

    # def save_local_images(self, xb, xe, yb, ye, proposal_number):
    #     self.save_local_rgb_image(xb, xe, yb, ye, proposal_number)
    #     self.save_local_mask_image(xb, xe, yb, ye, proposal_number)
    #     self.save_local_ground_truth_mask_image(xb, xe, yb, ye, proposal_number)


def save_array_as_image(image, path):
    image = Image.fromarray(image)
    image.save(path)
