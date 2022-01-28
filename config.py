'''
Modified version of the original code from Hu et al.

@author Hu et al.
@author Christian Wilms
@date 11/15/18
'''
RGB_MEAN = [123.68, 116.779, 103.939]
IMAGE_PATH_FORMAT = "data/coco/%s/%s"
IMAGE_NAME_FORMAT = "COCO_%s_%012d.jpg"
IMAGE_SET = "subVal2014_2"
ANNOTATION_TYPE = "subVal2014_2"
MY_RESULTS_FILE_NAME = "val.json"
ANNOTATION_FILE_FORMAT = "./data/coco/annotations/instances_%s.json"
SLIDING_WINDOW_SIZE = 10
MASK_SIZE = 40
BOX_SELECTION_PROCESS = 6
