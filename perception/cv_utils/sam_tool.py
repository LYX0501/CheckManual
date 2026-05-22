import numpy as np
from segment_anything import SamPredictor, sam_model_registry, SamAutomaticMaskGenerator
from constants import *
import cv2

def initialize_sam_model(sam_encoder=SAM_ENCODER_VERSION,
                         sam_checkpoint=SAM_CHECKPOINT_PATH,
                         device="cuda:0"):
    sam = sam_model_registry[sam_encoder](checkpoint=sam_checkpoint).to(device)
    sam_prompt_predictor = SamPredictor(sam)
    sam_general_predictor = SamAutomaticMaskGenerator(sam)
    print("Loaded SAM")
    return sam_prompt_predictor, sam_general_predictor

def sam_general_masking(image,bboxes,sam_predictor):
    sam_predictor.set_image(image)
    result_mask = []
    for bbox in bboxes:
        masks,scores,_ = sam_predictor.predict(box=bbox,multimask_output=True)
        mask = masks[np.argmax(scores)]
        result_mask.append(mask)
    return result_mask

def sam_prompt_masking(image,bboxes,sam_predictor):
    sam_predictor.set_image(image)
    result_mask = []
    for bbox in bboxes:
        masks,scores,_ = sam_predictor.predict(box=bbox,multimask_output=True)
        mask = masks[np.argmax(scores)]
        result_mask.append(mask)
    return result_mask
