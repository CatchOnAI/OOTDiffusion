import pdb

import numpy as np
import cv2
from PIL import Image, ImageDraw

label_map = {
    "background": 0,
    "hat": 1,
    "hair": 2,
    "sunglasses": 3,
    "upper_clothes": 4,
    "skirt": 5,
    "pants": 6,
    "dress": 7,
    "belt": 8,
    "left_shoe": 9,
    "right_shoe": 10,
    "head": 11,
    "left_leg": 12,
    "right_leg": 13,
    "left_arm": 14,
    "right_arm": 15,
    "bag": 16,
    "scarf": 17,
}

def extend_arm_mask(wrist, elbow, scale):
  wrist = elbow + scale * (wrist - elbow)
  return wrist

def hole_fill(img):
    img = np.pad(img[1:-1, 1:-1], pad_width = 1, mode = 'constant', constant_values=0)
    img_copy = img.copy()
    mask = np.zeros((img.shape[0] + 2, img.shape[1] + 2), dtype=np.uint8)

    cv2.floodFill(img, mask, (0, 0), 255)
    img_inverse = cv2.bitwise_not(img)
    dst = cv2.bitwise_or(img_copy, img_inverse)
    return dst

def refine_mask(mask):
    contours, hierarchy = cv2.findContours(mask.astype(np.uint8),
                                           cv2.RETR_CCOMP, cv2.CHAIN_APPROX_TC89_L1)
    area = []
    for j in range(len(contours)):
        a_d = cv2.contourArea(contours[j], True)
        area.append(abs(a_d))
    refine_mask = np.zeros_like(mask).astype(np.uint8)
    if len(area) != 0:
        i = area.index(max(area))
        cv2.drawContours(refine_mask, contours, i, color=255, thickness=-1)

    return refine_mask

def get_mask_location(model_type, category, model_parse: Image.Image, keypoint: dict, width=384,height=512):
    im_parse = model_parse.resize((width, height), Image.NEAREST)
    parse_array = np.array(im_parse)

    if model_type == 'hd':
        arm_width = 60
    elif model_type == 'dc':
        arm_width = 45
    else:
        raise ValueError("model_type must be \'hd\' or \'dc\'!")

    parse_head = (parse_array == 1).astype(np.float32) + \
                 (parse_array == 3).astype(np.float32) + \
                 (parse_array == 11).astype(np.float32)

    parser_mask_fixed = (parse_array == label_map["left_shoe"]).astype(np.float32) + \
                        (parse_array == label_map["right_shoe"]).astype(np.float32) + \
                        (parse_array == label_map["hat"]).astype(np.float32) + \
                        (parse_array == label_map["sunglasses"]).astype(np.float32) + \
                        (parse_array == label_map["bag"]).astype(np.float32)

    parser_mask_changeable = (parse_array == label_map["background"]).astype(np.float32)

    arms_left = (parse_array == 14).astype(np.float32)
    arms_right = (parse_array == 15).astype(np.float32)
    arms = arms_left + arms_right

    if category == 'dresses':
        parse_mask = (parse_array == 7).astype(np.float32) + \
                     (parse_array == 4).astype(np.float32) + \
                     (parse_array == 5).astype(np.float32) + \
                     (parse_array == 6).astype(np.float32)

        parser_mask_changeable += np.logical_and(parse_array, np.logical_not(parser_mask_fixed))

    elif category == 'upper_body':
        parse_mask = (parse_array == 4).astype(np.float32) + (parse_array == 7).astype(np.float32)
        parser_mask_fixed_lower_cloth = (parse_array == label_map["skirt"]).astype(np.float32) + \
                                        (parse_array == label_map["pants"]).astype(np.float32)
        parser_mask_fixed += parser_mask_fixed_lower_cloth
        parser_mask_changeable += np.logical_and(parse_array, np.logical_not(parser_mask_fixed))
    elif category == 'lower_body':
        parse_mask = (parse_array == 6).astype(np.float32) + \
                     (parse_array == 12).astype(np.float32) + \
                     (parse_array == 13).astype(np.float32) + \
                     (parse_array == 5).astype(np.float32)
        parser_mask_fixed += (parse_array == label_map["upper_clothes"]).astype(np.float32) + \
                             (parse_array == 14).astype(np.float32) + \
                             (parse_array == 15).astype(np.float32)
        parser_mask_changeable += np.logical_and(parse_array, np.logical_not(parser_mask_fixed))
    else:
        raise NotImplementedError

    # Load pose points
    # pose_data = keypoint["pose_keypoints_2d"]
    # pose_data = np.array(pose_data)
    # pose_data = pose_data.reshape((-1, 2))

    # im_arms_left = Image.new('L', (width, height))
    # im_arms_right = Image.new('L', (width, height))
    # arms_draw_left = ImageDraw.Draw(im_arms_left)
    # arms_draw_right = ImageDraw.Draw(im_arms_right)

    parse_mask = cv2.dilate(parse_mask, np.ones((5, 5), np.uint16), iterations=5)
    if category == 'dresses' or category == 'upper_body':
        neck_mask = (parse_array == 18).astype(np.float32)
        neck_mask = cv2.dilate(neck_mask, np.ones((5, 5), np.uint16), iterations=1)
        neck_mask = np.logical_and(neck_mask, np.logical_not(parse_head))
        parse_mask = np.logical_or(parse_mask, neck_mask)
        # arm_mask = cv2.dilate(np.logical_or(im_arms_left, im_arms_right).astype('float32'), np.ones((5, 5), np.uint16), iterations=4)
        parse_mask += np.logical_or(parse_mask, parse_mask)

    parse_mask = np.logical_and(parser_mask_changeable, np.logical_not(parse_mask))

    parse_mask_total = np.logical_or(parse_mask, parse_mask)
    inpaint_mask = 1 - parse_mask_total
    img = np.where(inpaint_mask, 255, 0)
    dst = hole_fill(img.astype(np.uint8))
    dst = refine_mask(dst)
    inpaint_mask = dst / 255 * 1
    mask = Image.fromarray(inpaint_mask.astype(np.uint8) * 255)
    mask_gray = Image.fromarray(inpaint_mask.astype(np.uint8) * 127)

    return mask, mask_gray
