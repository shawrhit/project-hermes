import copy
import math
import numpy as np
import cv2
import matplotlib.pyplot as plt
import scipy
import scipy.optimize
import torch
import torchvision
import torchvision.transforms.functional as tvtf
from torchvision.models.detection import MaskRCNN_ResNet50_FPN_V2_Weights
from pathlib import Path

# --- Constants ---

# these colours are used to draw boxes.
COLOURS = [
    tuple(int(colour_hex.strip('#')[i:i + 2], 16) for i in (0, 2, 4))
    for colour_hex in plt.rcParams['axes.prop_cycle'].by_key()['color']
]

# we use the default weights
weights = MaskRCNN_ResNet50_FPN_V2_Weights.DEFAULT
CLASS_NAMES = np.array(weights.meta["categories"])


# --- Helper Functions (Copied from find_distance.py) ---

def load_img(filename):
    """load image from file"""
    img = cv2.imread(filename)
    if img is None:
        print(f"Error: Could not load image from {filename}")
        return None
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def preprocess_image(image):
    """preprocess image for input into mask rcnn model"""
    image = tvtf.to_tensor(image)
    image = image.unsqueeze(dim=0)
    return image


def get_detections(maskrcnn, imgs, score_threshold=0.5, target_class=None):
    """
    Runs maskrcnn on images and returns detections.
    If target_class is specified (e.g., "bottle"), it only returns detections matching that class.
    """
    det = []
    lbls = []
    scores = []
    masks = []

    # Get the integer label for the target class
    target_label = None
    if target_class:
        try:
            target_label = list(CLASS_NAMES).index(target_class)
        except ValueError:
            print(f"Warning: Target class '{target_class}' not in model categories.")
            target_class = None  # Revert to detecting all

    for img in imgs:
        with torch.no_grad():
            result = maskrcnn(preprocess_image(img))[0]

        # Initial filter by score
        score_mask = result["scores"] > score_threshold

        # If a target class is specified, create a mask for it
        if target_class and target_label is not None:
            label_mask = result["labels"] == target_label
            final_mask = score_mask & label_mask
        else:
            final_mask = score_mask

        boxes = result["boxes"][final_mask].detach().cpu().numpy()
        det.append(boxes)
        lbls.append(result["labels"][final_mask].detach().cpu().numpy())
        scores.append(result["scores"][final_mask].detach().cpu().numpy())
        masks.append(result["masks"][final_mask])

    return det, lbls, scores, masks


# --- Geometry Functions ---

def tlbr_to_center1(boxes):
    points = []
    for tlx, tly, brx, bry in boxes:
        cx = (tlx + brx) / 2
        cy = (tly + bry) / 2
        points.append([cx, cy])
    return points


def tlbr_to_corner(boxes):
    points = []
    for tlx, tly, brx, bry in boxes:
        points.append((tlx, tly))
    return points


def tlbr_to_corner_br(boxes):
    points = []
    for tlx, tly, brx, bry in boxes:
        points.append((brx, bry))
    return points


def tlbr_to_area(boxes):
    areas = []
    for tlx, tly, brx, bry in boxes:
        cx = (brx - tlx)
        cy = (bry - tly)
        areas.append(abs(cx * cy))
    return areas


def get_horiz_dist_centre(boxes):
    pnts1 = np.array(tlbr_to_center1(boxes[0]))[:, 0]
    pnts2 = np.array(tlbr_to_center1(boxes[1]))[:, 0]
    return pnts1[:, None] - pnts2[None]


def get_horiz_dist_corner_tl(boxes):
    pnts1 = np.array(tlbr_to_corner(boxes[0]))[:, 0]
    pnts2 = np.array(tlbr_to_corner(boxes[1]))[:, 0]
    return pnts1[:, None] - pnts2[None]


def get_horiz_dist_corner_br(boxes):
    pnts1 = np.array(tlbr_to_corner_br(boxes[0]))[:, 0]
    pnts2 = np.array(tlbr_to_corner_br(boxes[1]))[:, 0]
    return pnts1[:, None] - pnts2[None]


def get_vertic_dist_centre(boxes):
    pnts1 = np.array(tlbr_to_center1(boxes[0]))[:, 1]
    pnts2 = np.array(tlbr_to_center1(boxes[1]))[:, 1]
    return pnts1[:, None] - pnts2[None]


def get_area_diffs(boxes):
    pnts1 = np.array(tlbr_to_area(boxes[0]))
    pnts2 = np.array(tlbr_to_area(boxes[1]))
    return abs(pnts1[:, None] - pnts2[None])


def get_dist_to_centre_tl(box, cntr):
    pnts = np.array(tlbr_to_corner(box))[:, 0]
    return abs(pnts - cntr)


def get_dist_to_centre_br(box, cntr):
    pnts = np.array(tlbr_to_corner_br(box))[:, 0]
    return abs(pnts - cntr)


# --- Cost and Tracking Functions ---

def get_cost(boxes, lbls=None, sz1=400):
    if not boxes[0].any() or not boxes[1].any():
        return np.array([])
    alpha = sz1
    beta = 10
    gamma = 5
    vert_dist = gamma * abs(get_vertic_dist_centre(boxes))
    horiz_dist = get_horiz_dist_centre(boxes)
    horiz_dist[horiz_dist < 0] = beta * abs(horiz_dist[horiz_dist < 0])
    area_diffs = get_area_diffs(boxes) / alpha
    cost = np.array([vert_dist, horiz_dist, area_diffs])
    cost = cost.sum(axis=0)

    if lbls is not None:
        if lbls[0].any() and lbls[1].any():
            for i in range(cost.shape[0]):
                for j in range(cost.shape[1]):
                    if (lbls[0][i] != lbls[1][j]):
                        cost[i, j] += 150
    return cost


# --- New Calibration Functions ---

def find_disparity_for_pair(model, left_path, right_path, target_class, sz1):
    """
    Loads a pair of images, finds the target object, and returns its disparity.
    """
    left_img = load_img(left_path)
    right_img = load_img(right_path)
    if left_img is None or right_img is None:
        return None

    imgs = [left_img, right_img]
    det, lbls, scores, masks = get_detections(model, imgs, target_class=target_class)

    # Check for errors
    if len(lbls[0]) == 0 or len(lbls[1]) == 0:
        print(f"Error: Target '{target_class}' not found in one or both images.")
        return None
    if len(lbls[0]) > 1 or len(lbls[1]) > 1:
        print(f"Error: Multiple '{target_class}'s found. Please use images with only one.")
        return None

    print(f"Found 1 '{target_class}' in each image. Calculating disparity...")

    # We know there's only one object, so cost/tracking is simple
    cost = get_cost(det, lbls, sz1=sz1)
    if cost.size == 0:
        return None
    tracks = scipy.optimize.linear_sum_assignment(cost)

    # Calculate disparity
    dists_tl = get_horiz_dist_corner_tl(det)
    dists_br = get_horiz_dist_corner_br(det)
    centre = sz1 / 2
    dctl = get_dist_to_centre_tl(det[0], cntr=centre)
    dcbr = get_dist_to_centre_br(det[0], cntr=centre)

    i, j = tracks[0][0], tracks[1][0]  # The only matched pair

    if dctl[i] < dcbr[i]:
        disparity = dists_tl[i][j]
    else:
        disparity = dists_br[i][j]

    if disparity <= 0:
        print(f"Error: Calculated disparity is {disparity}. Check camera setup.")
        return None

    print(f"Calculated disparity: {disparity:.2f} pixels")
    return disparity


def solve_calibration(d1, p1, d2, p2):
    """
    Solves the system of linear equations for D = C/p + K
    (d1, p1) = (distance_near, disparity_near)
    (d2, p2) = (distance_far, disparity_far)
    """
    print("\nSolving system of equations:")
    print(f"1) {d1} = C / {p1} + K")
    print(f"2) {d2} = C / {p2} + K")

    # Solve for C
    # d1 - d2 = C * (1/p1 - 1/p2)
    c_numerator = d1 - d2
    c_denominator = (1 / p1) - (1 / p2)
    C = c_numerator / c_denominator

    # Solve for K
    # K = d1 - C / p1
    K = d1 - (C / p1)

    return C, K


# --- Main Execution ---

def main():
    print("--- Stereo Camera Calibration Script ---")

    # 1. Get user input
    target_class = input("Enter the name of the calibration object (e.g., bottle): ").strip().lower()

    print("\n--- NEAR Point Calibration ---")
    dist_near = float(input("Enter known distance for NEAR object (in cm): "))
    path_left_near = input(f"Enter file path for NEAR left image (e.g., left_50.jpg): ")
    path_right_near = input(f"Enter file path for NEAR right image (e.g., right_50.jpg): ")

    print("\n--- FAR Point Calibration ---")
    dist_far = float(input("Enter known distance for FAR object (in cm): "))
    path_left_far = input(f"Enter file path for FAR left image (e.g., left_80.jpg): ")
    path_right_far = input(f"Enter file path for FAR right image (e.g., right_80.jpg): ")

    # 2. Load Model
    print("\nLoading model...")
    model = torchvision.models.detection.maskrcnn_resnet50_fpn_v2(weights=weights)
    _ = model.eval()
    print("Model loaded.")

    # Get image width (sz1) from the first image
    temp_img = load_img(path_left_near)
    if temp_img is None:
        return
    sz1 = temp_img.shape[1]
    del temp_img  # free memory

    # 3. Process pairs
    print("\nProcessing NEAR pair...")
    disp_near = find_disparity_for_pair(model, path_left_near, path_right_near, target_class, sz1)
    if disp_near is None:
        print("Calibration failed for NEAR pair.")
        return

    print("\nProcessing FAR pair...")
    disp_far = find_disparity_for_pair(model, path_left_far, path_right_far, target_class, sz1)
    if disp_far is None:
        print("Calibration failed for FAR pair.")
        return

    # 4. Solve and print results
    # Your script data: (80cm, 119.05px) and (50cm, 172.93px)
    # Let's check which is near/far
    if dist_near > dist_far:
        # Swap them if user entered far data first
        dist_near, dist_far = dist_far, dist_near
        disp_near, disp_far = disp_far, disp_near

    C_CONSTANT, K_OFFSET = solve_calibration(dist_near, disp_near, dist_far, disp_far)

    print("\n--- CALIBRATION COMPLETE ---")
    print("Your calibration constants are:")
    print(f"C_CONSTANT = {C_CONSTANT:.4f}")
    print(f"K_OFFSET = {K_OFFSET:.4f}")
    print("\nCopy these two lines into your `find_distance.py` script.")


if __name__ == "__main__":
    main()

