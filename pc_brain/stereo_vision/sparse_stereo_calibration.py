import numpy as np
import cv2
import time
import os
import torch
from ultralytics import YOLO
import matplotlib.pyplot as plt
import scipy.optimize

# --- 1. Configuration ---
# --- !! IMPORTANT: SET YOUR MODEL PATH !! ---
YOLO_MODEL_PATH = 'yolo11m.pt'  # Your YOLO model
CONF_THRESHOLD = 0.4  # Detection confidence

# --- !! IMPORTANT: SET YOUR IMAGE SHAPE (Width, Height) !! ---
# This MUST match the resolution of your ESP32-CAMs
IMAGE_SHAPE = (480, 640)  # Correct (Width, Height) format for cv2.resize

# --- 2. Load YOLO Model (once) ---
print(f"Loading YOLO model: {YOLO_MODEL_PATH}")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"--- Using device: {device} ---")
try:
    model = YOLO(YOLO_MODEL_PATH)
    model.to(device)
    CLASS_NAMES_MAP = model.names
    print("Model loaded.")
except Exception as e:
    print(f"\n--- FATAL ERROR ---")
    print(f"Could not load YOLO model '{YOLO_MODEL_PATH}'. Error: {e}")
    print("Make sure the file exists and 'ultralytics' is installed.")
    print("-------------------")
    exit()


# --- 3. Geometry & Cost Functions ---

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
    if not boxes[0].any() or not boxes[1].any(): return np.array([])
    pnts1 = np.array(tlbr_to_center1(boxes[0]))[:, 0]
    pnts2 = np.array(tlbr_to_center1(boxes[1]))[:, 0]
    return pnts1[:, None] - pnts2[None]


def get_horiz_dist_corner_tl(boxes):
    if not boxes[0].any() or not boxes[1].any(): return np.array([])
    pnts1 = np.array(tlbr_to_corner(boxes[0]))[:, 0]
    pnts2 = np.array(tlbr_to_corner(boxes[1]))[:, 0]
    return pnts1[:, None] - pnts2[None]


def get_horiz_dist_corner_br(boxes):
    if not boxes[0].any() or not boxes[1].any(): return np.array([])
    pnts1 = np.array(tlbr_to_corner_br(boxes[0]))[:, 0]
    pnts2 = np.array(tlbr_to_corner_br(boxes[1]))[:, 0]
    return pnts1[:, None] - pnts2[None]


def get_vertic_dist_centre(boxes):
    if not boxes[0].any() or not boxes[1].any(): return np.array([])
    pnts1 = np.array(tlbr_to_center1(boxes[0]))[:, 1]
    pnts2 = np.array(tlbr_to_center1(boxes[1]))[:, 1]
    return pnts1[:, None] - pnts2[None]


def get_area_diffs(boxes):
    if not boxes[0].any() or not boxes[1].any(): return np.array([])
    pnts1 = np.array(tlbr_to_area(boxes[0]))
    pnts2 = np.array(tlbr_to_area(boxes[1]))
    return abs(pnts1[:, None] - pnts2[None])


def get_dist_to_centre_tl(box, cntr):
    if not box.any(): return np.array([])
    pnts = np.array(tlbr_to_corner(box))[:, 0]
    return abs(pnts - cntr)


def get_dist_to_centre_br(box, cntr):
    if not box.any(): return np.array([])
    pnts = np.array(tlbr_to_corner_br(box))[:, 0]
    return abs(pnts - cntr)


def get_cost(boxes, lbls=None, sz1=400):
    if not boxes[0].any() or not boxes[1].any():
        return np.array([])
    alpha = sz1
    beta = 10
    gamma = 5  # Vertical distance penalty

    vert_dist = gamma * abs(get_vertic_dist_centre(boxes))
    horiz_dist = get_horiz_dist_centre(boxes)
    # Penalize moving left (negative disparity)
    horiz_dist[horiz_dist < 0] = beta * abs(horiz_dist[horiz_dist < 0])
    area_diffs = get_area_diffs(boxes) / alpha

    cost = np.array([vert_dist, horiz_dist, area_diffs])
    cost = cost.sum(axis=0)

    # Add high penalty for mismatching classes
    if lbls is not None:
        if lbls[0].any() and lbls[1].any():
            for i in range(cost.shape[0]):
                for j in range(cost.shape[1]):
                    if (lbls[0][i] != lbls[1][j]):
                        cost[i, j] += 1000
    return cost


def get_detections_yolo_raw(results, target_class, conf_threshold):
    det_list = []
    lbls_list = []

    for res in results:
        boxes = []
        labels = []
        for box in res.boxes:
            score = float(box.conf[0])
            class_id = int(box.cls[0])
            class_name = CLASS_NAMES_MAP.get(class_id, "Unknown")

            if score > conf_threshold and class_name == target_class:
                boxes.append(box.xyxy[0].cpu().numpy())
                labels.append(class_name)

        det_list.append(np.array(boxes))
        lbls_list.append(np.array(labels))

    return det_list, lbls_list


# --- 4. Main Processing Function ---

def get_disparity_for_object(left_img_path, right_img_path, target_class):
    global model, CLASS_NAMES_MAP

    print(f"\n  Processing pair for '{target_class}':")
    print(f"    L: {left_img_path}")
    print(f"    R: {right_img_path}")

    frame_l = cv2.imread(left_img_path)
    frame_r = cv2.imread(right_img_path)

    if frame_l is None or frame_r is None:
        print("    ERROR: Could not load images. Skipping.")
        return None

    frame_l = cv2.resize(frame_l, IMAGE_SHAPE)
    frame_r = cv2.resize(frame_r, IMAGE_SHAPE)

    results = model([frame_l, frame_r], device=device, verbose=False)
    det, lbls = get_detections_yolo_raw(results, target_class, CONF_THRESHOLD)

    if not det[0].any():
        print(f"    ERROR: No '{target_class}' found in LEFT image.")
        return None
    if not det[1].any():
        print(f"    ERROR: No '{target_class}' found in RIGHT image.")
        return None

    print(f"    Found {len(det[0])} in Left, {len(det[1])} in Right.")

    sz1 = frame_l.shape[1]
    centre = sz1 / 2
    cost = get_cost(det, lbls=lbls, sz1=sz1)

    if cost.size == 0:
        print("    ERROR: Cost matrix is empty, cannot match.")
        return None

    tracks = scipy.optimize.linear_sum_assignment(cost)

    dists_tl = get_horiz_dist_corner_tl(det)
    dists_br = get_horiz_dist_corner_br(det)
    dctl = get_dist_to_centre_tl(det[0], cntr=centre)
    dcbr = get_dist_to_centre_br(det[0], cntr=centre)

    final_dists = []
    for i, j in zip(*tracks):
        # We know the classes match because get_cost penalizes mismatches
        if dctl[i] < dcbr[i]:
            disparity = dists_tl[i][j]
        else:
            disparity = dists_br[i][j]

        if disparity > 0:
            final_dists.append(disparity)
            print(f"    Found a match for '{target_class}' with disparity: {disparity:.2f} px")

    if not final_dists:
        print(f"    ERROR: Could not find a valid match (positive disparity) for '{target_class}'.")
        return None

    # Use the median disparity for robustness if multiple objects are found
    best_disparity = np.median(final_dists)
    print(f"    Using median disparity: {best_disparity:.2f} px")
    return best_disparity


def solve_calibration_robust(all_points):
    """
    Solves D = C/d + K using linear regression (np.polyfit).
    all_points: A list of (disparity, distance) tuples.
    """
    disparities = np.array([p[0] for p in all_points])
    distances = np.array([p[1] for p in all_points])

    # We fit: D = m * (1/d) + c
    inv_disparities = 1.0 / disparities

    m, c = np.polyfit(inv_disparities, distances, 1)

    C = m  # The slope is our C_CONSTANT
    K = c  # The y-intercept is our K_OFFSET

    print("\n--- Solving with Linear Regression ---")
    print(f"  Used {len(all_points)} data points.")
    print("  Model: Distance = C * (1/disparity) + K")
    print(f"  Best-fit C (slope): {C:.4f}")
    print(f"  Best-fit K (y-intercept): {K:.4f}")

    # --- Show plot ---
    plt.figure(figsize=(10, 6))
    plt.scatter(inv_disparities, distances, color='red', label=f'Data Points ({len(all_points)} total)')

    x_fit = np.linspace(min(inv_disparities), max(inv_disparities), 100)
    y_fit = m * x_fit + c

    plt.plot(x_fit, y_fit, color='blue', label=f'Fit: D = {C:.2f}*(1/d) + {K:.2f}')
    plt.xlabel('1 / Disparity (1/px)')
    plt.ylabel('Distance (cm)')
    plt.title('Calibration: Linear Regression (Raw Images)')
    plt.legend()
    plt.grid(True)
    print("\nDisplaying calibration plot... Close the plot to continue.")
    plt.show(block=True)

    return C, K


def main():
    print("--- Robust 6-Point Calibration Finder (RAW IMAGES) ---")
    print("This tool will find C and K using the 'detect-detect-match' method.")

    # --- 1. Get Calibration Object ---
    target_class = input("Enter the target object class for calibration (e.g., 'bottle'): ").strip().lower()
    if target_class not in CLASS_NAMES_MAP.values():
        print(f"Warning: '{target_class}' is not in the model's standard class list.")
        print("Available classes:", list(CLASS_NAMES_MAP.values()))
        if input("Continue anyway? (y/n): ").lower() != 'y':
            return

    # --- 2. Define Calibration Points ---
    CALIBRATION_SET = [
        (40.0, "calibration_images/left40.jpg", "calibration_images/right40.jpg"),
        (50.0, "calibration_images/left50.jpg", "calibration_images/right50.jpg"),
        (60.0, "calibration_images/left60.jpg", "calibration_images/right60.jpg"),
        (80.0, "calibration_images/left80.jpg", "calibration_images/right80.jpg"),
        (120.0, "calibration_images/left120.jpg", "calibration_images/right120.jpg"),
        (150.0, "calibration_images/left150.jpg", "calibration_images/right150.jpg"),
    ]
    all_points = []  # (disparity, distance)

    for dist, left_path, right_path in CALIBRATION_SET:
        disparity = get_disparity_for_object(left_path, right_path, target_class)
        if disparity is not None:
            all_points.append((disparity, dist))
        else:
            print(f"  SKIPPING point for {dist}cm due to detection error.")

    if len(all_points) < 2:
        print("\n--- FATAL ERROR ---")
        print("Need at least 2 valid data points to perform calibration.")
        print("Please check your images, paths, and YOLO model.")
        print("-------------------")
        return

    # --- 4. Solve ---
    C, K = solve_calibration_robust(all_points)

    print("\n--- CALIBRATION COMPLETE ---")
    print("Your calibrated formula is: Distance = C / disparity + K")
    print("\nCopy these values into your 'raw image' server script:")
    print(f"C_CONSTANT = {C:.4f}")
    print(f"K_OFFSET = {K:.4f}")
    print("---------------------------------")

    # --- 5. Test with new values ---
    print("\n(Optional) Test formula with new values:")
    while True:
        try:
            test_disp_str = input("Enter a disparity (px) to test (or 'q' to quit): ")
            if test_disp_str.lower() == 'q':
                break
            test_disp = float(test_disp_str)
            if test_disp <= 0:
                print("  Disparity must be a positive number.")
                continue

            test_dist = (C / test_disp) + K
            print(f"  -> Calculated distance: {test_dist:.2f} cm")
        except ValueError:
            print("  Invalid input. Please enter a number or 'q'.")
        except Exception as e:
            print(f"An error occurred: {e}")

    print("Exiting.")


if __name__ == "__main__":
    main()
