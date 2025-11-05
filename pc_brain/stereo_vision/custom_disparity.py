import numpy as np
import cv2
import matplotlib.pyplot as plt
import scipy
import scipy.optimize
import torch
from ultralytics import YOLO
import time

# --- Constants ---

# These colours are used to draw boxes.
COLOURS = [
    tuple(int(colour_hex.strip('#')[i:i + 2], 16) for i in (0, 2, 4))
    for colour_hex in plt.rcParams['axes.prop_cycle'].by_key()['color']
]


# Note: CLASS_NAMES will be loaded from the YOLO model in main()

# --- Helper Functions ---

def get_yolo_detections(model, imgs, device, score_threshold=0.5, target_class=None):
    """
    Runs YOLO detection on a batch of images.
    """

    target_label_index = None
    if target_class:
        try:
            # Find the integer key (label index) for the target class name
            for key, value in model.names.items():
                if value == target_class:
                    target_label_index = key
                    break
            if target_label_index is not None:
                print(f"Filtering for target class: {target_class} (model label {target_label_index})")
                filter_classes = [target_label_index]
            else:
                print(f"Warning: Target class '{target_class}' not found in model categories.")
                filter_classes = None
        except Exception as e:
            print(f"Error finding target class: {e}")
            filter_classes = None
    else:
        filter_classes = None

    # Run batched inference
    # YOLO handles moving data to/from the device
    results = model(imgs, device=device, conf=score_threshold, classes=filter_classes)

    det, lbls, scores = [], [], []

    # Process results for each image
    for result in results:
        # Move results to CPU and convert to NumPy
        boxes = result.boxes.xyxy.cpu().numpy()
        labels = result.boxes.cls.cpu().numpy().astype(int)
        confs = result.boxes.conf.cpu().numpy()

        det.append(boxes)
        lbls.append(labels)
        scores.append(confs)

    # Note: YOLO does not return masks
    masks = [[] for _ in imgs]

    return det, lbls, scores, masks


def draw_detections(img, det, colours=COLOURS, obj_order=None):
    """draws the bounding boxes"""
    for i, (tlx, tly, brx, bry) in enumerate(det):
        color_index = i
        if obj_order is not None and i < len(obj_order):
            color_index = obj_order[i]

        color_index %= len(colours)
        c = colours[color_index]
        cv2.rectangle(img, (int(tlx), int(tly)), (int(brx), int(bry)), color=c, thickness=2)


def annotate_class2(img, det, lbls_array, class_map, conf=None, colours=COLOURS):
    """annotate the class labels with custom text"""
    # Note: lbls_array is not used here, class_map has the pre-formatted text
    for i, (tlx, tly, brx, bry) in enumerate(det):
        if i >= len(class_map): continue
        txt = class_map[i]
        if conf is not None:
            txt += f' {conf[i]:1.3f}'
        offset = 1
        cv2.rectangle(img,
                      (int(tlx) - offset, int(tly) - offset + 12),
                      (int(tlx) - offset + len(txt) * 12, int(tly)),
                      color=colours[i % len(colours)],
                      thickness=cv2.FILLED)
        ff = cv2.FONT_HERSHEY_PLAIN
        cv2.putText(img, txt, (int(tlx), int(tly) - 1 + 12), fontFace=ff, fontScale=1.0, color=(255,) * 3)


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
                    # This logic still works, it just compares the integer class IDs
                    if (lbls[0][i] != lbls[1][j]):
                        cost[i, j] += 150
    return cost


def process_stereo_pair(left_image_path, right_image_path, model, device, CLASS_NAMES, c_constant, k_offset):
    """
    Loads one pair of images and returns the annotated frames.
    """
    print(f"\n--- Processing Pair ---")
    print(f"  L: {left_image_path}\n  R: {right_image_path}")

    frame_l = cv2.imread(left_image_path)
    frame_r = cv2.imread(right_image_path)

    if frame_l is None or frame_r is None:
        print("  ERROR: Could not load images. Skipping this pair.")
        return None, None

    if frame_l.shape != frame_r.shape:
        print(f"  ERROR: Image shapes do not match. {frame_l.shape} vs {frame_r.shape}. Skipping.")
        return None, None

    # --- 4. RUN STEREO LOGIC ---
    print("  Processing frames...")
    # YOLO prefers BGR images, so we don't need to convert to RGB
    imgs = [frame_l, frame_r]

    # 1. Get Detections
    start_time = time.time()
    # Pass the device to get_yolo_detections
    det, lbls, scores, masks = get_yolo_detections(model, imgs, device, score_threshold=0.5)
    end_time = time.time()
    print(f"  Detection took {end_time - start_time:.4f} seconds.")

    if not det[0].any() or not det[1].any():
        print("  No objects found in one or both frames. Returning original images.")
        # Return BGR images
        return frame_l, frame_r

    sz1 = frame_r.shape[1]
    centre = sz1 / 2

    # Use the CLASS_NAMES array to print the names
    print(f"  Left Detections: {CLASS_NAMES[lbls[0]]}")
    print(f"  Right Detections: {CLASS_NAMES[lbls[1]]}")

    # 2. Get Cost Matrix
    cost = get_cost(det, lbls=lbls, sz1=centre)
    if cost.size == 0:
        print("  Cost matrix is empty, skipping.")
        return frame_l, frame_r

    # 3. Match Objects
    tracks = scipy.optimize.linear_sum_assignment(cost)

    # 4. Calculate Disparity
    dists_tl = get_horiz_dist_corner_tl(det)
    dists_br = get_horiz_dist_corner_br(det)

    final_dists = []
    dctl = get_dist_to_centre_tl(det[0], cntr=centre)
    dcbr = get_dist_to_centre_br(det[0], cntr=centre)

    for i, j in zip(*tracks):
        # Look up the class name string using the integer ID
        class_name = CLASS_NAMES[lbls[0][i]]
        if dctl[i] < dcbr[i]:
            final_dists.append((dists_tl[i][j], class_name))
        else:
            final_dists.append((dists_br[i][j], class_name))

    # 5. Calculate Distance
    if not final_dists:
        print("  No matched objects.")
        return frame_l, frame_r

    valid_disparities = [(d, lbl) for (d, lbl) in final_dists if d > 0]
    if not valid_disparities:
        print("  No valid positive disparities found.")
        return frame_l, frame_r

    fd = np.array([d for (d, lbl) in valid_disparities])

    # Use the corrected formula
    dists_away = c_constant / fd + k_offset
    cat_dist = []

    print("  --- RESULTS ---")
    for i in range(len(dists_away)):
        label = valid_disparities[i][1]
        dist_cm = dists_away[i]
        cat_dist.append(f'{label} {dist_cm:.1f}cm')
        print(f'  {label} is {dist_cm:.1f}cm away')
    print("  ---------------")

    # 6. Draw Results
    valid_tracks_0 = []
    valid_tracks_1 = []
    for i, j in zip(*tracks):
        disparity = dists_tl[i][j] if dctl[i] < dcbr[i] else dists_br[i][j]
        if disparity > 0:
            valid_tracks_0.append(i)
            valid_tracks_1.append(j)

    if not valid_tracks_0:
        print("  No valid tracks to draw.")
        return frame_l, frame_r

    tracks_for_drawing = (np.array(valid_tracks_0), np.array(valid_tracks_1))
    t1_drawing = [list(tracks_for_drawing[1]), list(tracks_for_drawing[0])]

    frames_ret = []
    # Use the original BGR images for drawing
    for i, imgi in enumerate(imgs):
        img = imgi.copy()  # imgi is already a BGR numpy array
        deti = det[i].astype(np.int32)
        track_list = list(tracks_for_drawing[i])

        # We need the class IDs for the annotate_class2 function
        lbls_i_list = lbls[i][track_list]

        draw_detections(img, deti[track_list], obj_order=list(t1_drawing[i]))
        # Pass the formatted distance strings (cat_dist)
        annotate_class2(img, deti[track_list], lbls_i_list, cat_dist)
        frames_ret.append(img)

    # Images are already BGR
    return frames_ret[0], frames_ret[1]


def main():
    # --- 1. SET CONSTANTS & PATHS ---

    C_CONSTANT = 11496.0111
    K_OFFSET = -16.4130

    IMAGE_PAIRS = [
        {"left": "calibration_images/left50.jpg", "right": "calibration_images/right50.jpg"},
        {"left": "calibration_images/left80.jpg", "right": "calibration_images/right80.jpg"},
        {"left": "calibration_images/left60.jpg", "right": "calibration_images/right60.jpg"},
        {"left": "calibration_images/left150.jpg", "right": "calibration_images/right150.jpg"}
    ]

    # --- 2. SET DEVICE ---
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"--- Using device: {device} ---")
    if device.type == 'cpu':
        print("Warning: CUDA not available. Running on CPU (will be slow).")

    # --- 3. LOAD MODEL ---
    print("Loading YOLO model...")
    # Load the nano model. It will download on first run.
    model = YOLO('yolov8s.pt')

    # Move model to device (YOLO handles this, but good practice)
    model.to(device)
    print("Model loaded.")

    # --- Create CLASS_NAMES array from the loaded model ---
    # model.names is a dict like {0: 'person', 1: 'bicycle', ...}
    # We create a numpy array of the *values*
    CLASS_NAMES = np.array(list(model.names.values()))

    # --- 4. PROCESS IMAGES ---
    processed_pairs = []
    for pair in IMAGE_PAIRS:
        left_frame, right_frame = process_stereo_pair(
            pair["left"],
            pair["right"],
            model,
            device,
            CLASS_NAMES,  # Pass the names array
            C_CONSTANT,
            K_OFFSET
        )
        if left_frame is not None and right_frame is not None:
            # Combine the pair horizontally
            combined_pair = np.hstack([left_frame, right_frame])
            processed_pairs.append(combined_pair)

    if not processed_pairs:
        print("\nNo images were processed successfully. Exiting.")
        return

    # --- 5. DISPLAY FINAL IMAGES ---
    print("\nDisplaying all results in separate windows. Press any key to exit all.")

    for i, img in enumerate(processed_pairs):
        # Show each combined pair (left + right) in its own window
        cv2.imshow(f"Stereo Distance Result {i + 1}", img)

    cv2.waitKey(0)  # Wait indefinitely until any key is pressed

    cv2.destroyAllWindows()
    print("Exited.")


if __name__ == '__main__':
    main()
