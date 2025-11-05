import numpy as np
import cv2
import time
import threading
import os
from flask import Flask, request
import torch
from ultralytics import YOLO
import matplotlib.pyplot as plt  # For COLOURS

# --- 1. Configuration ---
LEFT_CAM_ID = "A"
RIGHT_CAM_ID = "B"
LEFT_CAM_NAME = "left_cam"
RIGHT_CAM_NAME = "right_cam"

# --- 2. Load Your Calibrated Constants ---
# Solved from your 2-point calibration
# (e.g., from 50cm & 80cm data)
C_CONSTANT = 10367.5303
K_OFFSET = -8.9226
TARGET_CLASS = "bottle"  # What object are we looking for?

# --- 3. Global "Mailbox" Variables ---
latest_frame_left = None
latest_frame_right = None
latest_results_to_display = None  # This will hold the annotated frames
frame_lock = threading.Lock()

# --- 4. Load YOLO Model (once) ---
print("Loading YOLOv8 model...")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"--- Using device: {device} ---")
model = YOLO('yolov8n.pt')  # yolov8n.pt is the smallest, fastest model
model.to(device)
print(f"Model loaded. Ready to detect '{TARGET_CLASS}'")

# Get the target class index from the model
TARGET_CLS_INDEX = -1
for i, name in model.names.items():
    if name == TARGET_CLASS:
        TARGET_CLS_INDEX = i
        break
if TARGET_CLS_INDEX == -1:
    print(f"!!! FATAL ERROR: Target class '{TARGET_CLASS}' not found in YOLO model.")
    print("Available classes are:", model.names)
    exit()

# Colors for drawing
COLOURS = [
    tuple(int(colour_hex.strip('#')[i:i + 2], 16) for i in (0, 2, 4))
    for colour_hex in plt.rcParams['axes.prop_cycle'].by_key()['color']
]

# --- 5. The Flask Server Thread ---
app = Flask(__name__)


@app.route("/upload", methods=["POST"])
def upload_image():
    global latest_frame_left, latest_frame_right

    cam_id = request.args.get('cam')
    camera_name = "unknown_cam"

    if cam_id == LEFT_CAM_ID:
        camera_name = LEFT_CAM_NAME
    elif cam_id == RIGHT_CAM_ID:
        camera_name = RIGHT_CAM_NAME

    image_data = request.data
    nparr = np.frombuffer(image_data, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if img is not None:
        # --- IMPORTANT! ---
        # Apply the SAME flips you used during calibration!
        # if camera_name == RIGHT_CAM_NAME:
        #     img = cv2.flip(img, 0)
        # elif camera_name == LEFT_CAM_NAME:
        #     img = cv2.flip(img, 0)

        with frame_lock:
            if camera_name == LEFT_CAM_NAME:
                latest_frame_left = img
            elif camera_name == RIGHT_CAM_NAME:
                latest_frame_right = img

        return "Image received", 200
    else:
        print(f"Error: Received empty image from {cam_id}")
        return "Bad image data", 400


def run_flask_server():
    print(f"Starting Flask server thread... Listening on 0.0.0.0:5000")
    app.run(host='0.0.0.0', port=5000, debug=False)


# --- 6. Detection, Geometry, and Drawing Functions ---

def get_detections_yolo(imgs, score_threshold=0.5):
    """
    Runs the global YOLOv8 model on a list of images and filters for the global target class.
    """
    det = []
    lbls = []

    # Process images
    # We send both images to the GPU at once for batch processing
    results = model(imgs, stream=False, device=device, verbose=False)

    for res in results:
        # Filter for target class and confidence
        boxes = []
        labels = []
        for box in res.boxes:
            if box.conf[0] > score_threshold and box.cls[0] == TARGET_CLS_INDEX:
                boxes.append(box.xyxy[0].cpu().numpy())
                # Store the class name directly
                labels.append(TARGET_CLASS)

        det.append(np.array(boxes))
        lbls.append(np.array(labels))

    return det, lbls


# (All your geometry functions: tlbr_to_center1, get_horiz_dist_corner_tl, etc.)
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
    gamma = 5
    vert_dist = gamma * abs(get_vertic_dist_centre(boxes))
    horiz_dist = get_horiz_dist_centre(boxes)
    horiz_dist[horiz_dist < 0] = beta * abs(horiz_dist[horiz_dist < 0])
    area_diffs = get_area_diffs(boxes) / alpha
    cost = np.array([vert_dist, horiz_dist, area_diffs])
    cost = cost.sum(axis=0)

    # Optional: Add class label check back, but YOLO should only give one class
    if lbls is not None:
        if lbls[0].any() and lbls[1].any():
            for i in range(cost.shape[0]):
                for j in range(cost.shape[1]):
                    if (lbls[0][i] != lbls[1][j]):
                        cost[i, j] += 1000  # High penalty for mismatch
    return cost


def draw_detections(img, det, colours=COLOURS, obj_order=None):
    for i, (tlx, tly, brx, bry) in enumerate(det):
        color_index = i
        if obj_order is not None and i < len(obj_order):
            color_index = obj_order[i]
        color_index %= len(colours)
        c = colours[color_index]
        cv2.rectangle(img, (int(tlx), int(tly)), (int(brx), int(bry)), color=c, thickness=2)


def annotate_class2(img, det, class_map, colours=COLOURS):
    """annotate the class labels with custom text"""
    for i, (tlx, tly, brx, bry) in enumerate(det):
        if i >= len(class_map): continue
        txt = class_map[i]
        offset = 1
        cv2.rectangle(img,
                      (int(tlx) - offset, int(tly) - offset - 12),
                      (int(tlx) - offset + len(txt) * 12, int(tly)),
                      color=colours[i % len(colours)],
                      thickness=cv2.FILLED)
        ff = cv2.FONT_HERSHEY_PLAIN
        cv2.putText(img, txt, (int(tlx), int(tly) - 5), fontFace=ff, fontScale=1.0, color=(255,) * 3)


# --- 7. The Main Disparity & Depth Loop ---
def main_processing_loop():
    global latest_frame_left, latest_frame_right, latest_results_to_display

    print("Starting main processing loop... Press 'q' in OpenCV window to quit.")

    # --- Setup OpenCV Windows ---
    cv2.namedWindow("Stereo YOLO Output")

    start_time = time.time()
    frame_count = 0

    while True:
        frame_l, frame_r = None, None

        with frame_lock:
            # Check for a new, *complete* pair
            if latest_frame_left is not None and latest_frame_right is not None:
                frame_l = latest_frame_left.copy()
                frame_r = latest_frame_right.copy()
                # Clear mailboxes to wait for the *next* pair
                latest_frame_left = None
                latest_frame_right = None

        # Only proceed if we have a new pair
        if frame_l is not None and frame_r is not None:

            # --- This is the core YOLO + Stereo Logic ---

            # 1. Check for shape mismatch
            if frame_l.shape != frame_r.shape:
                print(f"Shape mismatch, skipping: L={frame_l.shape}, R={frame_r.shape}")
                continue

            # 2. Get Detections
            # Note: We don't convert to RGB. YOLO accepts BGR.
            imgs = [frame_l, frame_r]
            det, lbls = get_detections_yolo(imgs, score_threshold=0.5)

            cat_dist = []  # Reset distance list
            tracks_for_drawing = ([], [])  # Reset tracks

            # 3. Only proceed if we have detections in BOTH images
            if det[0].any() and det[1].any():
                sz1 = frame_r.shape[1]
                centre = sz1 / 2

                # 4. Get Cost Matrix
                cost = get_cost(det, lbls=lbls, sz1=centre)

                if cost.size > 0:
                    # 5. Match Objects
                    tracks = scipy.optimize.linear_sum_assignment(cost)

                    # 6. Calculate Disparity
                    dists_tl = get_horiz_dist_corner_tl(det)
                    dists_br = get_horiz_dist_corner_br(det)
                    dctl = get_dist_to_centre_tl(det[0], cntr=centre)
                    dcbr = get_dist_to_centre_br(det[0], cntr=centre)

                    final_dists = []
                    for i, j in zip(*tracks):
                        if dctl[i] < dcbr[i]:
                            final_dists.append((dists_tl[i][j], lbls[0][i]))
                        else:
                            final_dists.append((dists_br[i][j], lbls[0][i]))

                    # 7. Calculate Distance
                    if final_dists:
                        valid_disparities = [(d, lbl) for (d, lbl) in final_dists if d > 0]
                        if valid_disparities:
                            fd = np.array([d for (d, lbl) in valid_disparities])
                            dists_away = C_CONSTANT / fd + K_OFFSET

                            # Create labels
                            for i in range(len(dists_away)):
                                cat_dist.append(f'{valid_disparities[i][1]} {dists_away[i]:.1f}cm')

                            # Save tracks for drawing
                            valid_tracks_0 = []
                            valid_tracks_1 = []
                            for i, j in zip(*tracks):
                                disparity = dists_tl[i][j] if dctl[i] < dcbr[i] else dists_br[i][j]
                                if disparity > 0:
                                    valid_tracks_0.append(i)
                                    valid_tracks_1.append(j)
                            tracks_for_drawing = (np.array(valid_tracks_0), np.array(valid_tracks_1))

            # --- 8. Drawing Results ---

            # Draw raw detections on both frames
            if det[0].any():
                draw_detections(frame_l, det[0].astype(np.int32))
            if det[1].any():
                draw_detections(frame_r, det[1].astype(np.int32))

            # Draw distance annotations if we have them
            if cat_dist and tracks_for_drawing[0].size > 0:
                t1_drawing = [list(tracks_for_drawing[1]), list(tracks_for_drawing[0])]
                for i_img, frame in enumerate([frame_l, frame_r]):
                    deti = det[i_img].astype(np.int32)
                    track_list = list(tracks_for_drawing[i_img])
                    if track_list:
                        annotate_class2(frame, deti[track_list], cat_dist)

            # --- 9. Update FPS and Display ---
            frame_count += 1
            elapsed_time = time.time() - start_time
            fps = frame_count / elapsed_time

            cv2.putText(frame_l, f"FPS: {fps:.2f}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # Combine for display
            combined_image = np.hstack([frame_l, frame_r])

            with frame_lock:
                latest_results_to_display = combined_image.copy()

        # --- 10. Show the latest processed result ---
        with frame_lock:
            if latest_results_to_display is not None:
                cv2.imshow("Stereo YOLO Output", latest_results_to_display)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

        time.sleep(0.001)  # 1ms sleep to be CPU-friendly

    cv2.destroyAllWindows()
    print("Processing loop stopped.")


# --- 8. Start Everything ---
if __name__ == "__main__":
    server_thread = threading.Thread(target=run_flask_server, daemon=True)
    server_thread.start()

    main_processing_loop()

    print("Program shutting down.")

