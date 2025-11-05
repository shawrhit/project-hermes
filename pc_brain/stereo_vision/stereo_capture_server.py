from flask import Flask, request
import numpy as np
import cv2
import time
import threading
import os

# --- 1. Configuration ---

LEFT_CAM_ID = "A"
RIGHT_CAM_ID = "B"
LEFT_CAM_NAME = "left_cam"
RIGHT_CAM_NAME = "right_cam"

# --- SAVE CONFIGURATION ---
SAVE_DIR = "images"
LEFT_DIR = os.path.join(SAVE_DIR, LEFT_CAM_NAME)
RIGHT_DIR = os.path.join(SAVE_DIR, RIGHT_CAM_NAME)

# --- 2. Global "Mailbox" Variables ---
latest_frame_left = None
latest_frame_right = None
frame_lock = threading.Lock()

# --- 3. The Flask Server (runs in a separate thread) ---

# --- MODIFIED: Fixed variable name ---
app = Flask(__name__)


@app.route("/upload", methods=["POST"])
def upload_image():
    global latest_frame_left, latest_frame_right

    # --- MODIFIED: Read camera ID from URL (matches ESP32 code) ---
    cam_id = request.args.get('cam')

    if cam_id == LEFT_CAM_ID:
        camera_name = LEFT_CAM_NAME
    elif cam_id == RIGHT_CAM_ID:
        camera_name = RIGHT_CAM_NAME
    else:
        camera_name = "unknown_cam"
    # --- END MODIFICATION ---

    image_data = request.data
    nparr = np.frombuffer(image_data, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if img is not None:

        if camera_name == RIGHT_CAM_NAME:
            pass
            img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
            # img = cv2.flip(img, 1)  # 0 = vertical flip
        elif camera_name == LEFT_CAM_NAME:
            img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
            # img = cv2.flip(img, 0)  # 1 = horizontal flip

        # -----------------------------------------

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
    # Make sure your PC's firewall allows connections on port 5000
    app.run(host='0.0.0.0', port=5000)


# --- 4. The Main Data Collection Loop (Unchanged) ---
# This part of your code is perfect for calibration.
def main_collection_loop():
    global latest_frame_left, latest_frame_right

    print("Starting image collection loop...")
    print(f"Images will be saved to '{LEFT_DIR}' and '{RIGHT_DIR}'")
    print("\nHold your checkerboard in front of both cameras.")
    print("Press 's' to save the current pair. Press 'q' to quit.")

    # --- Create directories ---
    if not os.path.exists(SAVE_DIR): os.mkdir(SAVE_DIR)
    if not os.path.exists(LEFT_DIR): os.mkdir(LEFT_DIR)
    if not os.path.exists(RIGHT_DIR): os.mkdir(RIGHT_DIR)

    saved_pair_count = 0

    while True:
        frame_l_copy, frame_r_copy = None, None

        with frame_lock:
            if latest_frame_left is not None:
                frame_l_copy = latest_frame_left.copy()
            if latest_frame_right is not None:
                frame_r_copy = latest_frame_right.copy()

        # Display the live feeds
        if frame_l_copy is not None:
            cv2.imshow("Live Left (Press 's' to save)", frame_l_copy)
        if frame_r_copy is not None:
            cv2.imshow("Live Right (Press 's' to save)", frame_r_copy)

        key = cv2.waitKey(1) & 0xFF

        if key == ord('s'):
            frame_l_save, frame_r_save = None, None

            with frame_lock:
                # We check that *both* frames are new
                if latest_frame_left is not None and latest_frame_right is not None:
                    print("Both frames available. Saving...")
                    frame_l_save = latest_frame_left.copy()
                    frame_r_save = latest_frame_right.copy()

                    # Set to None to indicate we're waiting for a *new* pair
                    latest_frame_left = None
                    latest_frame_right = None
                else:
                    print("Waiting for both frames... Press 's' again when both windows are updated.")

            if frame_l_save is not None and frame_r_save is not None:
                # Use a timestamp for a unique filename
                filename = f"{int(time.time() * 1000)}.jpg"
                save_path_l = os.path.join(LEFT_DIR, filename)
                save_path_r = os.path.join(RIGHT_DIR, filename)

                cv2.imwrite(save_path_l, frame_l_save)
                cv2.imwrite(save_path_r, frame_r_save)

                saved_pair_count += 1
                print(f"SAVED pair {saved_pair_count}: {filename}")

        if key == ord('q'):
            break

        time.sleep(0.01)  # Small sleep to be CPU-friendly

    cv2.destroyAllWindows()
    print(f"Collection stopped. Total pairs saved: {saved_pair_count}")


# --- 5. Start Everything (Unchanged) ---
if __name__ == "__main__":
    # Start the Flask server in a background thread
    server_thread = threading.Thread(target=run_flask_server, daemon=True)
    server_thread.start()

    # Run the main collection loop in the main thread
    main_collection_loop()

    print("Program shutting down.")