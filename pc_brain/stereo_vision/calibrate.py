import numpy as np
import cv2
import glob
import os

# --- 1. Configuration ---
CHECKERBOARD = (5, 8)  # Number of internal corners (width, height)
SQUARE_SIZE_MM = 30  # The size of one square in millimeters

# --- 2. Setup ---
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# Prepare 3D "object points" (0,0,0), (1,0,0), (2,0,0) ...
# We scale by SQUARE_SIZE_MM to get real-world units
objp = np.zeros((CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
objp = objp * SQUARE_SIZE_MM  # Scale by square size

# Arrays to store object points and image points from all images
objpoints = []  # 3D points in real world space
imgpoints_left = []  # 2D points in left image plane
imgpoints_right = []  # 2D points in right image plane

# Get image paths
images_left = sorted(glob.glob('images/left_cam/*.jpg'))
images_right = sorted(glob.glob('images/right_cam/*.jpg'))

if not images_left or not images_right:
    print("Error: No images found. Did you run the collection script first?")
    exit()

print(f"Found {len(images_left)} left images and {len(images_right)} right images.")

# Make sure we have the same number of images
if len(images_left) != len(images_right):
    print("Error: Mismatched image counts. Check your 'images' folder.")
    exit()

img_shape = None

# --- 3. Find Corners ---
print("Finding corners in image pairs...")
# Loop through all image pairs
for img_left_path, img_right_path in zip(images_left, images_right):
    img_left = cv2.imread(img_left_path)
    img_right = cv2.imread(img_right_path)

    gray_left = cv2.cvtColor(img_left, cv2.COLOR_BGR2GRAY)
    gray_right = cv2.cvtColor(img_right, cv2.COLOR_BGR2GRAY)

    if img_shape is None:
        img_shape = gray_left.shape[::-1]  # (width, height)

    # Find the chessboard corners
    ret_left, corners_left = cv2.findChessboardCorners(gray_left, CHECKERBOARD, None)
    ret_right, corners_right = cv2.findChessboardCorners(gray_right, CHECKERBOARD, None)

    # --- If corners are found in BOTH images ---
    if ret_left and ret_right:
        print(f"  Found corners in: {os.path.basename(img_left_path)}")

        objpoints.append(objp)

        # Refine corner locations
        corners_left_subpix = cv2.cornerSubPix(gray_left, corners_left, (11, 11), (-1, -1), criteria)
        imgpoints_left.append(corners_left_subpix)

        corners_right_subpix = cv2.cornerSubPix(gray_right, corners_right, (11, 11), (-1, -1), criteria)
        imgpoints_right.append(corners_right_subpix)
    else:
        print(
            f"  Skipping pair (corners not found): {os.path.basename(img_left_path)} | {os.path.basename(img_right_path)}")

cv2.destroyAllWindows()

if not objpoints:
    print("\nError: No valid corner pairs found in any image. Check your CHECKERBOARD size or lighting.")
    exit()

print(f"\nCalibrating using {len(objpoints)} valid pairs...")

# --- 4. Calibrate Cameras (Individually) ---
ret_left, mtx_left, dist_left, rvecs_left, tvecs_left = cv2.calibrateCamera(
    objpoints, imgpoints_left, img_shape, None, None
)

ret_right, mtx_right, dist_right, rvecs_right, tvecs_right = cv2.calibrateCamera(
    objpoints, imgpoints_right, img_shape, None, None
)

print("Individual camera calibration complete.")

# --- 5. Calibrate Stereo Setup ---
print("Starting stereo calibration...")
(ret_stereo, mtx_left, dist_left, mtx_right, dist_right, R, T, E, F) = cv2.stereoCalibrate(
    objpoints, imgpoints_left, imgpoints_right,
    mtx_left, dist_left,
    mtx_right, dist_right,
    img_shape,
    criteria=criteria,
    flags=cv2.CALIB_FIX_INTRINSIC  # Fix intrinsics, as we just found them
)

print("Stereo calibration complete.")

# --- 6. Get Rectification Maps ---
print("Generating rectification maps...")
(R1, R2, P1, P2, Q, roi_left, roi_right) = cv2.stereoRectify(
    mtx_left, dist_left,
    mtx_right, dist_right,
    img_shape, R, T,
    flags=cv2.CALIB_ZERO_DISPARITY,
    alpha=-1  # -1=auto, 0=crop, 1=all pixels
)

print("Rectification complete.")

# --- 7. Save Calibration Data ---
print("Saving calibration data to 'stereo_calib.npz'...")
np.savez(
    "stereo_calib.npz",
    mtx_left=mtx_left,
    dist_left=dist_left,
    mtx_right=mtx_right,
    dist_right=dist_right,
    R=R,
    T=T,
    R1=R1,
    R2=R2,
    P1=P1,
    P2=P2,
    Q=Q,
    img_shape=img_shape
)

print("\nSUCCESS! Calibration file 'stereo_calib.npz' saved.")
print("You are now ready to run the 'processing_server.py' script.")