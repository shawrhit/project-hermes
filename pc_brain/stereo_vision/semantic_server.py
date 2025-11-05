# =============================================

import cv2, torch, time, threading, asyncio
import numpy as np
from fastapi import FastAPI, Request
from fastapi.responses import PlainTextResponse
import uvicorn
from ultralytics import YOLO
import scipy.optimize
from queue import Queue

# -----------------------------
# Configuration
# -----------------------------
LEFT_CAM_ID = "A"
RIGHT_CAM_ID = "B"

# Shared queue for uploaded images
frame_queue = Queue(maxsize=20)
latest_frames = {"A": None, "B": None}
frame_lock = threading.Lock()

# Calibration constants
C_CONSTANT = 11496.0111
K_OFFSET = -16.4130

# -----------------------------
# FastAPI Setup
# -----------------------------
app = FastAPI()

@app.post("/upload", response_class=PlainTextResponse)
async def upload_image(request: Request):
    cam_id = request.query_params.get("cam", None)
    if cam_id not in [LEFT_CAM_ID, RIGHT_CAM_ID]:
        return PlainTextResponse("Invalid cam ID", status_code=400)

    data = await request.body()
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        return PlainTextResponse("Bad image", status_code=400)

    # Adjust orientation (modify as per your camera mounting)
    if cam_id == LEFT_CAM_ID:
        img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    elif cam_id == RIGHT_CAM_ID:
        img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)

    with frame_lock:
        latest_frames[cam_id] = img
    try:
        frame_queue.put_nowait((cam_id, time.time()))
    except:
        pass  # drop if queue full (we only need latest)

    return PlainTextResponse("OK", status_code=200)


def run_fastapi():
    uvicorn.run(app, host="0.0.0.0", port=5000, log_level="info")


# -----------------------------
# Helper functions
# -----------------------------
def overlay_mask(frame, mask, color, alpha=0.4):
    if mask.dtype != np.uint8:
        mask = (mask > 0.5).astype(np.uint8)
    colored = np.zeros_like(frame)
    colored[:] = color
    frame[mask == 1] = cv2.addWeighted(frame, 1 - alpha, colored, alpha, 0)[mask == 1]
    return frame

def mask_centroid(mask):
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return 0, 0
    return xs.mean(), ys.mean()


# -----------------------------
# Inference Worker
# -----------------------------
def inference_worker():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    model = YOLO("yolo11l-seg.pt").to(device)
    CLASS_NAMES = np.array(list(model.names.values()))
    print("Model loaded.")

    fps_avg = 0
    color_pool = [(int(a), int(b), int(c)) for a,b,c in np.random.randint(0,255,(100,3))]

    while True:
        # Wait until both frames available
        with frame_lock:
            frameL = latest_frames.get(LEFT_CAM_ID, None)
            frameR = latest_frames.get(RIGHT_CAM_ID, None)

        if frameL is None or frameR is None:
            time.sleep(0.05)
            continue

        start_t = time.time()

        # Run YOLO11m-seg on both frames as batch
        results = model([frameL, frameR], device=device, conf=0.45)
        rL, rR = results[0], results[1]

        boxes_l = rL.boxes.xyxy.cpu().numpy()
        boxes_r = rR.boxes.xyxy.cpu().numpy()
        cls_l = rL.boxes.cls.cpu().numpy().astype(int)
        cls_r = rR.boxes.cls.cpu().numpy().astype(int)

        masks_l = rL.masks.data.cpu().numpy() if rL.masks is not None else []
        masks_r = rR.masks.data.cpu().numpy() if rR.masks is not None else []

        if len(boxes_l)==0 or len(boxes_r)==0:
            combined = np.hstack([frameL, frameR])
            cv2.imshow("Stereo Live", combined)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            continue

        # compute centroids
        cent_l = np.array([mask_centroid(m) for m in masks_l]) if len(masks_l)>0 else (boxes_l[:, :2]+boxes_l[:, 2:4])/2
        cent_r = np.array([mask_centroid(m) for m in masks_r]) if len(masks_r)>0 else (boxes_r[:, :2]+boxes_r[:, 2:4])/2

        # build cost
        dx = cent_l[:,None,0] - cent_r[None,:,0]
        dy = np.abs(cent_l[:,None,1] - cent_r[None,:,1])
        cost = np.abs(dx) + dy
        for i in range(len(cls_l)):
            for j in range(len(cls_r)):
                if cls_l[i] != cls_r[j]:
                    cost[i,j]+=200
        match = scipy.optimize.linear_sum_assignment(cost)

        # compute disparities
        disparities = []
        for i,j in zip(*match):
            disp = cent_l[i,0] - cent_r[j,0]
            disparities.append((i, j, disp))

        # draw results on left frame
        outL = frameL.copy()
        for k, (i,j,disp) in enumerate(disparities):
            if disp <= 0: continue
            dist = C_CONSTANT / disp + K_OFFSET
            color = color_pool[k % len(color_pool)]
            if len(masks_l)>i:
                mask = (masks_l[i] > 0.5).astype(np.uint8)
                overlay_mask(outL, mask, color)
            x1,y1,x2,y2 = boxes_l[i].astype(int)
            cv2.putText(outL, f"{CLASS_NAMES[cls_l[i]]} {dist:.1f}cm",
                        (x1, max(20,y1-10)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

        # display
        combined = np.hstack([outL, frameR])
        fps = 1/(time.time()-start_t+1e-8)
        fps_avg = 0.8*fps_avg + 0.2*fps
        cv2.putText(combined, f"FPS {fps_avg:.1f}", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0),2)
        cv2.imshow("Stereo Live", combined)

        if cv2.waitKey(1)&0xFF==ord('q'):
            break

    cv2.destroyAllWindows()


# -----------------------------
# Main entry
# -----------------------------
if __name__ == "__main__":
    # Start FastAPI server in background thread
    server_thread = threading.Thread(target=run_fastapi, daemon=True)
    server_thread.start()

    # Start inference worker
    inference_worker()
