import cv2
import numpy as np
import mediapipe as mp
import os
import sys

# Configuration
MODEL_IMG = "model.png"
CLOTHES_DIR = "clothes"
OUT = "results"
os.makedirs(OUT, exist_ok=True)

# MediaPipe initializations
mp_pose = mp.solutions.pose
mp_segmentation = mp.solutions.selfie_segmentation 

# Load model and detect landmarks
if not os.path.exists(MODEL_IMG):
    print(f"ERROR: '{MODEL_IMG}' file not found.")
    sys.exit(1)

img = cv2.imread(MODEL_IMG)
H, W = img.shape[:2]

pose = mp_pose.Pose(static_image_mode=True)
segmentation = mp_segmentation.SelfieSegmentation(model_selection=1)

rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
res = pose.process(rgb_img)

if not res.pose_landmarks:
    print("Landmarks not detected. Check image quality.")
    sys.exit(1)

lm = res.pose_landmarks.landmark

# Helper functions

def get_torso_rect(lm):
    sh_l = lm[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
    sh_r = lm[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
    hip_l = lm[mp_pose.PoseLandmark.LEFT_HIP.value]
    hip_r = lm[mp_pose.PoseLandmark.RIGHT_HIP.value]

    x_expand = 0.05
    
    x1 = int((min(sh_l.x, sh_r.x) - x_expand) * W)
    x2 = int((max(sh_l.x, sh_r.x) + x_expand) * W)
    y1 = int(min(sh_l.y, sh_r.y) * H)
    y2 = int(max(hip_l.y, hip_r.y) * H * 1.1) 
    
    x1 = max(0, x1); y1 = max(0, y1)
    x2 = min(W, x2); y2 = min(H, y2)
    
    return x1, y1, x2, y2

def ensure_alpha(cloth):
    if cloth.shape[2] == 4:
        rgb = cloth[:, :, :3]
        alpha = cloth[:, :, 3]
        return rgb, alpha
    
    gray = cv2.cvtColor(cloth, cv2.COLOR_BGR2GRAY)
    _, a = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)
    return cloth, a

def blend(bg, fg, alpha, x, y, body_mask): 
    h, w = fg.shape[:2]
    
    # Limit region of interest
    y_end = min(y + h, bg.shape[0])
    x_end = min(x + w, bg.shape[1])
    y_start = max(0, y)
    x_start = max(0, x)

    roi_h = y_end - y_start
    roi_w = x_end - x_start
    
    if roi_h <= 0 or roi_w <= 0: return bg 
    
    roi = bg[y_start:y_end, x_start:x_end]
    fg_cropped = fg[y_start-y:y_end-y, x_start-x:x_end-x]
    alpha_cropped = alpha[y_start-y:y_end-y, x_start-x:x_end-x]
    
    mask_cropped = body_mask[y_start:y_end, x_start:x_end]
    
    alpha_float = alpha_cropped.astype(float) / 255.0
    alpha_3ch = np.repeat(alpha_float[:, :, None], 3, axis=2)
    
    final_alpha = alpha_3ch * mask_cropped
    
    blended = (roi * (1 - final_alpha) + fg_cropped * final_alpha).astype(np.uint8)
    bg[y_start:y_end, x_start:x_end] = blended
    return bg

def refine_mask(mask_bgr, lm, W, H):
    # Refine mask to remove neck, wrists, and elbows
    sh_l = lm[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
    sh_r = lm[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
    wr_l = lm[mp_pose.PoseLandmark.LEFT_WRIST.value]
    wr_r = lm[mp_pose.PoseLandmark.RIGHT_WRIST.value]
    el_l = lm[mp_pose.PoseLandmark.LEFT_ELBOW.value]
    el_r = lm[mp_pose.PoseLandmark.RIGHT_ELBOW.value]
    
    shoulder_width = abs(sh_r.x - sh_l.x) * W
    
    neck_x = int((sh_l.x + sh_r.x) / 2 * W)
    neck_y = int(min(sh_l.y, sh_r.y) * H - H * 0.01)
    neck_radius = int(shoulder_width * 0.18)
    cv2.circle(mask_bgr, (neck_x, neck_y), neck_radius, (0, 0, 0), -1)

    wrist_radius = int(shoulder_width * 0.06)
    cv2.circle(mask_bgr, (int(wr_l.x * W), int(wr_l.y * H)), wrist_radius, (0, 0, 0), -1)
    cv2.circle(mask_bgr, (int(wr_r.x * W), int(wr_r.y * H)), wrist_radius, (0, 0, 0), -1)
    
    elbow_radius = int(shoulder_width * 0.10)
    cv2.circle(mask_bgr, (int(el_l.x * W), int(el_l.y * H)), elbow_radius, (0, 0, 0), -1)
    cv2.circle(mask_bgr, (int(el_r.x * W), int(el_r.y * H)), elbow_radius, (0, 0, 0), -1)

    return mask_bgr


torso_x1, torso_y1, torso_x2, torso_y2 = get_torso_rect(lm)
torso_w = torso_x2 - torso_x1
torso_h = torso_y2 - torso_y1

segmentation_results = segmentation.process(rgb_img)
condition = np.stack((segmentation_results.segmentation_mask,) * 3, axis=-1) > 0.1

torso_mask_3ch = condition.astype(np.float32)
torso_mask_255 = (torso_mask_3ch * 255).astype(np.uint8)
torso_mask_3ch_refined = refine_mask(torso_mask_255, lm, W, H).astype(np.float32) / 255.0
torso_mask_smooth = cv2.GaussianBlur(torso_mask_3ch_refined, (21, 21), 0)

cv2.imwrite(os.path.join(OUT, 'debug_refined_mask.png'), (torso_mask_smooth[:,:,0]*255).astype(np.uint8))
print("Refined mask created: debug_refined_mask.png")

cloth_files = [f for f in os.listdir(CLOTHES_DIR) if f.lower().endswith((".png",".jpg",".jpeg"))]

if not cloth_files:
    print(f"No clothing images found in '{CLOTHES_DIR}' folder!")
    sys.exit(1)

for idx, f in enumerate(cloth_files):
    cloth = cv2.imread(os.path.join(CLOTHES_DIR, f), cv2.IMREAD_UNCHANGED)
    if cloth is None:
        continue

    rgb, a = ensure_alpha(cloth)

    scale_w = 2
    scale_h = 1.34

    new_w = int(torso_w * scale_w)
    new_h = int(torso_h * scale_h)

    rgb = cv2.resize(rgb, (new_w, new_h))
    a = cv2.resize(a, (new_w, new_h))

    offset_x = torso_x1 - int((new_w - torso_w) / 2)
    offset_y = torso_y1 - int((new_h - torso_h) * 0.60) 

    out = img.copy()
    out = blend(out, rgb, a, offset_x, offset_y, torso_mask_smooth)

    path = os.path.join(OUT, f"tryon_{idx}_{os.path.splitext(f)[0]}.png")
    cv2.imwrite(path, out)
    print("Saved result:", path)

print("Done")
