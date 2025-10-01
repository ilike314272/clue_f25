import cv2
import torch
import torch.nn as nn
import torchvision.models as models
import numpy as np
import mediapipe as mp
import csv, time

# -----------------------------
# Gaze Estimation Model
# -----------------------------
class GazeEstimationModel(nn.Module):
    def __init__(self, backbone="resnet18", pretrained=False):
        super(GazeEstimationModel, self).__init__()
        if backbone == "resnet18":
            base = models.resnet18(weights=None if not pretrained else "IMAGENET1K_V1")
            self.features = nn.Sequential(*list(base.children())[:-1])
            in_dim = base.fc.in_features
        else:
            raise ValueError("Unsupported backbone")
        self.fc = nn.Linear(in_dim, 3)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        gaze = self.fc(x)
        return nn.functional.normalize(gaze, dim=1)

# -----------------------------
# Load model
# -----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = GazeEstimationModel(backbone="resnet18").to(device)
checkpoint = torch.load("ckpt/epoch_24_ckpt.pth.tar", map_location=device)
state_dict = checkpoint.get("model_state", checkpoint.get("state_dict", checkpoint))
state_dict = {k.replace("module.", "") if k.startswith("module.") else k: v for k,v in state_dict.items()}
model.load_state_dict(state_dict, strict=False)
model.eval()
print("✅ Model loaded successfully.")

# -----------------------------
# Face mesh
# -----------------------------
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5, refine_landmarks=True)

LEFT_PUPIL = 468
RIGHT_PUPIL = 473

# -----------------------------
# Preprocessing
# -----------------------------
def preprocess_face(face):
    if face.size == 0:
        return None
    img = cv2.resize(face, (224,224))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    img = (img - mean) / std
    return torch.from_numpy(img.astype(np.float32)).permute(2,0,1).unsqueeze(0).to(device)

# -----------------------------
# Head pose estimation
# -----------------------------
model_points = np.array([
    (0.0, 0.0, 0.0), (0.0, -330.0, -65.0),
    (-225.0, 170.0, -135.0), (225.0, 170.0, -135.0),
    (-150.0, -150.0, -125.0), (150.0, -150.0, -125.0)
], dtype=np.float32)

LANDMARK_IDS = {"nose_tip":1,"chin":152,"left_eye":33,"right_eye":263,"left_mouth":61,"right_mouth":291}

def get_head_pose(frame, landmarks):
    h, w, _ = frame.shape
    image_points = np.array([
        (landmarks[LANDMARK_IDS["nose_tip"]].x * w, landmarks[LANDMARK_IDS["nose_tip"]].y * h),
        (landmarks[LANDMARK_IDS["chin"]].x * w, landmarks[LANDMARK_IDS["chin"]].y * h),
        (landmarks[LANDMARK_IDS["left_eye"]].x * w, landmarks[LANDMARK_IDS["left_eye"]].y * h),
        (landmarks[LANDMARK_IDS["right_eye"]].x * w, landmarks[LANDMARK_IDS["right_eye"]].y * h),
        (landmarks[LANDMARK_IDS["left_mouth"]].x * w, landmarks[LANDMARK_IDS["left_mouth"]].y * h),
        (landmarks[LANDMARK_IDS["right_mouth"]].x * w, landmarks[LANDMARK_IDS["right_mouth"]].y * h),
    ], dtype=np.float32)
    focal_length = w
    cam_matrix = np.array([[focal_length,0,w/2],[0,focal_length,h/2],[0,0,1]],dtype=np.float32)
    dist_coeffs = np.zeros((4,1))
    success,rvec,tvec=cv2.solvePnP(model_points,image_points,cam_matrix,dist_coeffs,flags=cv2.SOLVEPNP_ITERATIVE)
    return success,rvec,tvec,cam_matrix,dist_coeffs

# -----------------------------
# CSV setup
# -----------------------------
csv_file = open("pupil_training_data.csv", "a", newline="")
writer = csv.writer(csv_file)
writer.writerow([
    "timestamp",
    "face_x","face_y","face_w","face_h","face_dist",
    "lp_x","lp_y","lp_size",
    "rp_x","rp_y","rp_size",
    "gaze_x","gaze_y"
])
start_time = time.time()
buffer = []

# -----------------------------
# Webcam loop
# -----------------------------
cap = cv2.VideoCapture(0)
invert_cam = False

while True:
    ret, frame = cap.read()
    if not ret: break
    if invert_cam:
        frame = cv2.flip(frame, 1)

    h,w,_ = frame.shape
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)

    if results.multi_face_landmarks:
        mesh = results.multi_face_landmarks[0].landmark

        success,rvec,tvec,cam_matrix,dist_coeffs=get_head_pose(frame,mesh)
        if not success: continue

        # crop face region
        xs=[int(l.x*w) for l in mesh]; ys=[int(l.y*h) for l in mesh]
        x_min,x_max=max(min(xs)-20,0),min(max(xs)+20,w)
        y_min,y_max=max(min(ys)-20,0),min(max(ys)+20,h)
        face_crop=frame[y_min:y_max,x_min:x_max]
        face_center=((x_min+x_max)//2,(y_min+y_max)//2)
        face_w, face_h = (x_max-x_min), (y_max-y_min)

        # predict gaze
        face_inp=preprocess_face(face_crop)
        if face_inp is None: continue
        with torch.no_grad():
            gaze_vec=model(face_inp).cpu().numpy()[0]

        rot_matrix,_=cv2.Rodrigues(rvec)
        gaze_dir=rot_matrix@gaze_vec.reshape(3,1)
        gaze_end,_=cv2.projectPoints(gaze_dir*300.0+tvec,rvec,tvec,cam_matrix,dist_coeffs)
        gaze_end=tuple(np.int32(gaze_end.reshape(-1,2))[0])

        # pupils
        lp = (int(mesh[LEFT_PUPIL].x*w), int(mesh[LEFT_PUPIL].y*h))
        rp = (int(mesh[RIGHT_PUPIL].x*w), int(mesh[RIGHT_PUPIL].y*h))

        # estimate pupil size (distance between upper/lower eyelid landmarks)
        # MediaPipe eye outline indices for pupil radius proxy
        lp_size = np.linalg.norm([
            mesh[159].x*w - mesh[145].x*w,
            mesh[159].y*h - mesh[145].y*h
        ])
        rp_size = np.linalg.norm([
            mesh[386].x*w - mesh[374].x*w,
            mesh[386].y*h - mesh[374].y*h
        ])

        # visualization
        cv2.circle(frame, lp, 3, (0,0,255), -1)
        cv2.circle(frame, rp, 3, (0,0,255), -1)
        cv2.circle(frame, face_center, 4, (0,255,0), -1)
        cv2.line(frame, face_center, gaze_end, (0,0,255), 2)

        # log data
        face_dist = float(tvec[2]) if success else -1
        buffer.append([
            time.time(),
            face_center[0], face_center[1], face_w, face_h, face_dist,
            lp[0], lp[1], lp_size,
            rp[0], rp[1], rp_size,
            gaze_end[0], gaze_end[1]
        ])

    cv2.imshow("Pupil Training Data Collector", frame)

    # flush buffer every 15s
    if time.time() - start_time >= 15 and buffer:
        writer.writerows(buffer)
        buffer = []
        start_time = time.time()

    key=cv2.waitKey(1)&0xFF
    if key==27: break
    elif key==ord('i'): invert_cam=not invert_cam

cap.release()
csv_file.close()
cv2.destroyAllWindows()
