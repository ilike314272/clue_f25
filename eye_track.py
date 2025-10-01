import cv2
import mediapipe as mp
import numpy as np
from collections import deque

invert_cam = False  # toggle state

# Mediapipe setup
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
)

# Camera
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)   # lower res = faster
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
w, h = int(cap.get(3)), int(cap.get(4))

# Rolling buffer for smoothing
smooth_buffer = deque(maxlen=3)

def smooth_vector(new_vec):
    smooth_buffer.append(new_vec)
    return np.mean(smooth_buffer, axis=0)

def classify_direction(vec, lm, iw, ih, yaw_thresh=12, pitch_thresh=5):
    yaw, pitch = vec

    # Horizontal
    if yaw > yaw_thresh:
        horiz = "RIGHT"
    elif yaw < -yaw_thresh:
        horiz = "LEFT"
    else:
        horiz = "CENTER"

    # Vertical
    pitch = -pitch  # invert sign (camera coords differ)
    if pitch > pitch_thresh:
        vert = "DOWN"
    elif pitch < -pitch_thresh:
        vert = "UP"
    else:
        # Fallback landmark heuristic
        nose_y = lm[1].y * ih
        eye_avg_y = ((lm[33].y + lm[263].y) / 2.0) * ih
        offset = eye_avg_y - nose_y
        if offset > 6:
            vert = "UP"
        elif offset < -6:
            vert = "DOWN"
        else:
            vert = "CENTER"

    return horiz, vert

# 3D model points
model_points = np.array([
    (0.0, 0.0, 0.0),             # Nose tip
    (0.0, -330.0, -65.0),        # Chin
    (-225.0, 170.0, -135.0),     # Left eye left corner
    (225.0, 170.0, -135.0),      # Right eye right corner
    (-150.0, -150.0, -125.0),    # Left mouth corner
    (150.0, -150.0, -125.0)      # Right mouth corner
], dtype=np.float64)

# Camera intrinsics
focal_length = w
center = (w/2, h/2)
cam_mtx = np.array([
    [focal_length, 0, center[0]],
    [0, focal_length, center[1]],
    [0, 0, 1]
], dtype=np.float64)
dist_coeffs = np.zeros((4,1))

print("🔥 Eye tracking started. Press 'q' to quit, 'i' to toggle camera inversion.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    if invert_cam:
        frame = cv2.flip(frame, 1)

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)

    if results.multi_face_landmarks:
        face_landmarks = results.multi_face_landmarks[0]
        ih, iw, _ = frame.shape
        lm = face_landmarks.landmark

        image_points = np.array([
            (lm[1].x*iw, lm[1].y*ih),
            (lm[152].x*iw, lm[152].y*ih),
            (lm[33].x*iw, lm[33].y*ih),
            (lm[263].x*iw, lm[263].y*ih),
            (lm[61].x*iw, lm[61].y*ih),
            (lm[291].x*iw, lm[291].y*ih)
        ], dtype=np.float64)

        success, rvec, tvec = cv2.solvePnP(model_points, image_points, cam_mtx, dist_coeffs)

        if success:
            nose = (int(image_points[0][0]), int(image_points[0][1]))
            gaze_end, _ = cv2.projectPoints(np.array([(0,0,1000.0)]), rvec, tvec, cam_mtx, dist_coeffs)
            gaze_end = (int(gaze_end[0][0][0]), int(gaze_end[0][0][1]))

            rmat, _ = cv2.Rodrigues(rvec)
            angles, _, _, _, _, _ = cv2.RQDecomp3x3(rmat)
            yaw, pitch = angles[1]*180, angles[0]*180

            if invert_cam:
                yaw = -yaw

            vec = smooth_vector((yaw, pitch))
            horiz, vert = classify_direction(vec, lm, iw, ih)

            cv2.putText(frame, f"yaw:{vec[0]:.1f} pitch:{vec[1]:.1f}", (30,h-90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
            cv2.putText(frame, f"Horizontal: {horiz}", (30,h-60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 2)
            cv2.putText(frame, f"Vertical:   {vert}", (30,h-30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)

            cv2.line(frame, nose, gaze_end, (0,0,255), 2)
            cv2.circle(frame, nose, 3, (0,255,0), -1)

    cv2.imshow("Eye Tracking", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('i'):
        invert_cam = not invert_cam
        print("Camera inversion:", "ON" if invert_cam else "OFF")

cap.release()
cv2.destroyAllWindows()
