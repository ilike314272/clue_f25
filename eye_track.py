import cv2
import mediapipe as mp
import numpy as np
from collections import deque
import tkinter as tk
from PIL import Image, ImageTk
import time, json

# Settings
invert_cam = True
smooth_window = 5

# Mediapipe
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
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
w, h = int(cap.get(3)), int(cap.get(4))

# Buffers
yaw_pitch_buffer = deque(maxlen=smooth_window)
left_pupil_buffer = deque(maxlen=smooth_window)
right_pupil_buffer = deque(maxlen=smooth_window)

def smooth_value(buffer, new_val):
    buffer.append(new_val)
    return np.mean(buffer, axis=0)

# Model points for head pose
model_points = np.array([
    (0.0, 0.0, 0.0),
    (0.0, -330.0, -65.0),
    (-225.0, 170.0, -135.0),
    (225.0, 170.0, -135.0),
    (-150.0, -150.0, -125.0),
    (150.0, -150.0, -125.0)
], dtype=np.float64)

focal_length = w
center = (w/2, h/2)
cam_mtx = np.array([
    [focal_length, 0, center[0]],
    [0, focal_length, center[1]],
    [0, 0, 1]
], dtype=np.float64)
dist_coeffs = np.zeros((4,1))

# Tkinter GUI
root = tk.Tk()
root.title("🔥 Eye Tracking Desktop App (Press 'i' to invert camera)")

frame_left = tk.Frame(root)
frame_left.pack(side="left")

frame_right = tk.Frame(root)
frame_right.pack(side="right", fill="y")

label = tk.Label(frame_left)
label.pack()

fps_label = tk.Label(frame_left, text="FPS: 0", font=("Arial", 12))
fps_label.pack()

ndjson_box = tk.Text(frame_right, width=40, height=25, font=("Courier", 10))
ndjson_box.pack()

last_time = time.time()

def classify_grid(nose, gaze_end, face_center, grid_w=200, grid_h=160):
    cx, cy = face_center
    dx, dy = gaze_end[0] - cx, gaze_end[1] - cy

    cell_w, cell_h = grid_w // 3, grid_h // 3
    if dx < -cell_w: col = "L"
    elif dx > cell_w: col = "R"
    else: col = "C"
    if dy < -cell_h: row = "U"
    elif dy > cell_h: row = "D"
    else: row = "C"

    return row + col if row+col != "CC" else "C"

def update_frame():
    global invert_cam, last_time
    ret, frame = cap.read()
    if not ret:
        root.after(10, update_frame)
        return

    if invert_cam:
        frame = cv2.flip(frame, 1)

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)

    yaw, pitch = 0, 0
    gaze_zone = "?"
    left_pupil, right_pupil = (0,0), (0,0)

    if results.multi_face_landmarks:
        ih, iw, _ = frame.shape
        lm = results.multi_face_landmarks[0].landmark

        # Draw debug landmarks
        for pt in lm[0:478]:
            x, y = int(pt.x*iw), int(pt.y*ih)
            cv2.circle(frame, (x,y), 1, (100,100,255), -1)

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
            yaw, pitch = smooth_value(yaw_pitch_buffer, (angles[1]*180, angles[0]*180))

            cv2.line(frame, nose, gaze_end, (0,0,255), 2)
            cv2.circle(frame, nose, 3, (0,255,0), -1)

            face_center = (int((lm[33].x+lm[263].x)/2*iw),
                           int((lm[33].y+lm[263].y)/2*ih))

            gaze_zone = classify_grid(nose, gaze_end, face_center)

            # Draw 3×3 grid
            grid_w, grid_h = 200, 160
            cell_w, cell_h = grid_w // 3, grid_h // 3
            for i in range(4):
                x = face_center[0] - grid_w//2 + i*cell_w
                cv2.line(frame, (x, face_center[1]-grid_h//2),
                         (x, face_center[1]+grid_h//2), (200,200,200), 1)
            for j in range(4):
                y = face_center[1] - grid_h//2 + j*cell_h
                cv2.line(frame, (face_center[0]-grid_w//2, y),
                         (face_center[0]+grid_w//2, y), (200,200,200), 1)

        left_pupil = (int(lm[468].x*iw), int(lm[468].y*ih))
        right_pupil = (int(lm[473].x*iw), int(lm[473].y*ih))
        left_pupil = tuple(map(int, smooth_value(left_pupil_buffer, left_pupil)))
        right_pupil = tuple(map(int, smooth_value(right_pupil_buffer, right_pupil)))
        cv2.circle(frame, left_pupil, 4, (0,255,0), -1)
        cv2.circle(frame, right_pupil, 4, (0,255,0), -1)

    # FPS
    now = time.time()
    fps = 1 / (now - last_time)
    last_time = now
    fps_label.config(text=f"FPS: {fps:.1f}")

    # NDJSON STREAM
    data = {
        "yaw": round(float(yaw), 2),
        "pitch": round(float(pitch), 2),
        "left_pupil": left_pupil,
        "right_pupil": right_pupil,
        "gaze_zone": gaze_zone,
        "fps": round(fps, 1)
    }
    ndjson_box.delete("1.0", tk.END)
    ndjson_box.insert(tk.END, json.dumps(data, separators=(",", ":")))

    # Show frame
    img = ImageTk.PhotoImage(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
    label.imgtk = img
    label.config(image=img)

    root.after(1, update_frame)

def on_key(event):
    global invert_cam
    if event.char.lower() == 'i':
        invert_cam = not invert_cam
        print("Camera inversion:", "ON" if invert_cam else "OFF")

root.bind("<Key>", on_key)
update_frame()
root.mainloop()

cap.release()
cv2.destroyAllWindows()
