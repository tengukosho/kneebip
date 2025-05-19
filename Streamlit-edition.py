import streamlit as st
import cv2
import mediapipe as mp
import math
import pandas as pd
import matplotlib.pyplot as plt
import tempfile
import collections
import time
import csv
import os

# -------------------------
# Calculate knee angle
def calculate_angle(p1, p2, p3):
    v1 = (p1[0] - p2[0], p1[1] - p2[1])
    v2 = (p3[0] - p2[0], p3[1] - p2[1])
    dot = v1[0]*v2[0] + v1[1]*v2[1]
    mag1 = math.sqrt(v1[0]**2 + v1[1]**2)
    mag2 = math.sqrt(v2[0]**2 + v2[1]**2)
    angle = math.degrees(math.acos(dot / (mag1 * mag2 + 1e-8)))
    return round(angle, 2)

# -------------------------
# Analyze CSV data and plot
def show_analysis(csv_path='knee_angles.csv'):
    df = pd.read_csv(csv_path)

    if df.empty:
        st.error("❌ No valid data recorded. Please ensure landmarks are visible during capture.")
        return

    avg_left = df['Left Knee Angle'].mean()
    avg_right = df['Right Knee Angle'].mean()
    diff = abs(avg_left - avg_right)
    warning_frames = df[(df['Left Knee Angle'] < 160) | (df['Right Knee Angle'] < 160)]

    st.subheader("📈 Knee Angle Analysis")
    st.write(f"**📊 Average Left Knee Angle:** {avg_left:.2f}°")
    st.write(f"**📊 Average Right Knee Angle:** {avg_right:.2f}°")
    st.write(f"**🔁 Angle Difference:** {diff:.2f}°")
    st.write(f"**⚠️ Frames below threshold (160°):** {len(warning_frames)}")

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(df['Frame'], df['Left Knee Angle'], label='Left Knee', color='blue')
    ax.plot(df['Frame'], df['Right Knee Angle'], label='Right Knee', color='green')
    ax.axhline(y=160, color='red', linestyle='--', label='Lower Threshold')
    ax.axhline(y=180, color='red', linestyle='--', label='Upper Threshold')
    ax.set_title("Knee Angles Over Time")
    ax.set_xlabel("Frame")
    ax.set_ylabel("Angle (degrees)")
    ax.legend()
    ax.grid(True)
    st.pyplot(fig)

    # Advice section
    st.subheader("💡 Advice:")
    if diff > 10:
        st.warning("⚠️ Significant difference in knee angles detected. Consider balancing your gait.")
    if avg_left < 160 or avg_right < 160:
        st.warning("⚠️ Knee angles are below the recommended threshold. Check your posture.")
    if avg_left > 180 or avg_right > 180:
        st.warning("⚠️ Knee angles exceed the normal range. Relax your knee posture.")
    total_frames = len(df)
    good_frames = total_frames - len(warning_frames)
    if good_frames >= 0.85 * total_frames:
        st.success("✅ Your gait looks good! Keep it up.")
    else:
        st.warning("⚠️ Less than 85% of your frames meet the knee angle threshold. Consider improving your gait.")

# -------------------------
# Run real-time camera and capture angles
def run_gait_capture(output_csv_path):
    mp_drawing = mp.solutions.drawing_utils
    mp_pose = mp.solutions.pose

    cap = cv2.VideoCapture(0)
    left_buffer = collections.deque(maxlen=5)
    right_buffer = collections.deque(maxlen=5)
    prev_time = 0
    frame_count = 0

    stframe = st.empty()
    sttext = st.empty()
    csv_file = open(output_csv_path, 'w', newline='')
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(['Frame', 'Left Knee Angle', 'Right Knee Angle'])

    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while cap.isOpened() and st.session_state.run_camera:
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(image)
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            if not results.pose_landmarks:
                sttext.warning(f"[Frame {frame_count}] ❗ No landmarks detected.")
                stframe.image(image, channels="BGR")
                continue

            try:
                lm = results.pose_landmarks.landmark
                lh = [lm[mp_pose.PoseLandmark.LEFT_HIP.value].x, lm[mp_pose.PoseLandmark.LEFT_HIP.value].y]
                lk = [lm[mp_pose.PoseLandmark.LEFT_KNEE.value].x, lm[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
                la = [lm[mp_pose.PoseLandmark.LEFT_ANKLE.value].x, lm[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]
                rh = [lm[mp_pose.PoseLandmark.RIGHT_HIP.value].x, lm[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
                rk = [lm[mp_pose.PoseLandmark.RIGHT_KNEE.value].x, lm[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]
                ra = [lm[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x, lm[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y]

                l_angle = calculate_angle(lh, lk, la)
                r_angle = calculate_angle(rh, rk, ra)

                if not math.isnan(l_angle) and not math.isnan(r_angle):
                    left_buffer.append(l_angle)
                    right_buffer.append(r_angle)

                    smooth_l = round(sum(left_buffer) / len(left_buffer), 2)
                    smooth_r = round(sum(right_buffer) / len(right_buffer), 2)
                    fps = 1 / (time.time() - prev_time + 1e-8)
                    prev_time = time.time()

                    sttext.text(f"Frame {frame_count}: Left = {smooth_l}°, Right = {smooth_r}°")
                    csv_writer.writerow([frame_count, smooth_l, smooth_r])

                    cv2.putText(image, f"Left Knee: {smooth_l}°", (40, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
                    cv2.putText(image, f"Right Knee: {smooth_r}°", (40, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                    cv2.putText(image, f"FPS: {int(fps)}", (40, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                else:
                    st.warning(f"[Frame {frame_count}] Angle calculation returned NaN.")

            except Exception as e:
                st.error(f"[Frame {frame_count}] Error during processing: {e}")
                continue

            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            stframe.image(image, channels="BGR")

    cap.release()
    csv_file.close()
    cv2.destroyAllWindows()

    show_analysis(csv_path=output_csv_path)

# -------------------------
# Streamlit app layout
st.set_page_config(page_title="Gait Analysis", layout="wide")

st.title("🦵 Knee Flexion/Extension Angle Measurement")

# Session state setup
if 'run_camera' not in st.session_state:
    st.session_state.run_camera = False

# Toggle button
if not st.session_state.run_camera:
    if st.button("▶️ Start Capture"):
        st.session_state.run_camera = True
else:
    if st.button("⏹️ Stop Capture"):
        st.session_state.run_camera = False

# Run camera if flag is set
if st.session_state.run_camera:
    output_path = "knee_angles.csv"
    run_gait_capture(output_path)
