import cv2
import mediapipe as mp
import math
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk
import collections
import time
import csv

# Hàm tính góc
def calculate_angle(p1, p2, p3):
    v1 = (p1[0] - p2[0], p1[1] - p2[1])
    v2 = (p3[0] - p2[0], p3[1] - p2[1])
    dot_product = v1[0] * v2[0] + v1[1] * v2[1]
    magnitude_v1 = math.sqrt(v1[0]**2 + v1[1]**2)
    magnitude_v2 = math.sqrt(v2[0]**2 + v2[1]**2)
    angle = math.degrees(math.acos(dot_product / (magnitude_v1 * magnitude_v2 + 1e-8)))
    return round(angle, 2)

# Hàm hiển thị giao diện sau khi ghi xong
def show_analysis_gui(csv_path='knee_angles.csv'):
    # Đọc dữ liệu từ file CSV
    df = pd.read_csv(csv_path)

    # Tính toán góc trung bình và sự khác biệt
    avg_left = df['Left Knee Angle'].mean()
    avg_right = df['Right Knee Angle'].mean()
    diff = abs(avg_left - avg_right)
    warning_frames = df[(df['Left Knee Angle'] < 160) | (df['Right Knee Angle'] < 160)]

    # Khởi tạo giao diện
    root = tk.Tk()
    root.title("Knee Angle Analysis")

    frame = tk.Frame(root, padx=10, pady=10)
    frame.pack()

    # Hiển thị các kết quả thống kê
    tk.Label(frame, text=f"📊 Average Left Knee Angle: {avg_left:.2f}°").pack(anchor='w')
    tk.Label(frame, text=f"📊 Average Right Knee Angle: {avg_right:.2f}°").pack(anchor='w')
    tk.Label(frame, text=f"🔁 Angle Difference: {diff:.2f}°").pack(anchor='w')
    tk.Label(frame, text=f"⚠️ Frames below threshold: {len(warning_frames)}").pack(anchor='w')

    # Biểu đồ
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(df['Frame'], df['Left Knee Angle'], label='Left Knee', color='blue')
    ax.plot(df['Frame'], df['Right Knee Angle'], label='Right Knee', color='green')
    ax.axhline(y=160, color='red', linestyle='--', label='Lower Threshold')
    ax.axhline(y=180, color='red', linestyle='--', label='Upper Threshold')
    ax.set_title("Knee Angles Over Time")
    ax.set_xlabel("Frame")
    ax.set_ylabel("Angle (degrees)")
    ax.legend()
    ax.grid(True)

    canvas = FigureCanvasTkAgg(fig, master=root)
    canvas.get_tk_widget().pack()
    canvas.draw()

    # Thêm khung hiển thị lời khuyên
    advice_frame = tk.Frame(root, padx=10, pady=10, bg="white")
    advice_frame.pack(fill='x', pady=10)
    tk.Label(advice_frame, text="💡 Advice:", font=("Arial", 12, "bold"), bg="white").pack(anchor='w')

    # Đưa ra lời khuyên
    if diff > 10:
        tk.Label(advice_frame, text="⚠️ Significant difference in knee angles detected. Consider balancing your gait.", bg="white").pack(anchor='w')
    if avg_left < 160 or avg_right < 160:
        tk.Label(advice_frame, text="⚠️ Knee angles are below the recommended threshold. Check your posture.", bg="white").pack(anchor='w')
    if avg_left > 180 or avg_right > 180:
        tk.Label(advice_frame, text="⚠️ Knee angles exceed the normal range. Relax your knee posture.", bg="white").pack(anchor='w')
    if len(warning_frames) == 0:
        tk.Label(advice_frame, text="✅ Your gait looks good! Keep it up.", bg="white").pack(anchor='w')

    # Hiển thị giao diện
    root.mainloop()

# Ghi video và tính góc
def run_gait_capture():
    mp_drawing = mp.solutions.drawing_utils
    mp_pose = mp.solutions.pose

    cap = cv2.VideoCapture(0)
    left_buffer = collections.deque(maxlen=5)
    right_buffer = collections.deque(maxlen=5)
    prev_time = 0
    frame_count = 0

    csv_file = open('knee_angles.csv', 'w', newline='')
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(['Frame', 'Left Knee Angle', 'Right Knee Angle'])

    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            results = pose.process(image)
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

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
                left_buffer.append(l_angle)
                right_buffer.append(r_angle)
                smooth_l = round(sum(left_buffer) / len(left_buffer), 2)
                smooth_r = round(sum(right_buffer) / len(right_buffer), 2)

                fps = 1 / (time.time() - prev_time + 1e-8)
                prev_time = time.time()

                cv2.putText(image, f"Left Knee: {smooth_l}°", (40, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
                cv2.putText(image, f"Right Knee: {smooth_r}°", (40, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                cv2.putText(image, f"FPS: {int(fps)}", (40, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

                csv_writer.writerow([frame_count, smooth_l, smooth_r])
            except:
                pass

            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            cv2.imshow("Gait Analysis - Press Q to Stop", image)

            if cv2.waitKey(5) & 0xFF == ord('q'):
                break

    cap.release()
    csv_file.close()
    cv2.destroyAllWindows()

    show_analysis_gui()  # Gọi giao diện khi xong

# Chạy chương trình
if __name__ == "__main__":
    run_gait_capture()