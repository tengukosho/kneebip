# 🦵 Real-Time Knee Flexion/Extension Angle Analyzer

This project is a real-time **knee angle measurement** system developed as a final project for the **Biomedical Image Processing** course. It utilizes **MediaPipe**, **OpenCV**, and **Streamlit** to capture and analyze knee flexion and extension angles via webcam.

---

## 🎯 Project Purpose

This project demonstrates how biomedical image processing can be used to create a non-invasive tool for gait and posture assessment. It provides real-time visualization, data recording, and automated analysis of knee joint angles.

---

## 📽️ Live Demo
👉 [Launch the App on Streamlit Cloud](https://your-app-url.streamlit.app) *(replace with your actual app URL)*

---

## 📦 Features

- 🎥 Real-time webcam capture
- 🧠 Landmark detection using MediaPipe Pose
- 📐 Calculation of left/right knee angles
- 📊 Visual analysis with matplotlib graphs
- 💾 CSV export of data
- 💡 Gait and posture advice based on joint symmetry and thresholds

---

## 🧰 Technologies Used

- **Streamlit** – web interface and app hosting
- **MediaPipe** – 2D pose estimation
- **OpenCV** – video processing
- **Pandas** – data handling
- **Matplotlib** – visualizations

---

## 🛠️ Installation & Setup
pip install -r requirements.txt

### 📁 Clone the repository
```bash
git clone https://github.com/YOUR_USERNAME/knee-angle-analyzer.git
cd knee-angle-analyzer