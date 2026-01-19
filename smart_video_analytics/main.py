import streamlit as st
import cv2
import numpy as np
from PIL import Image
import yaml
import tempfile
from detector import ObjectDetector
from datetime import datetime
import pandas as pd
import os

st.set_page_config(page_title="Smart Video Analytics System", layout="wide")

if "webcam_running" not in st.session_state:
    st.session_state.webcam_running = False

if "video_running" not in st.session_state:
    st.session_state.video_running = False

if "line_counts" not in st.session_state:
    st.session_state.line_counts = {"IN": 0, "OUT": 0}

if "prev_centroids" not in st.session_state:
    st.session_state.prev_centroids = []

if "log_data" not in st.session_state:
    st.session_state.log_data = []

with open("config.yaml", "r") as f:
    cfg_yaml = yaml.safe_load(f)

use_case_options = list(cfg_yaml["features"].keys())
use_case = st.sidebar.selectbox("Select Use-case", use_case_options)
features = cfg_yaml["features"][use_case]

st.sidebar.markdown("### Features Enabled")
for k, v in features.items():
    st.sidebar.write(f"{k}: {v}")

confidence = st.sidebar.slider("Confidence Threshold", 0.1, 1.0, 0.4, 0.05)
detector = ObjectDetector(confidence=confidence)

tab1, tab2, tab3, tab4 = st.tabs(
    ["Webcam", "Image Upload", "Video Upload", "Analytics"]
)

with tab1:
    st.header("Webcam Capture")
    col1, col2 = st.columns(2)
    if col1.button("Start Webcam"):
        st.session_state.webcam_running = True
    if col2.button("Stop Webcam"):
        st.session_state.webcam_running = False

    if st.session_state.webcam_running:
        cam_input = st.camera_input("Capture Frame")
        if cam_input:
            img = cv2.cvtColor(np.array(Image.open(cam_input)), cv2.COLOR_RGB2BGR)
            detections = detector.detect(img)

            person_count = sum(1 for d in detections if d["label"] == "person")
            cv2.putText(img, f"People: {person_count}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            for d in detections:
                x1, y1, x2, y2 = d["bbox"]
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

            st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

with tab2:
    st.header("Upload Image")
    uploaded = st.file_uploader("Choose Image", type=["jpg", "png", "jpeg"])
    if uploaded:
        img = cv2.cvtColor(np.array(Image.open(uploaded)), cv2.COLOR_RGB2BGR)
        detections = detector.detect(img)

        person_count = sum(1 for d in detections if d["label"] == "person")
        cv2.putText(img, f"People: {person_count}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        for d in detections:
            x1, y1, x2, y2 = d["bbox"]
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

        st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

with tab3:
    st.header("Upload Video")
    uploaded_video = st.file_uploader("Upload Video", type=["mp4", "avi", "mov"])

    col1, col2 = st.columns(2)
    if col1.button("Start Video"):
        st.session_state.video_running = True
    if col2.button("Stop Video"):
        st.session_state.video_running = False

    if uploaded_video and st.session_state.video_running:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_video.read())
        cap = cv2.VideoCapture(tfile.name)
        stframe = st.empty()
        frame_id = 0
        frame_skip = 2

        while cap.isOpened() and st.session_state.video_running:
            ret, frame = cap.read()
            if not ret:
                break

            frame_id += 1
            if frame_id % frame_skip != 0:
                continue

            detections = detector.detect(frame)
            h, w, _ = frame.shape
            line_y = int(h * 0.5)

            if features.get("line_crossing"):
                cv2.line(frame, (0, line_y), (w, line_y), (255, 0, 0), 2)

            current_centroids = []
            person_count = 0

            for d in detections:
                if d["label"] == "person":
                    person_count += 1
                    x1, y1, x2, y2 = d["bbox"]
                    cx = int((x1 + x2) / 2)
                    cy = int((y1 + y2) / 2)
                    current_centroids.append(cy)

                    for prev_cy in st.session_state.prev_centroids:
                        if prev_cy < line_y and cy >= line_y:
                            st.session_state.line_counts["IN"] += 1
                        elif prev_cy > line_y and cy <= line_y:
                            st.session_state.line_counts["OUT"] += 1

                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            st.session_state.prev_centroids = current_centroids

            cv2.putText(frame, f"People: {person_count}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            if features.get("line_crossing"):
                cv2.putText(frame,
                            f"IN: {st.session_state.line_counts['IN']} OUT: {st.session_state.line_counts['OUT']}",
                            (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

            st.session_state.log_data.append({
                "timestamp": datetime.now(),
                "people": person_count,
                "in": st.session_state.line_counts["IN"],
                "out": st.session_state.line_counts["OUT"]
            })

            stframe.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
                          use_container_width=True)

        cap.release()

        if len(st.session_state.log_data) > 0:
            os.makedirs("logs", exist_ok=True)
            df = pd.DataFrame(st.session_state.log_data)
            csv_path = f"logs/analytics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            df.to_csv(csv_path, index=False)
            st.success(f"CSV saved: {csv_path}")

with tab4:
    st.header("Analytics Dashboard")

    if len(st.session_state.log_data) > 0:
        df = pd.DataFrame(st.session_state.log_data)
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        st.subheader("Hourly People Count")
        st.line_chart(df.resample("H", on="timestamp")["people"].mean())
        st.subheader("Daily People Count")
        st.line_chart(df.resample("D", on="timestamp")["people"].mean())
    else:
        st.info("No analytics data yet.")
