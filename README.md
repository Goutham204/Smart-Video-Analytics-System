# Smart Video Analytics System

The Smart Video Analytics System is an end-to-end computer vision application for real-time object detection and video analytics. Optimized for CPU-only environments, it supports people counting, ROI-based zone monitoring, and IN/OUT line crossing analytics, making it suitable for malls, offices, parking areas, and public spaces.

# Project Overview

This system enables organizations to monitor crowd movement and object activity in real-time using computer vision. Key capabilities include:

- People Counting: Accurate counting of individuals entering and exiting designated areas
- ROI-Based Zone Monitoring: Monitor specific regions of interest to track activity
- Line Crossing Analytics: Detect IN/OUT movement across defined lines
- Low-Resource Optimization: CPU-only inference using adaptive frame skipping and confidence-based processing

The system includes a professional Streamlit dashboard for visualization and CSV logging, providing actionable insights through time-series analytics.

# Features

- Real-Time Object Detection: YOLOv8-Nano-based inference on images, videos, and live webcam streams
- People Counting & Line Crossing: Centroid tracking for movement detection and analytics
- ROI-Based Alerts: Configurable zones for crowd monitoring and alert generation
- Dashboard & Visualization: Streamlit frontend with start/stop controls and hour/day-wise analytics
- Exportable Results: Timestamped CSV export for further analysis or reporting
- CPU Optimization: Adaptive frame skipping and confidence-based inference for smooth operation on low-resource machines

# Technologies Used

- Programming Language: Python
- Computer Vision / ML: PyTorch, YOLOv8-Nano, OpenCV
- Frontend/UI: Streamlit
- Data Processing & Analytics: pandas, NumPy
- Deployment & Logging: CSV export, Streamlit dashboard

# Output

- Real-time object detection and monitoring
- People counting in designated areas
- Line crossing analytics for IN/OUT movement
- ROI-based crowd alert system
- CSV logs for operational insights

# Ideal Use Cases

- Malls, retail stores, and offices for footfall and crowd monitoring
- Parking lots and public spaces for traffic or people movement analysis
- AI enthusiasts learning real-time computer vision and deployment on CPU-only systems

# License

This project is open-source under the MIT License.
