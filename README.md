# MaixCamPro Embedded Vision Target Tracker

基于 **MaixCamPro + OpenCV** 的实时目标检测与跟踪系统。

本项目利用嵌入式视觉算法对摄像头采集图像进行实时处理，实现目标识别、锁定、跟踪，并通过 UART 串口将目标坐标发送至 MCU，用于后续运动控制或激光追踪应用。


## ✨ Features

- 📷 Real-time image acquisition using MaixCamPro camera
- 🔍 OpenCV-based target detection
- 🎯 Automatic target locking and tracking
- 📉 EMA filtering for coordinate smoothing
- 🚀 Motion prediction when target is temporarily lost
- ⚡ UART communication with MCU
- 🖥️ Real-time display of tracking results


## 🛠️ Hardware

- MaixCamPro AI Camera
- MCU control board (UART communication)
- Embedded control system


## 💻 Software

- Python
- OpenCV
- NumPy
- MaixPy


## 🔧 System Workflow

