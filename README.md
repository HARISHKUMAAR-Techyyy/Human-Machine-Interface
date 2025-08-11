# ===============================
# README.md
# ===============================
# Hand Gesture Mouse Control (Python + OpenCV + Mediapipe)

This project lets you control your computer mouse using *hand gestures* detected via your webcam.  
It uses *OpenCV* for video input, *Mediapipe* for landmark detection, and *PyAutoGUI* for mouse control.

---

opencv-python
mediapipe
pyautogui## 📦 Requirements
Create a requirements.txt file with:
opencv-python
mediapipe
pyautogui


---

## ⚙ Installation
Make sure Python 3.7+ is installed.


Clone or download
git clone https://github.com/Vtswamy/Human-Machine-Interface-.git
cd <>
Install dependencies
pip install -r requirements.txt


---

## ▶ Usage
Run:
python Main.py


---

## 🖐 Gestures
- *Move Mouse* → Move your *index finger tip* (landmark 8) in the air.
- *Click* → Bring *thumb tip* (landmark 4) close to *index finger tip*.
- *Exit* → Press ESC.

---

## 💡 Tips
- Ensure good lighting and avoid background clutter.
- Keep hand visible to camera.
- Move slowly at first to get used to tracking sensitivity.
