
# 🧠 Emotion Detection Web App

A Flask-based web application that detects facial emotions in real-time using a webcam feed. Includes additional informational pages such as **About** and **Team**.

---

## 📂 Project Structure

```
emotion-detector/
├── app.py
├── camera.py
├── templates/
│   ├── index.html
│   ├── about.html
│   └── team.html
├── static/
│   └── images/
│       ├── srun_borath.jpg
│       ├── sov_yongkhang.jpg
│       ├── thun_sreypich.jpg
│       ├── toeum_sophorn.jpg
│       └── sok_socheata.jpg
├── README.md
└── requirements.txt
```

---

## 🚀 Features

- 🎥 Real-time webcam video feed with emotion prediction
- 👤 About and Team pages with Bootstrap styling
- 📱 Responsive design using Bootstrap 5
- 🧠 Deep learning backend (in `camera.py`) for emotion classification

---

## 🛠️ Requirements

- Python 3.7+
- Flask
- OpenCV (`cv2`)
- TensorFlow/Keras (optional, depending on model)
- NumPy

You can install all dependencies using:

```bash
pip install -r requirements.txt
```

**Example `requirements.txt`:**
```txt
Flask
opencv-python
numpy
```

Add additional dependencies like `tensorflow`, `keras`, etc., if used in `camera.py`.

---

## 🧪 How to Run the App

1. **Clone the project:**

```bash
git clone https://github.com/yourusername/emotion-detector.git
cd emotion-detector
```

2. **Install dependencies:**

```bash
pip install -r requirements.txt
```

3. **Start the server:**

```bash
python app.py
```

4. **Open the browser and go to:**

```
http://127.0.0.1:5000/
```

---

## 🌐 Available Pages

| Route         | Description                  |
|---------------|------------------------------|
| `/`           | Live webcam feed             |
| `/about`      | About the application        |
| `/team`       | Team member profile cards    |
| `/video_feed` | MJPEG stream for live video  |

---

## 👥 Team Members

- Srun Borath
- Sov YongKhang
- Thun Sreypich
- Toeum Sophorn
- Sok Socheata

**Bachelor of ICT – Year 3 Semester 2 (2025)**

---

## 📸 Image Placement

Place all team member images inside:

```
static/images/
```

Example:
```
static/images/srun_borath.jpg
```

---

## 📌 Notes

- Make sure your webcam is connected and accessible.
- This app is for local development only. For deployment, consider using production servers like **Gunicorn** and **nginx**.
- Customize the emotion model logic in `camera.py`.

---

## 📃 License

MIT License — free to use and modify.
