import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load emotion model only
emotion_model = load_model("model/fer_model.h5")
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

class VideoCamera:
    def __init__(self):
        self.video = cv2.VideoCapture(0)
        self.video.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.video.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    def __del__(self):
        self.video.release()

    def get_frame(self):
        success, frame = self.video.read()
        if not success:
            return None, "No Frame", 0

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)

        if len(faces) == 0:
            _, jpeg = cv2.imencode('.jpg', frame)
            return jpeg.tobytes(), "No Face", 0

        x, y, w, h = faces[0]

        try:
            face_gray = gray[y:y + h, x:x + w]
            face_gray = cv2.resize(face_gray, (48, 48))
            face_gray = face_gray.astype("float32") / 255.0
            face_gray = np.expand_dims(face_gray, axis=(0, -1))

            emotion_pred = emotion_model.predict(face_gray, verbose=0)[0]
            emotion_conf = int(np.max(emotion_pred) * 100)
            emotion_label = emotion_labels[np.argmax(emotion_pred)]

            # Draw bounding box and label
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, f"{emotion_label} ({emotion_conf}%)", (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        except Exception as e:
            print("Emotion prediction error:", e)
            emotion_label, emotion_conf = "Error", 0

        _, jpeg = cv2.imencode('.jpg', frame)
        return jpeg.tobytes(), emotion_label, emotion_conf
