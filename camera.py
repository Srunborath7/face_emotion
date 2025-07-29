import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load emotion model
emotion_model = load_model("model/emotion_model.h5")  
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

class VideoCamera:
    def __init__(self):
        self.video = cv2.VideoCapture(0)
        self.video.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.video.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        # Load Haar Cascade for face detection
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )

    def __del__(self):
        self.video.release()

    def get_frame(self):
        success, frame = self.video.read()
        if not success:
            return None, "No Frame", 0

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

        emotion_label = "No Face"
        emotion_conf = 0

        for (x, y, w, h) in faces:
            face = gray[y:y + h, x:x + w]

            try:
                # Resize to model input size
                face = cv2.resize(face, (48, 48))
                # Convert grayscale face to RGB (3 channels)
                face = cv2.cvtColor(face, cv2.COLOR_GRAY2RGB)
                # Normalize pixel values
                face = face.astype("float32") / 255.0
                # Add batch dimension
                face = np.expand_dims(face, axis=0)  # shape: (1, 48, 48, 3)

                # Predict emotion
                pred = emotion_model.predict(face, verbose=0)[0]
                conf = int(np.max(pred) * 100)
                label = emotion_labels[np.argmax(pred)]

                # Update label if confidence higher than current
                if conf > emotion_conf:
                    emotion_conf = conf
                    emotion_label = label

                # Draw bounding box and label on frame
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(frame, f"{label} ({conf}%)", (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

            except Exception as e:
                print("Prediction Error:", e)
                continue

        _, jpeg = cv2.imencode('.jpg', frame)
        return jpeg.tobytes(), emotion_label, emotion_conf