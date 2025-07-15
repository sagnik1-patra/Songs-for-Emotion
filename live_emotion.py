import cv2
from tensorflow.keras.models import load_model
import numpy as np
import matplotlib.pyplot as plt

# Check OpenCV for required attributes
if not hasattr(cv2, 'CascadeClassifier'):
    raise ImportError(
        "❌ Your OpenCV installation does not support CascadeClassifier.\n"
        "➡️ Fix it with: pip uninstall opencv-python opencv-python-headless -y && pip install opencv-contrib-python"
    )

# ✅ Load the trained model
model_path = r"C:\Users\sagni\Downloads\Emotion based Music\emotion_cnn_model.h5"
emotion_model = load_model(model_path)
print("✅ Model loaded successfully from", model_path)

# ✅ Define emotion labels
emotion_classes = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']

# ✅ Load Haar cascade for face detection
haar_cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
face_cascade = cv2.CascadeClassifier(haar_cascade_path)
if face_cascade.empty():
    raise FileNotFoundError(f"❌ Haar cascade file not found at {haar_cascade_path}")
print("✅ Haar Cascade loaded successfully")

# ✅ Start webcam
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # Use CAP_DSHOW for Windows
if not cap.isOpened():
    raise RuntimeError("❌ Could not open webcam")
print("✅ Webcam started successfully")

plt.ion()  # Turn on interactive mode for matplotlib
fig, ax = plt.subplots(figsize=(8, 6))

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("❌ Failed to grab frame")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

        for (x, y, w, h) in faces:
            # Draw rectangle
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

            # Preprocess face
            face = gray[y:y+h, x:x+w]
            face = cv2.resize(face, (48, 48))
            face = face / 255.0
            face = face.reshape(1, 48, 48, 1)

            # Predict emotion
            predictions = emotion_model.predict(face, verbose=0)
            emotion_idx = np.argmax(predictions)
            confidence = np.max(predictions)
            label = f"{emotion_classes[emotion_idx]} ({confidence*100:.1f}%)"

            # Put label text on frame
            cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX,
                        0.9, (0, 255, 0), 2)

        # Convert BGR to RGB for matplotlib
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        ax.imshow(rgb_frame)
        ax.axis('off')
        plt.draw()
        plt.pause(0.001)
        ax.clear()

except KeyboardInterrupt:
    print("🛑 Interrupted by user (Ctrl+C)")

finally:
    cap.release()
    plt.ioff()
    plt.close()
    print("✅ Webcam released successfully")
