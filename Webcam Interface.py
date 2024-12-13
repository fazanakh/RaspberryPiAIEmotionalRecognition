import cv2
import numpy as np
from imutils.video import VideoStream
import time
import tensorflow as tf

# Load the TFLite model and allocate tensors
interpreter = tf.lite.Interpreter(model_path='model.tflite')
interpreter.allocate_tensors()

# Get input and output tensors
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

print("Model Input Details:", input_details)
print("Model Output Details:", output_details)

# Emotion labels
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
emotion_colors = {
    'Angry': (0, 0, 255),
    'Disgust': (0, 255, 0),
    'Fear': (255, 0, 0),
    'Happy': (0, 255, 255),
    'Sad': (255, 0, 255),
    'Surprise': (255, 255, 0),
    'Neutral': (255, 255, 255)
}

# Initialize the webcam
current_cam = 0
vs = VideoStream(src=current_cam).start()
time.sleep(2.0)

last_emotion = None
last_confidence = 0
emotion_history = []

# Load DNN-based face detector
face_net = cv2.dnn.readNetFromCaffe(
    "deploy.prototxt.txt", 
    "res10_300x300_ssd_iter_140000.caffemodel"
)

def preprocess_face(face):
    face = cv2.resize(face, (48, 48))  # Resize to 48x48
    face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)  # Convert to RGB
    face = face / 255.0  # Normalize
    face = np.reshape(face, (1, 48, 48, 3)).astype(np.float32)  # Reshape and ensure float32
    return face

def switch_camera(vs, current_cam):
    vs.stop()
    current_cam = (current_cam + 1) % 2  # Assuming you have 2 cameras, 0 and 1
    vs = VideoStream(src=current_cam).start()
    time.sleep(2.0)
    return vs, current_cam

def detect_faces_dnn(frame):
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0,
                                 (300, 300), (104.0, 177.0, 123.0))
    face_net.setInput(blob)
    detections = face_net.forward()

    faces = []
    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.5:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            faces.append((startX, startY, endX - startX, endY - startY))
    return faces

while True:
    frame = vs.read()
    frame = cv2.resize(frame, (640, 480))
    
    faces = detect_faces_dnn(frame)

    if len(faces) == 0:
        print("No faces detected")
    else:
        for (x, y, w, h) in faces:
            face = frame[y:y+h, x:x+w]
            face = preprocess_face(face)

            print("Preprocessed face shape:", face.shape)
            print("Preprocessed face values (first 5):", face[0, :5, :5, 0])

            interpreter.set_tensor(input_details[0]['index'], face)
            interpreter.invoke()
            predictions = interpreter.get_tensor(output_details[0]['index'])

            print("Raw model output:", predictions)

            max_index = np.argmax(predictions[0])
            confidence = predictions[0][max_index]

            if confidence > 0.5:  # Change the emotion only if confidence is above 50%
                current_emotion = emotion_labels[max_index]
                print(f"Detected emotion: {current_emotion}, Confidence: {confidence:.2f}")
                print("All emotions and their confidences:")
                for idx, emotion in enumerate(emotion_labels):
                    print(f"{emotion}: {predictions[0][idx] * 100:.2f}%")

                # Smooth the emotion detection by averaging
                emotion_history.append(current_emotion)
                if len(emotion_history) > 10:
                    emotion_history.pop(0)
                most_common_emotion = max(set(emotion_history), key=emotion_history.count)

                if last_emotion is None or (most_common_emotion != last_emotion and abs(confidence - last_confidence) > 0.3):
                    print(f"Updating emotion from {last_emotion} to {most_common_emotion}")
                    last_emotion = most_common_emotion
                    last_confidence = confidence

                color = emotion_colors[last_emotion]
                label = f"{last_emotion} ({confidence*100:.2f}%)"
            
                cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
                cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
    
    cv2.imshow("Emotion Detector", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break
    elif key == ord("s"):  # Press 's' to switch cameras
        vs, current_cam = switch_camera(vs, current_cam)

cv2.destroyAllWindows()
vs.stop()
