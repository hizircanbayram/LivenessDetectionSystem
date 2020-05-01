from keras.preprocessing.image import img_to_array
import imutils
import cv2
from keras.models import load_model
import numpy as np



# parameters for loading data and images
detection_model_path = 'emotion_detector/haarcascade_files/haarcascade_frontalface_default.xml'
emotion_model_path = 'emotion_detector/models/_mini_XCEPTION.102-0.66.hdf5'

# hyper-parameters for bounding boxes shape
# loading models
face_detection = cv2.CascadeClassifier(detection_model_path)
emotion_classifier = load_model(emotion_model_path, compile=False)
EMOTIONS = ["angry", "disgust", "scared", "happy", "sad", "surprised", "neutral"]

def predict_emotion(img):
    roi = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    roi = cv2.resize(roi, (64, 64))
    roi = roi.astype("float") / 255.0
    roi = img_to_array(roi)
    roi = np.expand_dims(roi, axis=0)
    preds = emotion_classifier.predict(roi)[0]
    label = EMOTIONS[preds.argmax()]
    return label

