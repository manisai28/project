import cv2
import numpy as np
from tensorflow.keras.models import load_model
from load_images import load_images_from_db

# Load trained model and define person IDs
model = load_model("face_auth_model.h5")
person_ids = ['cr', 'kotra', 'shasha','manisai','manisai1','manisai2','manisai3']

def authenticate_person(captured_image):
    captured_image = cv2.resize(captured_image, (128, 128))
    captured_image = np.array(captured_image) / 255.0
    captured_image = np.expand_dims(captured_image, axis=0)

    prediction = model.predict(captured_image)
    person_id_index = np.argmax(prediction)
    confidence = prediction[0][person_id_index]

    # Debugging output
    print("Prediction:", prediction)
    print("Predicted Class Index:", person_id_index)
    print("Confidence:", confidence)

    # Check if the index is within range and adjust confidence threshold
    if confidence > 0.9:
        if person_id_index < len(person_ids):
            return person_ids[person_id_index]
        else:
            return "Invalid prediction index"
    else:
        return "Unknown"


# Open the camera (adjust the index if needed, e.g., use 1 or 2)
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

print("Press 'c' to capture an image.")

captured_face = None
while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame.")
        break

    cv2.imshow("Capture Image", frame)

    # Capture image on pressing 'c'
    if cv2.waitKey(1) & 0xFF == ord('c'):
        captured_face = frame
        break

cap.release()
cv2.destroyAllWindows()

if captured_face is not None:
    # Authenticate the captured image
    authenticated_person = authenticate_person(captured_face)
    print("Authenticated person:", authenticated_person)
else:
    print("No image was captured.")
