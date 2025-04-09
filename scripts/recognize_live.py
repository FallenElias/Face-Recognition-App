import cv2
import face_recognition
import pickle
import os
import numpy as np

# === 1. Load Encodings ===
project_folder = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
encodings_path = os.path.join(project_folder, 'encodings', 'encodings.pickle')

print("Loading face encodings...")
with open(encodings_path, 'rb') as f:
    data = pickle.load(f)

print(f"Known users: {set(data['names'])}")

# === 2. Start Webcam ===
cap = cv2.VideoCapture(0)

# Reduce resolution:
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)


print("Starting real-time face recognition. Press 'q' to exit.")

frame_count = 0
while True:
    ret, frame = cap.read()
    if not ret:
        continue
    
    frame_count += 1
    if frame_count % 7 != 0:
        # Skip this frame
        continue

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Detect faces
    boxes = face_recognition.face_locations(rgb, model='hog')
    encodings = face_recognition.face_encodings(rgb, boxes)

    names = []

    # === 3. Compare each face
    for encoding in encodings:
        matches = face_recognition.compare_faces(data['encodings'], encoding, tolerance=0.5)
        name = "Unknown"

        # Check if any match
        if True in matches:
            matchedIdxs = [i for (i, b) in enumerate(matches) if b]
            counts = {}
            for i in matchedIdxs:
                counts[data['names'][i]] = counts.get(data['names'][i], 0) + 1
            name = max(counts, key=counts.get)

        names.append(name)

    # === 4. Draw rectangles + names
    for ((top, right, bottom, left), name) in zip(boxes, names):
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.putText(frame, name, (left, top - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)

    cv2.imshow('Live Recognition - Press q to exit', frame)

    # Quit condition
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
