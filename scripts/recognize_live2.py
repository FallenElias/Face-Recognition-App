import cv2
import os
import numpy as np
import pickle
import tensorflow as tf
from scipy.spatial.distance import cosine

# === 1. Paths ===
project_folder = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
model_path = os.path.join(project_folder, 'models', 'face_recognition_model_saved')
embeddings_path = os.path.join(project_folder, 'embeddings', 'custom_embeddings.pickle')

# === 2. Load SavedModel with signature ===
print("Loading custom face recognition model...")
full_model = tf.keras.models.load_model(model_path)
infer = full_model.signatures["serving_default"]

# === 3. Load Stored Embeddings ===
if not os.path.exists(os.path.dirname(embeddings_path)):
    os.makedirs(os.path.dirname(embeddings_path))

if os.path.exists(embeddings_path):
    with open(embeddings_path, 'rb') as f:
        data = pickle.load(f)
    known_names = data['names']
    known_embeddings = data['embeddings']
else:
    known_names = []
    known_embeddings = []
    print("No existing embeddings found. Empty database.")

print(f"Known users: {set(known_names)}")

# === 4. Preprocessing Function ===
def preprocess_face(image):
    face = cv2.resize(image, (224, 224))  # MobileNetV2 input size
    face = face.astype('float32') / 255.0
    face = np.expand_dims(face, axis=0)
    return face

# === 5. Webcam Setup ===
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

print("Starting real-time recognition. Press 'q' to exit.")
frame_count = 0

# === 6. Load Haar Cascade (must exist in /models/) ===
haar_path = os.path.join(project_folder, "models", "haarcascade_frontalface_default.xml")
face_cascade = cv2.CascadeClassifier(haar_path)

while True:
    ret, frame = cap.read()
    if not ret:
        continue

    frame_count += 1
    if frame_count % 7 != 0:
        continue  # Frame skipping for performance

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        face_img = frame[y:y+h, x:x+w]
        face_preprocessed = preprocess_face(face_img)

        # Use serving_default signature to get embedding
        output_dict = infer(tf.convert_to_tensor(face_preprocessed))
        embedding = list(output_dict.values())[0].numpy()[0]

        # Compare to known embeddings
        min_dist = float('inf')
        name = "Unknown"

        for idx, known_emb in enumerate(known_embeddings):
            dist = cosine(embedding, known_emb)
            if dist < min_dist:
                min_dist = dist
                name = known_names[idx]

        if min_dist > 0.5:
            name = "Unknown"

        # Draw results
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(frame, f"{name}", (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    cv2.imshow('Real-Time Recognition - Press q to exit', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
