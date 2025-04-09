import os
import cv2
import numpy as np
import pickle
import tensorflow as tf
from tqdm import tqdm

# === 1. Paths ===
project_folder = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
dataset_path = os.path.join(project_folder, 'dataset')
model_path = os.path.join(project_folder, 'models', 'face_recognition_model_saved')
embeddings_path = os.path.join(project_folder, 'embeddings', 'custom_embeddings.pickle')
haar_path = os.path.join(project_folder, 'models', 'haarcascade_frontalface_default.xml')

# === 2. Load Model & Haar Cascade ===
print("Loading custom face recognition model...")
model = tf.keras.models.load_model(model_path)
infer = model.signatures["serving_default"]

face_cascade = cv2.CascadeClassifier(haar_path)

# === 3. Face Preprocessing ===
def preprocess_face(image):
    face = cv2.resize(image, (224, 224))
    face = face.astype('float32') / 255.0
    face = np.expand_dims(face, axis=0)
    return face

# === 4. Load existing embeddings (if any) ===
existing_users = set()
if os.path.exists(embeddings_path):
    with open(embeddings_path, 'rb') as f:
        old_data = pickle.load(f)
    existing_users = set(old_data['names'])
    all_embeddings = list(old_data['embeddings'])
    all_names = list(old_data['names'])
else:
    all_embeddings = []
    all_names = []
    print("No previous embedding file found. Starting fresh.")

# === 5. Process only new users ===
print("Processing users...")

for user_name in os.listdir(dataset_path):
    if user_name in existing_users:
        print(f"Skipping {user_name} (already embedded)")
        continue

    user_folder = os.path.join(dataset_path, user_name)
    if not os.path.isdir(user_folder):
        continue

    print(f"Processing new user: {user_name}")
    image_files = [f for f in os.listdir(user_folder) if f.endswith(('.jpg', '.png'))]
    user_embeddings = []

    for img_file in tqdm(image_files):
        img_path = os.path.join(user_folder, img_file)
        img = cv2.imread(img_path)
        if img is None:
            continue

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        if len(faces) == 0:
            print(f"  No face detected in: {img_file}")
            continue

        x, y, w, h = faces[0]
        face_img = img[y:y+h, x:x+w]
        face_input = preprocess_face(face_img)

        output_dict = infer(tf.convert_to_tensor(face_input))
        embedding = list(output_dict.values())[0].numpy()[0]

        user_embeddings.append(embedding)

    if user_embeddings:
        avg_embedding = np.mean(user_embeddings, axis=0)
        all_embeddings.append(avg_embedding)
        all_names.append(user_name)
        print(f"  Embedded {len(user_embeddings)} images for {user_name}")
    else:
        print(f"  No valid faces found for {user_name}")

# === 6. Save updated embeddings ===
final_data = {"names": all_names, "embeddings": all_embeddings}

if not os.path.exists(os.path.dirname(embeddings_path)):
    os.makedirs(os.path.dirname(embeddings_path))

with open(embeddings_path, 'wb') as f:
    pickle.dump(final_data, f)

print("✅ Successfully updated embeddings with new users!")
