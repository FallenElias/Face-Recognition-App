import cv2
import face_recognition
import os
import pickle

# === 1. Absolute Paths ===
project_folder = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
dataset_dir = os.path.join(project_folder, 'dataset')
encodings_dir = os.path.join(project_folder, 'encodings')
encodings_path = os.path.join(encodings_dir, 'encodings.pickle')
user_list_path = os.path.join(encodings_dir, 'user_list.txt')

# === 2. Load Existing Encodings ===
if os.path.exists(encodings_path):
    print("Loading existing encodings...")
    with open(encodings_path, 'rb') as f:
        data = pickle.load(f)
    known_encodings = data['encodings']
    known_names = data['names']
else:
    print("No existing encodings found, creating new...")
    known_encodings = []
    known_names = []

# === 3. Load Known Users List ===
if os.path.exists(user_list_path):
    with open(user_list_path, 'r') as f:
        known_usernames = [line.strip() for line in f]
else:
    known_usernames = []

print(f"Known users so far: {known_usernames}")

# === 4. Loop Over Dataset Folders ===
print("Checking dataset folders...")

for user_name in os.listdir(dataset_dir):
    user_folder = os.path.join(dataset_dir, user_name)
    if not os.path.isdir(user_folder):
        continue  # Skip files, only process folders

    if user_name in known_usernames:
        print(f"Skipping already processed user: {user_name}")
        continue  # Already processed

    print(f"Processing NEW user: {user_name}")

    image_count = 0
    for image_name in os.listdir(user_folder):
        if not image_name.lower().endswith(('.png', '.jpg', '.jpeg')):
            continue

        img_path = os.path.join(user_folder, image_name)
        image = cv2.imread(img_path)

        if image is None:
            print(f"  Skipped invalid image: {img_path}")
            continue

        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        boxes = face_recognition.face_locations(rgb, model='hog')

        if len(boxes) == 0:
            print(f"  No face detected in: {img_path}")
            continue

        encodings = face_recognition.face_encodings(rgb, boxes)

        for encoding in encodings:
            known_encodings.append(encoding)
            known_names.append(user_name)
            image_count += 1

    print(f"  Processed {image_count} valid images for '{user_name}'.")

    # === 5. Add User to Known User List ===
    known_usernames.append(user_name)

# === 6. Save Updated Encodings ===
print("Saving updated encodings...")
os.makedirs(encodings_dir, exist_ok=True)

with open(encodings_path, 'wb') as f:
    pickle.dump({'encodings': known_encodings, 'names': known_names}, f)

# === 7. Save Updated Known User List ===
with open(user_list_path, 'w') as f:
    for name in known_usernames:
        f.write(name + '\n')

print("Successfully updated encodings with new users only!")
print(f"Current known users: {known_usernames}")
