import os
import pickle
import shutil

# === Setup ===
project_folder = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
dataset_path = os.path.join(project_folder, 'dataset')

# === Ask for mode ===
print("Which system are you using?")
print("1 = face_recognition (encodings.pickle)")
print("2 = custom_model (custom_embeddings.pickle)")
mode = input("Enter 1 or 2: ").strip()

if mode == "1":
    pickle_path = os.path.join(project_folder, 'encodings', 'encodings.pickle')
elif mode == "2":
    pickle_path = os.path.join(project_folder, 'embeddings', 'custom_embeddings.pickle')
else:
    print("❌ Invalid choice. Exiting.")
    exit()

# === Ask for username to delete ===
username = input("Enter the username you want to delete: ").strip()

# === Delete from .pickle ===
if not os.path.exists(pickle_path):
    print(f"❌ Pickle file not found: {pickle_path}")
else:
    with open(pickle_path, 'rb') as f:
        data = pickle.load(f)

    if username not in data['names']:
        print(f"⚠️ User '{username}' not found in {os.path.basename(pickle_path)}.")
    else:
        # Filter user out
        new_names = []
        new_embeddings = []
        for name, emb in zip(data['names'], data['embeddings']):
            if name != username:
                new_names.append(name)
                new_embeddings.append(emb)

        # Save updated pickle
        with open(pickle_path, 'wb') as f:
            pickle.dump({"names": new_names, "embeddings": new_embeddings}, f)

        print(f"✅ User '{username}' removed from {os.path.basename(pickle_path)}.")

# === Delete dataset folder ===
user_folder = os.path.join(dataset_path, username)
if os.path.exists(user_folder):
    shutil.rmtree(user_folder)
    print(f"🗑️ Deleted dataset folder: {user_folder}")
else:
    print(f"📂 No folder found for user at: {user_folder}")
