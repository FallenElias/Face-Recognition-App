
# 👤 Real-Time Facial Recognition System (Desktop GUI)

A full-featured Python-based real-time facial recognition system with a modern GUI interface. Supports both pre-trained and custom deep learning models for user registration, recognition, and management.

---

## 📦 Features

- 🎯 **Two model modes**:
  - `face_recognition`: Pretrained dlib model with HOG/CNN + 128D encodings
  - `custom_model`: MobileNetV2 + 128D embeddings (fine-tuned)
- 📸 Real-time recognition via webcam
- 👤 Register new users with automatic image capture
- 🔁 Update recognition database without retraining the model
- 🗑️ Delete users with synced image folder and model database cleanup
- ✅ Built-in GUI using Tkinter for non-technical users
- 📋 User list dynamically displayed and managed

---

## 🧰 Tech Stack

| Component         | Description                            |
|------------------|----------------------------------------|
| **Language**      | Python 3.9+                            |
| **GUI**           | Tkinter                                |
| **Model**         | MobileNetV2 (Keras) or face_recognition |
| **Detection**     | Haar Cascade / HOG                     |
| **Embedding**     | 128D Dense layer / dlib encodings      |
| **Frameworks**    | TensorFlow, OpenCV, face_recognition   |

---

## 📂 Project Structure

```
Live Facial Recognition/
├── scripts/                 # Python scripts
│   ├── main_gui.py
│   ├── register_user.py
│   ├── update_model.py / update_model2.py
│   ├── recognize_live.py / recognize_live2.py
│   └── remove_user.py
├── models/                  # Saved .h5 or SavedModel format
├── encodings/               # Pickle files for face_recognition
├── embeddings/              # Pickle files for custom_model
├── dataset/                 # Captured face images by user
├── requirements.txt
├── environment.yml
└── README.md
```

---

## 🚀 Getting Started

### 1. Clone the Repo

```bash
git clone https://github.com/YOUR_USERNAME/face-recognition-gui.git
cd face-recognition-gui
```

### 2. Create and Activate the Environment

```bash
conda create -n face_recog_env python=3.9
conda activate face_recog_env
pip install -r requirements.txt
```

Or use the full Conda snapshot:

```bash
conda env create -f environment.yml
conda activate face_recog_env
```

### 3. Run the GUI Application

```bash
python scripts/main_gui.py
```

---

## 🧠 How It Works

1. **Face detection** via Haar cascade (OpenCV) or HOG (dlib)
2. **Preprocessing**: crop, resize to 224×224, normalize
3. **Embedding extraction**:
   - `face_recognition` model → dlib encoding (128D)
   - `custom_model` → MobileNetV2 embedding (128D)
4. **Comparison**:
   - Euclidean (dlib)
   - Cosine similarity (custom)
5. **Recognition**: matches stored vectors with live face
6. **Display**: bounding box and username on camera feed

---

## 🧬 Model Comparison

| Model             | Training Required | Output Vector | Distance Metric | Supports New Users? |
|------------------|-------------------|----------------|------------------|----------------------|
| face_recognition | ❌ No              | 128D encoding  | Euclidean        | ✅ Yes               |
| custom_model     | ✅ Yes             | 128D embedding | Cosine Similarity| ✅ Yes               |

---

## 🛠️ Key Features via GUI

| Feature           | Description                                                  |
|------------------|--------------------------------------------------------------|
| Register User     | Capture 150 webcam images and store in `dataset/{name}/`     |
| Update Model      | Extract and update embeddings/encodings into `.pickle` file |
| Start Recognition | Live camera, real-time recognition & labeling                |
| Delete User       | Remove all data (images + vector + name) for that user       |
| Switch Model      | Dropdown to toggle between recognition engines               |

---

## 🧪 Training Details (Custom Model)

- Based on **MobileNetV2**
- Final layers added:
  - `GlobalAveragePooling2D`
  - `Dense(128)` → Embedding
  - `Dense(31)` → Softmax classifier (only for training)
- Trained with:
  - `sparse_categorical_crossentropy` (because labels are integers)
  - Optimizer: Adam
  - Fine-tuned from layer 100 onward
- Final validation accuracy: **~71%**

---

## 🧠 Key Concepts

| Term                  | Meaning & Use |
|-----------------------|---------------|
| **Embedding**         | Vector that numerically represents a face (128D) |
| **Encoding**          | Synonym used in `face_recognition` library |
| **Cosine Similarity** | Measures angle between two vectors (better for embeddings) |
| **L2 Normalization**  | Scales embedding to unit length for stable comparison |
| **Haar Cascade**      | Classic OpenCV method for real-time face detection |
| **Triplet Loss**      | Advanced training strategy to pull/push faces in embedding space |
| **ArcFace**           | State-of-the-art embedding method using angular margin |

---

## 🚧 Limitations

| Limitation                          | Workaround / Future Work           |
|-------------------------------------|------------------------------------|
| No liveness/spoof detection         | Add blink/movement detection       |
| No face tracking                    | Integrate OpenCV or dlib tracker   |
| Softmax training lacks margin       | Use Triplet Loss or ArcFace        |

---

## 🔮 Extensions & Future Ideas

- ✅ Web-based dashboard (Flask/Django)
- ✅ Triplet Loss / ArcFace implementation
- ✅ Export to TensorFlow Lite for mobile
- ✅ Cloud-based embedding database (e.g., Firebase)
- ✅ Multi-camera input support
- ✅ Alerts or notifications for unknown faces

---

## 📋 Requirements

Generated via `pip freeze > requirements.txt`. Includes:
- opencv-python
- face_recognition
- tensorflow
- keras
- tqdm
- numpy
- pillow

---

## 📃 License

MIT License  
Feel free to use and modify for personal or academic projects. Contributions welcome!


