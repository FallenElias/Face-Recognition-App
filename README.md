
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

## 🧩 Business Applications & Use Cases

This project is more than a technical prototype — it represents a scalable solution for multiple industries seeking to leverage facial recognition for automation, security, and personalization.

---

### 🏪 Application 1: Autonomous Stores

In an autonomous retail environment, facial recognition allows:

- **Frictionless entry and payment**: registered users can enter stores and be charged automatically
- **Personalized experiences**: greet users by name, offer tailored promotions
- **Loss prevention**: restrict or track entry for banned individuals
- **Multi-camera tracking**: track customer movement and behavior anonymously

**Relevant industries:** Retail, convenience stores, airport shops, Amazon Go-style locations

---

### 🏨 Application 2: Keyless Access in Hospitality

Facial recognition can replace hotel keycards and improve guest experience:

- **Contactless room access**: guests are recognized at the door without cards
- **Improved check-in process**: verified via webcam or kiosk
- **VIP detection and service triggers**
- **Integration with smart locks and IoT systems**

**Relevant industries:** Hotels, co-living spaces, vacation rentals, smart homes

---

## 🚧 Business Challenges & Future Extensions

| Area                      | Limitation                        | Proposed Solution                                  |
|---------------------------|-----------------------------------|----------------------------------------------------|
| Recognition Accuracy      | Basic softmax loss                | Use **Triplet Loss** or **ArcFace**, larger datasets |
| Performance               | Model may lag on low-spec systems | Use **TensorFlow Lite**, **batch inference**, or **Spark** |
| Real-time Scalability     | Limited to one cam/feed           | Add **multi-camera** and cloud embedding support    |
| Physical Integration      | Software-only                     | Add **Arduino**, **Raspberry Pi**, or **smart lock** APIs |
| Security & Liveness       | Faces can be spoofed              | Add **blink detection**, **challenge-response**     |

---

### 🌍 Market Insights (from presentation)

- 58% of hotels plan to deploy facial recognition within 3 years (HSMAI Study)
- 73% of users prefer keyless access (Deloitte Digital Hotels)
- Smart building solutions grow at 25% CAGR (Oracle Hospitality)

---

## 📈 Strategic Value

This facial recognition system serves as a launchpad for:
- **Smart retail & hotel automation**
- **Enterprise employee access systems**
- **Event & venue identity management**
- **Secure authentication and surveillance**

By leveraging real-time recognition with a flexible architecture and GUI, this platform is highly adaptable to commercial environments where identity verification is essential.

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


