import cv2
import os

user_name = os.environ.get("NEW_USER_NAME")
if not user_name:
    user_name = input("Enter the new user's name (no spaces!):")
    
# Absolute path fix:
project_folder = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
save_dir = os.path.join(project_folder, 'dataset', user_name)
os.makedirs(save_dir, exist_ok=True)

cap = cv2.VideoCapture(0)
count = 0

print("Starting face capture. Press 'q' to quit early.")

haar_cascade_path = './haarcascade_frontalface_default.xml'
if not os.path.exists(haar_cascade_path):
    print("Downloading Haar cascade file...")
    import urllib.request
    urllib.request.urlretrieve(
        'https://github.com/opencv/opencv/raw/master/data/haarcascades/haarcascade_frontalface_default.xml',
        haar_cascade_path
    )
    print("Downloaded!")

face_cascade = cv2.CascadeClassifier(haar_cascade_path)


while True:
    ret, frame = cap.read()
    if not ret:
        continue

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        count += 1
        face = frame[y:y+h, x:x+w]
        face = cv2.resize(face, (160, 160))
        cv2.imwrite(f"{save_dir}/{user_name}_{count}.jpg", face)
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(frame, f'Image {count}', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    cv2.imshow('Capturing Faces - Press q to stop', frame)

    if count >= 150:
        print("Captured 150 images!")
        break
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("Early exit.")
        break

cap.release()
cv2.destroyAllWindows()
print(f"Images saved in: {save_dir}")
