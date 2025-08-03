import cv2
import os
import numpy as np

# === CONFIGURATION ===
haar_file = 'haarcascade_frontalface_default.xml'
datasets_dir = 'datasets'
person_name = 'Kris'
sample_count = 25
image_size = (130, 100)
camera_index = 0  # Try 1 if 0 doesn't work

# === CREATE DIRECTORY IF NEEDED ===
person_path = os.path.join(datasets_dir, person_name)
os.makedirs(person_path, exist_ok=True)

# === INIT HAAR CASCADE AND CAMERA ===
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + haar_file)
webcam = cv2.VideoCapture(camera_index)

# === STEP 1: DATA COLLECTION ===
print(f"[INFO] Collecting data for {person_name}...")

count = 1
while count <= sample_count:
    success, frame = webcam.read()
    if not success:
        continue

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 4)

    for (x, y, w, h) in faces:
        face = gray[y:y + h, x:x + w]
        face_resize = cv2.resize(face, image_size)
        file_path = f"{person_path}/{count}.png"
        cv2.imwrite(file_path, face_resize)
        count += 1

        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

    cv2.imshow("Data Collection", frame)
    if cv2.waitKey(10) == 27:  # ESC key
        break

print("[INFO] Data collection complete.")
cv2.destroyAllWindows()

# === STEP 2: TRAINING ===
print("[INFO] Training model...")

(images, labels, names, id_counter) = ([], [], {}, 0)

for subdir in os.listdir(datasets_dir):
    names[id_counter] = subdir
    subdir_path = os.path.join(datasets_dir, subdir)

    for filename in os.listdir(subdir_path):
        if filename.endswith(".png"):
            filepath = os.path.join(subdir_path, filename)
            images.append(cv2.imread(filepath, 0))
            labels.append(id_counter)

    id_counter += 1

images = np.array(images)
labels = np.array(labels)

# Load recognizer (install opencv-contrib-python if this fails)
model = cv2.face.FisherFaceRecognizer_create()
model.train(images, labels)

print("[INFO] Training complete.")

# === STEP 3: REAL-TIME RECOGNITION ===
print("[INFO] Starting real-time recognition...")

cnt = 0
while True:
    success, frame = webcam.read()
    if not success:
        continue

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        face = gray[y:y + h, x:x + w]
        face_resize = cv2.resize(face, image_size)

        prediction = model.predict(face_resize)
        label, confidence = prediction[0], prediction[1]

        if confidence < 800:
            name_text = f"{names[label]} - {confidence:.0f}"
            cv2.putText(frame, name_text, (x - 10, y - 10), 
                        cv2.FONT_HERSHEY_COMPLEX, 1, (51, 255, 255), 2)
            cnt = 0
        else:
            cnt += 1
            cv2.putText(frame, "Unknown", (x - 10, y - 10), 
                        cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 2)

            if cnt > 100:
                print("[ALERT] Unknown person detected!")
                cv2.imwrite("unknown.jpg", frame)
                cnt = 0

        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)

    cv2.imshow("Face Recognition", frame)
    if cv2.waitKey(10) == 27:  # ESC key
        break

webcam.release()
cv2.destroyAllWindows()
