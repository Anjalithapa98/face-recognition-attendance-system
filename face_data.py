import cv2
import os

user_name = input("Enter the name of the person: ").strip()

dataset_path = os.path.join("dataset", user_name)
os.makedirs(dataset_path, exist_ok=True)
print(f"[INFO] Created folder: {dataset_path}")

cam = cv2.VideoCapture(0)
cam.set(3, 640)  
cam.set(4, 480)  

face_detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
print(f"\n[INFO] Initializing face capture for '{user_name}'. Look at the camera and press 'q' to quit early.")

count = 0

while True:
    ret, frame = cam.read()
    if not ret:
        print("[ERROR] Failed to grab frame.")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_detector.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5)

    for (x, y, w, h) in faces:
        count += 1
        face_img = gray[y:y+h, x:x+w]
        file_path = os.path.join(dataset_path, f"{count}.jpg")
        cv2.imwrite(file_path, face_img)

        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        cv2.putText(frame, f"Image {count}", (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

    cv2.imshow('Face Data Collection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    elif count >= 50:
        break

print(f"\n[INFO] Face capture completed for '{user_name}'. Total images: {count}")
cam.release()
cv2.destroyAllWindows()
