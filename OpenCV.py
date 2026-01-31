import face_recognition
import cv2
import pyttsx3
import time

# Load your image and encode it
known_image = face_recognition.load_image_file("your_face.jpg")
known_encoding = face_recognition.face_encodings(known_image)[0]

engine = pyttsx3.init()
has_said_hi = False

# Start webcam
video_capture = cv2.VideoCapture(0)

while True:
    ret, frame = video_capture.read()
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)  # Speed up
    rgb_small_frame = small_frame[:, :, ::-1]  # BGR to RGB

    face_locations = face_recognition.face_locations(rgb_small_frame)
    face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

    for face_encoding in face_encodings:
        matches = face_recognition.compare_faces([known_encoding], face_encoding)

        if True in matches:
            if not has_said_hi:
                print("Hi Sir 👋")
                engine.say("Hi Sir")
                engine.runAndWait()
                has_said_hi = True
        else:
            has_said_hi = False

    cv2.imshow('Jarvis Cam', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()
