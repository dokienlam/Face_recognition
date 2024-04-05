import cv2
import os
import tkinter as tk

def Input_data(name_entry):
    
    a = name_entry.get()  
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    cap = cv2.VideoCapture(0)
    save_folder = r'E:\Code\Code_AI\Recognition_face_data'
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    while True:
        ret, frame = cap.read()

        faces = face_cascade.detectMultiScale(frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        if not ret:
            break

        for (x, y, w, h) in faces:
    
            face_roi = frame[y:y+h, x:x+w]

            image_name = f"{a}.jpg"
            save_path = os.path.join(save_folder, image_name)
            cv2.imwrite(save_path, frame)
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            
        cv2.imshow('Face_recognition', frame)

        if cv2.waitKey(1) == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()