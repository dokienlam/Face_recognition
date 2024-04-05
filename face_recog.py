import cv2
from load_images import load_encoding_images
from face_detection import detect_known_faces
def face_recog():
    
    load_encoding_images(r'E:\Code\Code_AI\Recognition_face_data')

    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()

        face_locations, face_names, confidence_scores = detect_known_faces(frame)
        for face_loc, name in zip(face_locations, face_names):
            y1, x2, y2, x1 = face_loc[0], face_loc[1], face_loc[2], face_loc[3]

            cv2.putText(
                frame, name, (x1, y1 - 10), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 200), 2
            )
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 200), 4)

        cv2.imshow("Frame", frame)
        if cv2.waitKey(1) == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()
