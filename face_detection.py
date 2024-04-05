import face_recognition
import cv2
import numpy as np

known_face_encodings = []
known_face_names = []
frame_resizing = 0.25

def detect_known_faces(frame, threshold=0.7):
    small_frame = cv2.resize(
        frame, (0, 0), fx=frame_resizing, fy=frame_resizing
    )
    rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
    face_locations = face_recognition.face_locations(rgb_small_frame)
    face_encodings = face_recognition.face_encodings(
        rgb_small_frame, face_locations
    )

    face_names = []
    confidence_scores = []  

    for face_encoding in face_encodings:
        matches = face_recognition.compare_faces(
            known_face_encodings, face_encoding, tolerance=threshold
        )
        name = "Unknown"
        confidence = None  

        if True in matches:
            match_indices = [i for i, match in enumerate(matches) if match]
            distances = face_recognition.face_distance(
                known_face_encodings, face_encoding
            )
            best_match_index = np.argmin(distances)

            if distances[best_match_index] <= threshold:
                name = known_face_names[best_match_index]
                confidence = 1 - distances[best_match_index]

        face_names.append(f"{name}: {confidence:.2f}")
        confidence_scores.append(confidence)

    face_locations = np.array(face_locations)
    face_locations = face_locations / frame_resizing
    return face_locations.astype(int), face_names, confidence_scores