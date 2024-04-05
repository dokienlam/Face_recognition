import cv2
import os
import face_recognition
import glob

known_face_encodings = []
known_face_names = []
def load_encoding_images(images_path):
    images_path = glob.glob(
        os.path.join(
            r'E:\Code\Code_AI\Recognition_face_data',
            "*.*"
        )
    )
    print("{} encoding images found.".format(len(images_path)))

    for img_path in images_path:
        img = cv2.imread(img_path)
        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        basename = os.path.basename(img_path)
        (filename, ext) = os.path.splitext(basename)
        img_encoding = face_recognition.face_encodings(rgb_img)[0]
        known_face_encodings.append(img_encoding)
        known_face_names.append(filename)
        # known_face_names.append(basename)
    print("Encoding images loaded")