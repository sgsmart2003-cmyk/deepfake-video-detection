import cv2
import os
import numpy as np
from mtcnn import MTCNN

MAX_SEQ_LENGTH = 20
IMG_SIZE = 224

def process_video(video_path, number_of_frames=MAX_SEQ_LENGTH):
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_indices = np.linspace(0, total_frames - 1, number_of_frames).astype(int)

    output_folder = 'processed_images'
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    detector = MTCNN()

    frames = []
    i = 0
    current_frame = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if current_frame == frame_indices[i]:
            frames.append(frame)
            i += 1
            if i >= number_of_frames:
                break
        current_frame += 1

    cap.release()

    cropped_faces = []
    for idx, frame in enumerate(frames):
        if frame is None:
            cropped_faces.append(np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8))
            continue

        result = detector.detect_faces(frame)
        if result:
            x, y, w, h = result[0]['box']
            x, y = max(0, x), max(0, y)
            cropped_face = frame[y:y+h, x:x+w]
            cropped_faces.append(cropped_face)
        else:
            print(f"No face detected in frame {idx}")
            cropped_faces.append(np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8))

    resized_faces = []
    for idx, face in enumerate(cropped_faces):
        if face is not None and face.size > 0:
            resized_face = cv2.resize(face, (IMG_SIZE, IMG_SIZE))
            resized_faces.append(resized_face)

            save_path = os.path.join(output_folder, f'processed_face_{idx}.png')
            cv2.imwrite(save_path, resized_face)
        else:
            print(f"Skipping invalid face at index {idx}")

    print("processed face images saved successfully.")
    return np.array(resized_faces)