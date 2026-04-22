import os
import cv2
import numpy as np
import tensorflow as tf
print(tf.__version__)

model = tf.keras.models.load_model("./model/deepfake_video_model.h5")




IMG_SIZE = 224
MAX_SEQ_LENGTH = 20
NUM_FEATURES = 2048


def build_feature_extractor():
    feature_extractor = tf.keras.applications.InceptionV3(
        weights="imagenet",
        include_top=False,
        pooling="avg",
        input_shape=(IMG_SIZE,IMG_SIZE,3),
    )
    preprocess_input= tf.keras.applications.inception_v3.preprocess_input

    inputs = tf.keras.Input((IMG_SIZE, IMG_SIZE, 3))
    preprocessed = preprocess_input(inputs)

    outputs = feature_extractor(preprocessed)
    return tf.keras.Model(inputs, outputs, name="feature_extractor")

feature_extractor = build_feature_extractor()

def prepare_single_video(frames):
    frames = frames[None, ...]
    frame_mask = np.zeros(shape=(1, MAX_SEQ_LENGTH,), dtype="bool")
    frame_features = np.zeros(shape=(1, MAX_SEQ_LENGTH, NUM_FEATURES), dtype="float32")

    for i, batch in enumerate(frames):
        video_length = batch.shape[0]
        length = min(MAX_SEQ_LENGTH, video_length)
        for j in range(length):
            frame_features[i, j , :] = feature_extractor.predict(batch[None, j, :])
        frame_mask[i, :length] = 1

    return frame_features, frame_mask

def sequence_prediction(frames):
    # processed_images_folder = 'processed_images'
    
    # Ensure the folder contains cropped images
    # if not os.path.exists(processed_images_folder) or not os.listdir(processed_images_folder):
    #     raise ValueError("No cropped images found in the folder.")

    # predictions = []
    # for image_file in os.listdir(processed_images_folder):
    #     image_path = os.path.join(processed_images_folder, image_file)
    #     img = preprocess_image(image_path)
    #     pred = model.predict(img)
    #     predictions.append(pred)
    #     os.remove(image_path)

    # # Step 4: Average the predictions
    # average_prediction = np.mean(predictions)

    # # Step 5: Classify as fake or real
    # threshold = 0.5  # Assuming a binary classification with sigmoid output
    # classification = 'FAKE' if average_prediction > threshold else 'REAL'

    # return classification, average_prediction
    frame_features, frame_mask = prepare_single_video(frames)
    return model.predict([frame_features, frame_mask])[0]