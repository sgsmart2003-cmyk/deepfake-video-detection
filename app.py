from flask import Flask, request, jsonify, render_template
import os
import frame_maker
import predictor
import cv2
import base64
import numpy as np

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')


UPLOAD_FOLDER = os.path.join(os.getcwd(), 'uploads')
CROPPED_IMAGES_FOLDER = os.path.join(os.getcwd(), 'processed_images')  # Path to cropped images folder
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
if not os.path.exists(CROPPED_IMAGES_FOLDER):
    os.makedirs(CROPPED_IMAGES_FOLDER)

@app.route('/predict', methods=['POST'])
def predict():
    
    if 'video' not in request.files:
        return jsonify({'error': 'No Video file Provided'}), 400

    video_file = request.files['video']
    if video_file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    
    video_path = os.path.join(app.config['UPLOAD_FOLDER'], video_file.filename)
    video_file.save(video_path)
    
    
    try:
        frames = frame_maker.process_video(video_path)
    except Exception as e:
        return jsonify({"error": f"Error processing video: {str(e)}"}), 500

    
    image_buffers = []
    try:
        for filename in os.listdir(CROPPED_IMAGES_FOLDER):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):  
                img = cv2.imread(img_path)
                
                
                success, buffer = cv2.imencode('.jpg', img)
                if success:
                    
                    img_base64 = base64.b64encode(buffer).decode('utf-8')
                    image_buffers.append(img_base64)
    except Exception as e:
        return jsonify({"error": f"Error encoding images: {str(e)}"}), 500
    
    
    try:
        
        prediction = predictor.sequence_prediction(frames)
        print(f"probablity of being FAKE : {prediction}")
    
        classification = round(np.mean(prediction))
        print(f"The predicted class of the video is {prediction}")
    except Exception as e:
        return jsonify({"error": f"Error during prediction: {str(e)}"}), 500
    
    
    print(classification)
    response_data = {
        "classification": classification,
        "average_prediction": float(prediction), 
        "images": image_buffers  
    }

    return jsonify(response_data), 200

if __name__ == '__main__':
    app.run(port=5001)

