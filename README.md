Deepfake Video Detection System

Overview

This project detects whether a video is real or fake (deepfake) using deep learning techniques.
It processes video frames, extracts faces, and predicts authenticity using a trained model.


Features

* Upload video for analysis
* Extract frames from video
* Detect faces using MTCNN
* Predict deepfake probability
* Display final result (REAL / FAKE)


Technologies Used

* Python
* Flask
* OpenCV (cv2)
* TensorFlow / Keras
* MTCNN
* NumPy

How It Works

1. User uploads a video
2. Frames are extracted from the video
3. Faces are detected using MTCNN
4. Faces are resized and processed
5. Model predicts probability of being fake
6. Final result is displayed

Project Structure

```
deepfake-video-detection/
│── app.py
│── frame_maker.py
│── predictor.py
│── requirements.txt
│── templates/
│   └── index.html
│── model/
│   └── deepfake_video_model.h5
```
How to Run

1. Clone the repository:

```
git clone https://github.com/your-username/deepfake-video-detection.git
```

2. Navigate to folder:

```
cd deepfake-video-detection
```

3. Create virtual environment:

```
python -m venv venv
source venv/bin/activate   # Mac/Linux
```

4. Install dependencies:

```
pip install -r requirements.txt
```

5. Run the app:

```
python app.py
```

6. Open browser:

```
http://127.0.0.1:5001
```

Note

* Model file may not be included due to GitHub size limits
* You can add your own trained model in the `model/` folder

 Output

* Displays probability of fake video
* Final verdict: REAL or FAKE

Author

Sunny Gupta

Acknowledgement

This project is inspired by recent research in deepfake detection using deep learning.
