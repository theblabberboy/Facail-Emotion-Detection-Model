# Facial Emotion Detection

This project implements a **Facial Emotion Detection** model using **Convolutional Neural Networks (CNN)**. The model classifies human emotions from facial expressions, utilizing **Keras** with a **TensorFlow** backend. The project also includes real-time emotion detection using **OpenCV** to capture and classify emotions from video feeds. The goal of the project is to explore deep learning's potential in recognizing human emotions, with applications ranging from driver safety to educational environments.

## Project Overview

Facial emotion detection is an important area of computer vision with multiple real-world applications, such as:
- **Driver Drowsiness Detection** to improve road safety.
- **Analyzing Student Behavior** in classrooms to improve learning outcomes.
- **Surveillance Systems** to detect suspicious behavior.

## Key Features

- **Data Preparation**: Uses `ImageDataGenerator` to handle image processing, resize them to 48x48, convert them to grayscale, and prepare the data for model training.
- **CNN Architecture**: A sequential model built in **Keras** with multiple convolutional layers, pooling, and dropout for regularization.
- **Training**: The model is trained using categorical cross-entropy as the loss function, and **Adam** optimizer with a learning rate of `0.0001`.
- **Real-time Emotion Detection**: Integrates **OpenCV** to perform real-time emotion detection through video feeds, displaying the predicted emotion on detected faces.
  
## Approach & Methodology

### 1. Data Preparation
We used the **ImageDataGenerator** class from **Keras** to preprocess images. The input images were resized to `48x48`, converted to grayscale, and batch processing was set to `64`. The dataset included seven emotion classes:
- Angry
- Disgust
- Fear
- Happy
- Sad
- Surprise
- Neutral

### 2. CNN Architecture
The model consists of several convolutional layers, max-pooling layers, and dropout to reduce overfitting. The key layers are:
- **First Convolutional Layer**: 32 filters, 3x3 kernel size, ReLU activation, input shape of `48x48x1`.
- **Subsequent Convolutional Layers**: 64 filters with similar configurations.
- **Fully Connected Layer**: A dense layer of 1024 units followed by a 50% dropout rate.
- **Output Layer**: A softmax activation function for classifying emotions.

### 3. Model Training
The model was compiled with the following configuration:
- Loss Function: `categorical_crossentropy`
- Optimizer: **Adam** with a learning rate of `0.0001` and a decay of `1e-6`
- Metric: Accuracy
The model was trained for `50 epochs` using the training and validation datasets.

### 4. Saving the Model
After training, the model’s architecture and weights were saved for future use:
- Model architecture: `emotion_model.json`
- Model weights: `emotion_model.h5`

### 5. Real-Time Emotion Detection
For real-time detection, we used **OpenCV** to access the webcam or video feed. Detected faces were resized to `48x48` and passed through the CNN for emotion prediction. The predicted emotion was displayed on the screen in real-time.

## How to Use

- **Train the model** using the provided `train_model.py` script.
- **Run real-time emotion detection** using the `real_time_emotion_detection.py` script, which utilizes a webcam feed:
  ```bash
  python real_time_emotion_detection.py

## Files

- `train_model.py`: Script to train the CNN on the dataset.
- `real_time_emotion_detection.py`: Script to detect emotions in real-time using a webcam feed.
- `emotion_model.json`: Saved model architecture.
- `emotion_model.h5`: Saved model weights.
- `README.md`: This file.

## Results

The trained CNN model achieved high accuracy in classifying the seven emotions from facial expressions, with potential applications in:

- **Automotive**: Detecting driver fatigue and emotions.
- **Education**: Monitoring student engagement and emotions.
- **Surveillance**: Detecting suspicious or harmful emotions.

## Conclusion

This project successfully demonstrated how deep learning techniques like CNNs can be applied to facial emotion detection with high accuracy. Future improvements could include expanding the dataset, refining the model, and deploying it on edge devices for real-time performance.

## References

- [Keras Documentation](https://keras.io/)
- [TensorFlow Documentation](https://www.tensorflow.org/)
- [OpenCV Documentation](https://opencv.org/)
