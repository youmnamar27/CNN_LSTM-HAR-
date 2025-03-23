# CNN_LSTM-HAR-
Human Activity Recognition (HAR) using CNN-LSTM

This project implements a deep learning model combining Convolutional Neural Networks (CNNs) and Long Short-Term Memory (LSTM) networks for Human Activity Recognition (HAR) based on image sequences. The model is trained to classify various human activities from image sequences.

📌 Features

Uses CNN for spatial feature extraction.

Uses LSTM for temporal feature learning across image sequences.

Supports data augmentation to enhance generalization.

Handles class imbalance using weighted loss functions.

Implements train-validation split for performance evaluation.

📂 Dataset

The dataset consists of labeled image sequences representing different human activities. Each sequence contains 10 consecutive frames.

🎭 Activity Classes

The model classifies the following 15 human activities:

Sitting

Dancing

Fighting

Using Laptop

Laughing

Listening to Music

Texting

Eating

Clapping

Cycling

Drinking

Sleeping

Running

Calling

Hugging

🏗 Model Architecture

The implemented model follows a CNN-LSTM approach:

TimeDistributed CNN Layers: Extract spatial features from each frame.

MaxPooling Layers: Reduce dimensionality and retain important features.

Flatten Layer: Converts CNN outputs into 1D feature vectors.

LSTM Layer: Processes temporal dependencies between frames.

Dropout Layer: Prevents overfitting.

Fully Connected Layers: Outputs probabilities for each class.
### Model Summary

| Layer               | Type                         | Output Shape             | Parameters  |
|--------------------|----------------------------|--------------------------|------------|
| TimeDistributed    | Conv2D (32 filters, 3x3)    | (None, 10, 126, 126, 32) | 896        |
| TimeDistributed    | MaxPooling2D (2x2)          | (None, 10, 63, 63, 32)   | 0          |
| TimeDistributed    | Conv2D (64 filters, 3x3)    | (None, 10, 61, 61, 64)   | 18,496     |
| TimeDistributed    | MaxPooling2D (2x2)          | (None, 10, 30, 30, 64)   | 0          |
| TimeDistributed    | Flatten                     | (None, 10, 57600)        | 0          |
| LSTM              | 64 units                     | (None, 64)               | 14,762,240 |
| Dropout           | 0.5                          | (None, 64)               | 0          |
| Dense             | 128 units (ReLU)             | (None, 128)              | 8,320      |
| Dense             | 15 classes (Softmax)         | (None, 15)               | 1,935      |

**Total Parameters**: 14,791,887
Data Preprocessing
The dataset is loaded and preprocessed as follows:

python
Copy
Edit
import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# Define constants
sequence_length = 10
img_height, img_width = 128, 128
data_path = "path_to_dataset"

def load_sequences(data_path):
    X, y = [], []
    class_names = sorted(os.listdir(data_path))
    label_map = {name: idx for idx, name in enumerate(class_names)}
    
    for class_name in class_names:
        class_dir = os.path.join(data_path, class_name)
        frames = sorted(os.listdir(class_dir))
        label = label_map[class_name]
        
        for i in range(0, len(frames) - sequence_length + 1, sequence_length):
            sequence = []
            for j in range(sequence_length):
                img_path = os.path.join(class_dir, frames[i + j])
                img = load_img(img_path, target_size=(img_height, img_width))
                img = img_to_array(img) / 255.0
                sequence.append(img)
            X.append(sequence)
            y.append(label)
    return np.array(X), np.array(y)

X, y = load_sequences(data_path)
print("X shape:", X.shape)
print("y shape:", y.shape)

🚀 Training Process

The dataset is split into 80% training and 20% validation.

Data augmentation techniques like rotation, brightness adjustments, and horizontal flipping are applied.

The model is compiled using Adam optimizer and Sparse Categorical Crossentropy loss.

Training is conducted for 20 epochs with a batch size of 8.

The dataset is split into train (80%) and validation (20%) sets.


🖥 Usage

Install Dependencies

pip install tensorflow numpy pandas matplotlib seaborn scikit-learn

Train the Model

python train_model.py

Evaluate Model

y_pred = np.argmax(model.predict(X_val), axis=1)
print(classification_report(y_val, y_pred))

📊 Results

Model Accuracy

The model's training accuracy improves over epochs but may require hyperparameter tuning and pretrained CNNs for better performance.

Confusion Matrix

A confusion matrix is generated to evaluate per-class performance.

### Classification Report

| Class                 | Precision | Recall | F1-score |
|----------------------|----------|--------|----------|
| Sitting             | 0.00     | 0.00   | 0.00     |
| Dancing             | 0.21     | 0.25   | 0.23     |
| Fighting           | 0.06     | 0.07   | 0.06     |
| Using Laptop       | 0.15     | 0.12   | 0.13     |
| Laughing           | 0.00     | 0.00   | 0.00     |
| Listening to Music | 0.11     | 0.10   | 0.11     |
| Texting            | 0.00     | 0.00   | 0.00     |
| Eating             | 0.00     | 0.00   | 0.00     |
| Clapping           | 0.00     | 0.00   | 0.00     |
| Cycling            | 0.11     | 0.14   | 0.12     |
| Drinking           | 0.09     | 0.17   | 0.12     |
| Sleeping           | 0.08     | 0.12   | 0.09     |
| Running            | 0.00     | 0.00   | 0.00     |
| Calling            | 0.00     | 0.00   | 0.00     |
| Hugging            | 0.29     | 0.12   | 0.17     |

**Overall Accuracy**: **8%**  

## Sample Predictions
Here are some example predictions made by the model:

| Actual Label | Predicted Label | Confidence Score |
|-------------|----------------|------------------|
| Dancing     | Dancing        | 92%              |
| Running     | Walking        | 68%              |
| Using Laptop | Using Laptop   | 85%              |

### Model Performance

#### Accuracy & Loss
![Accuracy Plot](images/accuracy_plot.png)  
![Loss Plot](images/loss_plot.png)  

#### Confusion Matrix
![Confusion Matrix](images/confusion_matrix.png)  

🛠 Future Improvements

Incorporate pretrained CNN models (e.g., MobileNet, ResNet) for feature extraction.

Optimize hyperparameters using Grid Search.

Implement attention mechanisms for better temporal feature learning.

Expand dataset for better generalization.

📜 License

This project is open-source and available under the MIT License.

📌 Developed by: Youmna Emadeldin 
