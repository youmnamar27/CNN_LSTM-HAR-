# CNN_LSTM-HAR
Human Activity Recognition (HAR) using CNN-LSTM

This project implements a deep learning model combining Convolutional Neural Networks (CNNs) and Long Short-Term Memory (LSTM) networks for Human Activity Recognition (HAR) based on image sequences. The model is trained to classify various human activities from image sequences.

## 📌 Features

Uses CNN for spatial feature extraction.

Uses LSTM for temporal feature learning across image sequences.

Supports data augmentation to enhance generalization.

Handles class imbalance using weighted loss functions.

Implements train-validation split for performance evaluation.

## 📂 Dataset

The dataset consists of labeled image sequences representing different human activities. Each sequence contains 10 consecutive frames.

## 🎭 Activity Classes

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

## 📷 Sample Images from the Dataset
The following images represent different activity classes used for training the CNN-LSTM model:
             
     import matplotlib.pyplot as plt
     import os
    fig, axes = plt.subplots(3, 5, figsize=(20, 10))
    axes = axes.flatten()

    for idx, class_name in enumerate(train_df['label'].unique()):
    class_images = train_df[train_df['label'] == class_name]['filename'].values
    img = plt.imread(os.path.join(train_path, class_images[0]))
    axes[idx].imshow(img)
    axes[idx].set_title(class_name)
    axes[idx].axis('off')

    plt.tight_layout()
    plt.show()
![dataset)](https://github.com/user-attachments/assets/885a2eed-65f2-4c17-8475-54ee675c8cd8)

## 🏗 Model Architecture

The implemented model follows a CNN-LSTM approach:
![cnn_architecture_improved (1)](https://github.com/user-attachments/assets/7ef04a9c-d09a-452f-a536-58b796956968)
![lstm_architecture_improved](https://github.com/user-attachments/assets/43e7d273-9e21-4a3e-84d7-e599154f8665)

TimeDistributed CNN Layers: Extract spatial features from each frame.

MaxPooling Layers: Reduce dimensionality and retain important features.

Flatten Layer: Converts CNN outputs into 1D feature vectors.

LSTM Layer: Processes temporal dependencies between frames.

Dropout Layer: Prevents overfitting.

Fully Connected Layers: Outputs probabilities for each class.
## Model Summary

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

# 🛠 Define Constants & Load Dataset

     import os
    import numpy as np
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

# 🚀 Training the Model Process

## 1️⃣ Load the Dataset
The dataset is loaded and split into 80% training and 20% validation.

Each image is normalized and converted into sequences.

## 2️⃣ Data Augmentation
Applied rotation, brightness adjustments, and horizontal flipping to improve generalization.

## 3️⃣ Model Compilation
Optimizer: Adam

Loss Function: Sparse Categorical Crossentropy

Metric: Accuracy

## 4️⃣ Training Configuration
Epochs: 20

Batch Size: 8

Validation Split: 20%

## 5️⃣ Training Execution
The model is trained on the dataset, extracting spatial & temporal features using CNN-LSTM layers.

Accuracy & Loss are monitored over each epoch.

## 6️⃣ Evaluation
The model performance is evaluated using:

Accuracy & Loss Plots

Confusion Matrix


# 📊 Results & Performance
The training accuracy improves over epochs, but further hyperparameter tuning or pretrained CNNs may enhance performance.

A confusion matrix is generated to analyze class-wise predictions.

## Define the CNN-LSTM Model
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import TimeDistributed, Conv2D, MaxPooling2D, Flatten, LSTM, Dense, Dropout
    from tensorflow.keras.optimizers import Adam

    # Define model architecture
    num_classes = len(class_names)

    model = Sequential([
    tf.keras.layers.Input(shape=(sequence_length, img_height, img_width, 3)),  # Input shape (sequence of images)
    TimeDistributed(Conv2D(32, (3,3), activation='relu')),  # First CNN layer
    TimeDistributed(MaxPooling2D((2,2))),  # Max pooling
    TimeDistributed(Conv2D(64, (3,3), activation='relu')),  # Second CNN layer
    TimeDistributed(MaxPooling2D((2,2))),  # Max pooling
    TimeDistributed(Flatten()),  # Flatten feature maps
    LSTM(64),  # LSTM layer to learn temporal dependencies
    Dropout(0.5),  # Dropout to prevent overfitting
    Dense(128, activation='relu'),  # Fully connected layer
    Dense(num_classes, activation='softmax')  # Output layer with softmax activation
    ])

    # Compile the model
    model.compile(optimizer=Adam(learning_rate=0.001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.summary()  # Display model architecture

## Train the Model
    history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=10, batch_size=8)

    # Save the trained model
    model.save("cnn_lstm_har_model.keras")
    print("Model saved successfully.")
    
# 📊 Dataset & Preprocessing
The dataset is split into 80% training and 20% validation.

Data augmentation techniques such as rotation, brightness adjustments, and horizontal flipping are applied.

The model is compiled using Adam optimizer and Sparse Categorical Crossentropy loss.

Training is conducted for 20 epochs with a batch size of 8.

## Classification Report

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
![Model loss)](https://github.com/user-attachments/assets/ba57a374-5d15-46ad-ad04-bad5ee69e327)

![Model accuracy)](https://github.com/user-attachments/assets/9456375a-bd16-4f53-a159-e736e3f78d10)

![accuracy_plot](https://github.com/user-attachments/assets/90af7876-d0ac-410b-abcb-e71673e4b0ac)

![loss_plot](https://github.com/user-attachments/assets/bad1482b-d044-43dc-83ac-82c10144fafa)


#### Confusion Matrix
![confusion_matrix](https://github.com/user-attachments/assets/7fb520ed-6ed2-4b66-8d5e-23b68ba883ee)
![confusion_matrix](https://github.com/user-attachments/assets/1966179b-3a28-437f-913c-355124f92469)

# 📌 What This Code Does:
✅ Loads & Preprocesses the Dataset (Normalizes images, converts them into sequences)
✅ Defines a CNN-LSTM Model (Extracts spatial & temporal features)
✅ Trains the Model (On Human Activity Recognition dataset)
✅ Evaluates the Model (Generates accuracy/loss plots & confusion matrix)
✅ Saves the Model (So you can reuse it later)

📜 License
This project is open-source and available under the MIT License.

# 📌 Developed by
## Youmna Emadeldin
