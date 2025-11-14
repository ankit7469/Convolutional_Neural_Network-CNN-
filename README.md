Convolutional Neural Network
---------------------------------------

#  Convolutional Neural Network (CNN) — Image Classification (TensorFlow/Keras)

This project demonstrates a **Convolutional Neural Network (CNN)** built using **TensorFlow & Keras**  
to classify images from the **CIFAR-10 dataset**.  
The model learns features such as edges, textures, colors, and shapes using convolution operations.

---

##  Project Highlights

- Used **CIFAR-10** image dataset (60,000 color images)
- Built a **multi-layer CNN** for image classification
- Used **Conv2D + MaxPooling** for feature extraction
- Used **Softmax** for multi-class output
- Trained model on 10 classes (airplane, car, cat, dog, etc.)
- Visualized predictions with actual images

---

##  What I Learned

### ✔ Convolutional Neural Networks (CNN)
- CNNs automatically learn features from images  
- Best suited for **image classification** and **computer vision** tasks  

### ✔ Convolution (Conv2D)
- Detects edges, corners, patterns  
- Extracts important features

### ✔ MaxPooling
- Reduces spatial size  
- Makes model faster & reduces overfitting

### ✔ Softmax Activation
- Converts outputs into class probabilities  
- Used for multi-class classification

### ✔ Accuracy & Loss Visualization
- Helps understand training behavior  
- Shows model improvement over epochs

---

## 🧾 Tech Used
- Python  
- TensorFlow / Keras  
- NumPy  
- Matplotlib  

---

## 💻 Code Overview

```python
model = keras.Sequential([
    layers.Conv2D(32, (3,3), activation='relu', input_shape=(32,32,3)),
    layers.MaxPooling2D(2,2),

    layers.Conv2D(64, (3,3), activation='relu'),
    layers.MaxPooling2D(2,2),

    layers.Conv2D(128, (3,3), activation='relu'),
    layers.MaxPooling2D(2,2),

    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.3),

    layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

history = model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test))
