[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1sEAsQ1rvcUNI2nRA0uPKdaN42kfCsm02?usp=sharing)

# 🖼️ Image Classification using TensorFlow and Keras

![TensorFlow](https://img.shields.io/badge/TensorFlow-2.0+-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)
![Keras](https://img.shields.io/badge/Keras-Deep%20Learning-D00000?style=for-the-badge&logo=keras&logoColor=white)
![CNN](https://img.shields.io/badge/CNN-Computer%20Vision-76B900?style=for-the-badge&logo=nvidia&logoColor=white)
![Python](https://img.shields.io/badge/Python-3.8-3776AB?style=for-the-badge&logo=python&logoColor=white)

A deep learning project implementing a Convolutional Neural Network (CNN) to classify images from the CIFAR-10 dataset. This model can identify objects across 10 different categories with high accuracy and can be easily adapted for custom image recognition tasks.

## 📋 Project Overview

Image Classification is a fundamental computer vision task that allows machines to categorize images into predefined classes. This project:

- Builds and trains a CNN model from scratch using TensorFlow and Keras
- Uses the CIFAR-10 dataset containing 60,000 32x32 color images across 10 classes
- Implements data augmentation techniques to improve model generalization
- Demonstrates how to apply transfer learning for enhanced performance
- Provides utilities for making predictions on custom user images

## 🔍 CIFAR-10 Dataset

The CIFAR-10 dataset consists of:
- 50,000 training images
- 10,000 testing images
- 10 object classes: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck
- 32x32 pixel color images

Each image is labeled with exactly one category, making this a multi-class classification problem.

## 🧠 What is a CNN?

A Convolutional Neural Network (CNN) is a specialized deep learning architecture designed specifically for processing pixel data in images:

- **Convolutional layers** extract features from images using sliding filters
- **Pooling layers** reduce dimensionality while preserving important information
- **Fully connected layers** interpret extracted features for classification

CNNs excel at:
- Image and video recognition
- Object detection and segmentation
- Face recognition
- Medical image analysis

## 🏗️ Model Architecture

Our implemented CNN model consists of:

```
Model: Sequential
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
Conv2D (Conv2D)              (None, 30, 30, 32)        896       
_________________________________________________________________
BatchNormalization          (None, 30, 30, 32)         128       
_________________________________________________________________
MaxPooling2D (MaxPooling2D)  (None, 15, 15, 32)        0         
_________________________________________________________________
Conv2D (Conv2D)              (None, 13, 13, 64)        18496     
_________________________________________________________________
BatchNormalization          (None, 13, 13, 64)         256       
_________________________________________________________________
MaxPooling2D (MaxPooling2D)  (None, 6, 6, 64)          0         
_________________________________________________________________
Conv2D (Conv2D)              (None, 4, 4, 128)         73856     
_________________________________________________________________
BatchNormalization          (None, 4, 4, 128)          512       
_________________________________________________________________
MaxPooling2D (MaxPooling2D)  (None, 2, 2, 128)         0         
_________________________________________________________________
Flatten (Flatten)            (None, 512)               0         
_________________________________________________________________
Dense (Dense)                (None, 128)               65664     
_________________________________________________________________
Dropout (Dropout)            (None, 128)               0         
_________________________________________________________________
Dense (Dense)                (None, 10)                1290      
=================================================================
Total params: 161,098
Trainable params: 160,650
Non-trainable params: 448
_________________________________________________________________
```

## 📊 Performance Metrics

The model achieves:
- **Training Accuracy**: ~87%
- **Validation Accuracy**: ~83%
- **Test Accuracy**: ~82%

![Accuracy Plot](https://your-repo-path-to-image/accuracy_plot.png)
![Loss Plot](https://your-repo-path-to-image/loss_plot.png)

## 💻 Implementation Details

### Data Preprocessing

```python
# Data loading and normalization
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

# One-hot encode the labels
y_train = tf.keras.utils.to_categorical(y_train, 10)
y_test = tf.keras.utils.to_categorical(y_test, 10)

# Data augmentation
data_augmentation = tf.keras.Sequential([
  tf.keras.layers.experimental.preprocessing.RandomFlip("horizontal"),
  tf.keras.layers.experimental.preprocessing.RandomRotation(0.1),
  tf.keras.layers.experimental.preprocessing.RandomZoom(0.1),
])
```

### Model Building

```python
def build_cnn_model():
    model = tf.keras.Sequential([
        # First convolutional block
        tf.keras.layers.Conv2D(32, (3, 3), padding='valid', activation='relu', input_shape=(32, 32, 3)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        
        # Second convolutional block
        tf.keras.layers.Conv2D(64, (3, 3), padding='valid', activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        
        # Third convolutional block
        tf.keras.layers.Conv2D(128, (3, 3), padding='valid', activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        
        # Fully connected layers
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(10, activation='softmax')
    ])
    
    return model
```

### Training Process

```python
# Compile the model
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Add early stopping and learning rate reduction
callbacks = [
    tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True),
    tf.keras.callbacks.ReduceLROnPlateau(factor=0.1, patience=5)
]

# Train the model
history = model.fit(
    x_train, y_train,
    validation_split=0.2,
    epochs=100,
    batch_size=64,
    callbacks=callbacks
)
```

### Prediction on Custom Images

```python
def predict_image(image_path, model):
    # Load and preprocess the image
    img = tf.keras.preprocessing.image.load_img(
        image_path, target_size=(32, 32)
    )
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    # Make prediction
    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions[0])
    
    return predicted_class, predictions[0]
```

## 🚀 Getting Started

### Prerequisites
- Python 3.8 or higher
- TensorFlow 2.0+
- Keras
- NumPy
- Matplotlib

### Running the Project

1. Open the notebook in Google Colab using the badge at the top
2. Run all cells to train the model
3. Upload your own images for testing using the custom prediction function

## 🛠️ Resources Used

- **Python Version:** 3.8
- **Packages:** TensorFlow, Keras, matplotlib, numpy
- **Model:** Sequential CNN
- **IDE:** Google Colaboratory
- **CIFAR-10 Dataset:** 
  - [Kaggle Competition](https://www.kaggle.com/c/cifar-10)
  - [Dataset Details](https://www.cs.toronto.edu/~kriz/cifar.html)

## 📈 Future Improvements

- Implement more sophisticated CNN architectures (ResNet, Inception)
- Apply transfer learning from models pre-trained on ImageNet
- Add real-time image classification using webcam input
- Expand to other datasets with higher resolution images
- Optimize model for mobile deployment with TensorFlow Lite
- Experiment with model quantization for reduced size and inference time
- Implement model pruning to remove redundant connections

## 📚 References

- Krizhevsky, A. (2009). Learning Multiple Layers of Features from Tiny Images.
- Goodfellow, I., Bengio, Y., & Courville, A. (2016). Deep Learning. MIT Press.
- TensorFlow Documentation: [https://www.tensorflow.org/](https://www.tensorflow.org/)
- Keras Documentation: [https://keras.io/](https://keras.io/)

## 👨‍💻 Author

Dishant - [GitHub Profile](https://github.com/Dishant27)

---

**Note**: This project demonstrates deep learning concepts for image classification and can be extended for various computer vision applications.