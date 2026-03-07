# Image-Based-Animal-Type-Classification-for-Cow-and-Buffaloes

## Overview
This project implements a Convolutional Neural Network (CNN) using TensorFlow and Keras to classify images as either 'Cow' or 'Buffalo'. The model is trained on a custom dataset of cow and buffalo images, split into training and testing sets, with data augmentation applied to the training data.

## Dataset
The dataset consists of images categorized into 'Cow' and 'Buffalo' classes. The images are expected to be located in Google Drive at `/content/drive/MyDrive/Colab Notebooks/datasets/Cow` and `/content/drive/MyDrive/Colab Notebooks/datasets/Buffalo`.

-   **Initial Count**: Each class (Cow and Buffalo) initially contains 515 images.
-   **Data Split**: The dataset is split into 80% for training and 20% for testing. 
    -   Training data: 824 images (412 Cow, 412 Buffalo)
    -   Testing data: 206 images (103 Cow, 103 Buffalo)

## Model Architecture
The classification model is a Sequential CNN with the following layers:

-   **Input Layer**: `(150, 150, 3)` for RGB images resized to 150x150 pixels.
-   **Block 1**:
    -   `Conv2D(32, (3,3), activation='relu')`
    -   `BatchNormalization()`
    -   `MaxPooling2D(2,2)`
-   **Block 2**:
    -   `Conv2D(64, (3,3), activation='relu')`
    -   `BatchNormalization()`
    -   `MaxPooling2D(2,2)`
-   **Block 3**:
    -   `Conv2D(128, (3,3), activation='relu')`
    -   `BatchNormalization()`
    -   `MaxPooling2D(2,2)`
-   **Block 4**:
    -   `Conv2D(256, (3,3), activation='relu')`
    -   `BatchNormalization()`
    -   `MaxPooling2D(2,2)`
-   **Flatten Layer**
-   **Dense Layers**:
    -   `Dense(256, activation='relu')`
    -   `Dropout(0.5)`
-   **Output Layer**:
    -   `Dense(1, activation='sigmoid')` (for binary classification)

### Compilation
The model is compiled with:
-   **Optimizer**: `adam`
-   **Loss Function**: `binary_crossentropy`
-   **Metrics**: `accuracy`

## Data Augmentation
`ImageDataGenerator` is used for training data augmentation to improve model generalization. The augmentation techniques include:
-   `rescale=1./255`
-   `rotation_range=20`
-   `zoom_range=0.2`
-   `shear_range=0.2`
-   `horizontal_flip=True`

Test data is only rescaled.

## Installation and Setup
To run this project, you'll need Python and the following libraries:

-   `tensorflow`
-   `keras`
-   `matplotlib`
-   `numpy`
-   `shutil`
-   `os`

You can install the required Python packages using pip:
```bash
pip install tensorflow matplotlib numpy
```

Make sure your dataset is structured as follows in your Google Drive:
```
/content/drive/MyDrive/Colab Notebooks/datasets/
├── Buffalo/
│   ├── Buffalo_1.jpg
│   ├── ...
├── Cow/
│   ├── Cow_1.jpg
│   ├── ...
```

## Usage
1.  **Mount Google Drive**: Execute the cell to mount your Google Drive to access the dataset.
2.  **Prepare Data**: The notebook automatically creates `train` and `test` directories within the `datasets` folder and splits the images.
3.  **Build and Train Model**: Run the cells to define the CNN architecture, compile it, and train it using the `model.fit()` method.
4.  **Evaluate and Predict**: After training, you can evaluate the model's performance and use it to make predictions on new images.

## Results
The model was trained for 20 epochs.

During training, the accuracy generally improved, reaching a high training accuracy, and a notable validation accuracy was observed towards the end.

Example Prediction:
-   For `Buffalo_1.jpg`, the model predicted: `Buffalo 🐃` with `Confidence: 100.0 %`

Performance over epochs:
| Epoch | Accuracy (Train) | Loss (Train) | Accuracy (Validation) | Loss (Validation) |
|-------|------------------|--------------|-----------------------|-------------------|
| 1     | 0.7131           | 1.9639       | 0.6262                | 0.7763            |
| ...   | ...              | ...          | ...                   | ...               |
| 18    | 0.9616           | 0.1188       | 0.9029                | 0.5586            |
| 19    | 0.9335           | 0.1423       | 0.8398                | 0.8020            |
| 20    | 0.9590           | 0.1081       | 0.8447                | 0.8076            |

_(Note: Initial validation loss was high, and accuracy fluctuated, suggesting potential overfitting or complex data dynamics that data augmentation helped address partially.)_

## Future Improvements
-   **Transfer Learning**: Utilize pre-trained models (e.g., VGG, ResNet, Inception) for potentially better performance with less data.
-   **Hyperparameter Tuning**: Optimize learning rate, batch size, and network architecture parameters.
-   **More Data Augmentation**: Experiment with more advanced augmentation techniques.
-   **Larger Dataset**: Collect more diverse images of cows and buffaloes.
-   **Regularization**: Implement more robust regularization techniques to combat overfitting.

## Contact
For any questions or suggestions, please open an issue in this repository.
