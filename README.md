🌌 Supernova Classification with Deep Learning

This project implements a Neural Network (NN) to classify supernovae into three categories using astrophysical data.

🔬 Pipeline Overview

Data Preprocessing

Read raw .csv / .dat files with Pandas.

Clean and structure the dataset, applying normalization and One-Hot Encoding with scikit-learn.

Reshape the input data into tensors suitable for NN training.

Model Architecture (TensorFlow/Keras)

Convolutional layers (Conv2D) for feature extraction.

Batch Normalization to stabilize and speed up training.

MaxPooling & Dropout to reduce overfitting.

Dense layers with softmax activation for 3-class classification.

EarlyStopping callback to restore best weights and avoid overfitting.

Training

Framework: TensorFlow / Keras.

Optimizer: Adam.

Loss: Categorical Crossentropy.

Evaluation metrics: Accuracy and ROC-AUC (using scikit-learn).

Evaluation

Achieves strong test accuracy and reliable AUC scores.

Outputs predicted probabilities for each of the 3 classes.

🛠️ Technologies Used

TensorFlow / Keras → Model building & training

scikit-learn → Preprocessing & evaluation metrics

NumPy → Matrix operations

Pandas → Data loading & manipulation
