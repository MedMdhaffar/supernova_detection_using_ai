# 🌌 Supernova Classification with Deep Learning  

This project implements a **Neural Network (NN)** to classify **supernovae** into **three categories** using astrophysical data.  

---

## 🔬 Pipeline Overview  

- **Data Preprocessing**  
  - Load raw `.csv` / `.dat` files with **Pandas**  
  - Apply **One-Hot Encoding** and normalization with **scikit-learn**  
  - Reshape input data into tensors for NN training  

- **Model Architecture (TensorFlow/Keras)**  
  - `Conv2D` layers for feature extraction  
  - `BatchNormalization` to stabilize and speed up training  
  - `MaxPooling` & `Dropout` to reduce overfitting  
  - Fully connected `Dense` layers with `softmax` activation for **3-class classification**  
  - `EarlyStopping` to avoid overfitting and restore best weights  

- **Training**  
  - Framework: **TensorFlow / Keras**  
  - Optimizer: **Adam**  
  - Loss: **Categorical Crossentropy**  
  - Metrics: **Accuracy** and **ROC-AUC** (via scikit-learn)  

- **Evaluation**  
  - Model is tested on unseen data (`X_test`, `y_test`)  
  - Outputs include **accuracy**, **loss**, and **AUC score**  

---

## 🛠️ Technologies Used  

- [TensorFlow](https://www.tensorflow.org/) / [Keras](https://keras.io/) → Model building & training  
- [scikit-learn](https://scikit-learn.org/) → Preprocessing & metrics  
- [NumPy](https://numpy.org/) → Numerical operations  
- [Pandas](https://pandas.pydata.org/) → Data handling  

---

## 🚀 Usage  

```python
# Train model
model.fit(X_train, y_train, 
          epochs=50, 
          validation_split=0.1, 
          callbacks=[early_stop], 
          batch_size=10)

# Evaluate
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f"Test accuracy: {test_acc:.4f}")

# Predict probabilities
y_pred_probs = model.predict(X_test)

# Compute AUC
from sklearn.metrics import roc_auc_score
auc = roc_auc_score(y_test, y_pred_probs)
print("AUC Score:", auc)
