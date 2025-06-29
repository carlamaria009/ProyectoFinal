# Importación de librerías y dataset
import kagglehub
import pandas as pd
import numpy as np
import os
import sys
import librosa
import seaborn as sns
import matplotlib.pyplot as plt

# to play the audio files
from IPython.display import Audio

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense, Dropout, Flatten
from tensorflow.keras.callbacks import ReduceLROnPlateau

import keras
from keras.callbacks import ReduceLROnPlateau
from keras.models import Sequential
from keras.layers import Dense, Conv1D, MaxPooling1D, Flatten, Dropout, BatchNormalization
from keras.optimizers import Adam
from keras.regularizers import l2

import joblib

def ejecutar_modelo_cnn(data_path="src/",
                   models_path="models/"):
  # 📦 Cargar datos de entrenamiento y prueba
  x_train, y_train, feature_names = joblib.load(os.path.join(data_path, "train.pkl"))
  x_test, y_test, feature_names = joblib.load(os.path.join(data_path, "test.pkl"))

  # 🔁 Aplanar entradas si están expandidas (para audio)
  if len(x_train.shape) > 2:
      x_train = x_train.reshape(x_train.shape[0], -1)
      x_test = x_test.reshape(x_test.shape[0], -1)

  # 🏷️ Cargar nombres de clases (emociones)
  class_labels_path = os.path.join(data_path, "class_labels.npy")
  if os.path.exists(class_labels_path):
      class_names = np.load(class_labels_path, allow_pickle=True)
  else:
      raise FileNotFoundError("❌ No se encontró el archivo class_labels.npy con los nombres de las emociones.")

  # 🔄 Decodificar etiquetas One-Hot a índices
  y_train_labels = np.argmax(y_train, axis=1)
  y_test_labels = np.argmax(y_test, axis=1)

  # Definición del modelo
  model = Sequential()
  model.add(Input(shape=(x_train.shape[1], 1)))

  # Aplicando Regularización L2
  model = Sequential()
  model.add(Input(shape=(x_train.shape[1], 1)))
  model.add(Dense(256, activation='relu', kernel_regularizer=l2(0.001)))
  model.add(Dropout(0.3))
  model.add(Dense(128, activation='relu', kernel_regularizer=l2(0.001)))
  model.add(Dropout(0.3))
  model.add(Dense(64, activation='relu', kernel_regularizer=l2(0.001)))
  model.add(Dropout(0.3))
  model.add(Flatten())
  model.add(Dense(6, activation='softmax'))

  # Compilar el modelo antes de entrenarlo
  # Compilar el modelo con una tasa de aprendizaje ajustada
  optimizer = Adam(learning_rate=0.0001)
  model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
  model.summary()


  model_cnn = model
  tf.keras.utils.plot_model(model_cnn, rankdir='LR',show_dtype=True)
  rlrp = ReduceLROnPlateau(monitor='loss', factor=0.4, verbose=0, patience=2, min_lr=0.0000000005)
  history=model_cnn.fit(x_train, y_train, batch_size=64, epochs=50, validation_data=(x_test, y_test), callbacks=[rlrp])

  # 💾 Guardar modelo entrenado
  model_path = os.path.join(models_path, "cnn.pkl")
  joblib.dump(model_cnn, model_path)
  print(f"📦 Modelo guardado en: {model_path}")
  #grafico_perdida(history)

  plt.figure(figsize=(12, 4))
  # Pérdida
  plt.subplot(1, 2, 1)
  plt.plot(history.history['loss'], label='Pérdida de Entrenamiento')
  plt.plot(history.history['val_loss'], label='Pérdida de Validación')
  plt.title('Pérdida durante el Entrenamiento')
  plt.xlabel('Épocas')
  plt.ylabel('Pérdida')
  plt.legend()

  # Precisión
  plt.subplot(1, 2, 2)
  plt.plot(history.history['accuracy'], label='Precisión de Entrenamiento')
  plt.plot(history.history['val_accuracy'], label='Precisión de Validación')
  plt.title('Precisión durante el Entrenamiento')
  plt.xlabel('Épocas')
  plt.ylabel('Precisión')
  plt.legend()

  plt.tight_layout()
  plt.show()

  # 6. Evaluación y Comparación del Modelo

  # Predicciones en datos de prueba
  # Cargar las clases guardadas
  class_labels = np.load(os.path.join(data_path, "class_labels.npy"), allow_pickle=True)  

  # Crear un nuevo encoder con esas clases
  encoder = OneHotEncoder(categories=[class_labels], handle_unknown='ignore', sparse_output=False)

  # "Ajustar" el encoder con los nombres de clase
  # Esto es necesario para que sklearn lo considere "fitted"
  encoder.fit(np.array(class_labels).reshape(-1, 1))

  pred_test = model.predict(x_test)
  y_pred = encoder.inverse_transform(pred_test)

  # Crear DataFrame de predicciones
  df = pd.DataFrame(columns=['Predicted Labels', 'Actual Labels'])
  df['Predicted Labels'] = y_pred.flatten()
  df['Actual Labels'] = encoder.inverse_transform(y_test).flatten()
  df.head(10)

  # Matriz de confusión
  cm = confusion_matrix(encoder.inverse_transform(y_test), y_pred)
  plt.figure(figsize=(12, 10))
  cm_df = pd.DataFrame(cm, index=encoder.categories_[0], columns=encoder.categories_[0])
  sns.heatmap(cm_df, linecolor='white', cmap='Blues', linewidth=1, annot=True, fmt='')
  plt.title('Matriz de confusión', size=20)
  plt.xlabel('Predicción', size=14)
  plt.ylabel('Real', size=14)
  plt.show()

  # Informe de clasificación
  print(classification_report(encoder.inverse_transform(y_test), y_pred, zero_division=0))

  return model_cnn, x_test

#ejecutar_modelo_cnn()
