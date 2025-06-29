import os
import numpy as np
import joblib
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
from sklearn.metrics import classification_report
from models.metrics import metrics_values

def run_cnn2d(
    data_path="src/",
    models_path="models/"
):
    # 📦 Cargar datos
    x_train, y_train, feature_names = joblib.load(os.path.join(data_path, "train.pkl"))
    x_val, y_val, _ = joblib.load(os.path.join(data_path, "val.pkl"))
    x_test, y_test, _ = joblib.load(os.path.join(data_path, "test.pkl"))

    # 🏷️ Cargar nombres de clases
    class_labels_path = os.path.join(data_path, "class_labels.npy")
    if os.path.exists(class_labels_path):
        class_names = np.load(class_labels_path, allow_pickle=True).tolist()
        print("✅ Clases cargadas:", class_names)
    else:
        raise FileNotFoundError("❌ No se encontró el archivo class_labels.npy.")

    n_classes = y_train.shape[1]

    # ✅ Normalizar si es necesario
    x_train = x_train.astype("float32") / 255.0
    x_val = x_val.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0

    # ✅ Añadir canal para CNN: (128, 128) -> (128, 128, 1)
    if len(x_train.shape) == 3:
        x_train = np.expand_dims(x_train, -1)
        x_val = np.expand_dims(x_val, -1)
        x_test = np.expand_dims(x_test, -1)

    input_shape = x_train.shape[1:]

    # 🧠 Modelo CNN 2D
    model = models.Sequential([
        layers.Input(shape=input_shape),
        layers.Conv2D(32, kernel_size=(3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.BatchNormalization(),

        layers.Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.BatchNormalization(),

        layers.Conv2D(128, kernel_size=(3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.BatchNormalization(),

        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(n_classes, activation='softmax')
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    model.summary()

    # ⏱️ Callbacks
    early_stop = callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

    # 🚆 Entrenamiento
    print("🚀 Entrenando CNN 2D...")
    history = model.fit(
        x_train, y_train,
        validation_data=(x_val, y_val),
        epochs=50,
        batch_size=32,
        callbacks=[early_stop],
        verbose=2
    )

    # 💾 Guardar modelo
    model_path = os.path.join(models_path, "cnn2d.h5")
    model.save(model_path)
    print(f"📦 Modelo CNN 2D guardado en: {model_path}")

    # 🧪 Evaluación
    y_pred_probs = model.predict(x_test)
    y_pred_labels = np.argmax(y_pred_probs, axis=1)
    y_test_labels = np.argmax(y_test, axis=1)

    print("📈 Evaluación final en conjunto de prueba:")
    metrics_values(y_test_labels, y_pred_labels, class_names)

    return model, x_test, feature_names
