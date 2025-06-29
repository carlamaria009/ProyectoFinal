def run_cnn2d(data_path="src/", models_path="models/"):
    import os
    import numpy as np
    import joblib
    import tensorflow as tf
    from tensorflow.keras import layers, models, callbacks, regularizers
    from sklearn.metrics import classification_report
    from models.metrics import metrics_values

    # 📦 Cargar datos
    x_train, y_train, feature_names = joblib.load(os.path.join(data_path, "train.pkl"))
    x_val, y_val, _ = joblib.load(os.path.join(data_path, "val.pkl"))
    x_test, y_test, _ = joblib.load(os.path.join(data_path, "test.pkl"))

    # Asegurarse de que tenga canal de entrada (para seguridad)
    if x_train.ndim == 3:
        x_train = np.expand_dims(x_train, -1)
        x_val = np.expand_dims(x_val, -1)
        x_test = np.expand_dims(x_test, -1)

    # 🏷️ Cargar nombres de clases
    class_labels_path = os.path.join(data_path, "class_labels.npy")
    if os.path.exists(class_labels_path):
        class_names = np.load(class_labels_path, allow_pickle=True).tolist()
        print("✅ Clases cargadas:", class_names)
    else:
        raise FileNotFoundError("❌ No se encontró el archivo class_labels.npy.")

    n_classes = y_train.shape[1]

    # 🧠 Modelo CNN 2D con L2 y BatchNormalization
    model = models.Sequential([
        layers.Input(shape=(128, 128, 1)),

        layers.Conv2D(32, (3, 3), activation='relu',
                      kernel_regularizer=regularizers.l2(0.01)),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),

        layers.Conv2D(64, (3, 3), activation='relu',
                      kernel_regularizer=regularizers.l2(0.01)),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),

        layers.Conv2D(128, (3, 3), activation='relu',
                      kernel_regularizer=regularizers.l2(0.01)),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),

        layers.Flatten(),
        layers.Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.01)),
        layers.BatchNormalization(),
        layers.Dropout(0.5),

        layers.Dense(n_classes, activation='softmax')
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    early_stop = callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

    print("🚀 Entrenando modelo CNN 2D con L2 y BatchNormalization...")
    history = model.fit(
        x_train, y_train,
        validation_data=(x_val, y_val),
        epochs=25,
        batch_size=32,
        callbacks=[early_stop],
        verbose=2
    )

    # 💾 Guardar modelo
    model_path = os.path.join(models_path, "cnn2d_model.h5")
    model.save(model_path)
    print(f"📦 Modelo CNN 2D guardado en: {model_path}")

    # 🧪 Evaluación en test
    y_pred_probs = model.predict(x_test)
    y_pred_labels = np.argmax(y_pred_probs, axis=1)
    y_test_labels = np.argmax(y_test, axis=1)

    print("📈 Evaluación final en conjunto de prueba:")
    metrics_values(y_test_labels, y_pred_labels, class_names)

    return model, x_test, feature_names
