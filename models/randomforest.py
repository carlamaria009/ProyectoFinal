from models.metrics import metrics_values
from sklearn.ensemble import RandomForestClassifier
import os
import numpy as np
import joblib

def run_random_forest(
    data_path="src/",
    models_path="models/"
):
    # 📦 Cargar datos
    x_train, y_train, feature_names = joblib.load(os.path.join(data_path, "train.pkl"))
    x_val, y_val, _ = joblib.load(os.path.join(data_path, "val.pkl"))
    x_test, y_test, _ = joblib.load(os.path.join(data_path, "test.pkl"))

    # 🔁 Aplanar entradas si es necesario
    if len(x_train.shape) > 2:
        x_train = x_train.reshape(x_train.shape[0], -1)
        x_val = x_val.reshape(x_val.shape[0], -1)
        x_test = x_test.reshape(x_test.shape[0], -1)

    # 🏷️ Cargar nombres de clases
    class_labels_path = os.path.join(data_path, "class_labels.npy")
    if os.path.exists(class_labels_path):
        class_names = np.load(class_labels_path, allow_pickle=True).tolist()
        print("✅ Clases cargadas:", class_names)
    else:
        raise FileNotFoundError("❌ No se encontró el archivo class_labels.npy.")

    # 🔄 Decodificar etiquetas One-Hot
    y_train_labels = np.argmax(y_train, axis=1)
    y_val_labels = np.argmax(y_val, axis=1)
    y_test_labels = np.argmax(y_test, axis=1)

    # 🧪 Combinar entrenamiento y validación
    x_combined = np.concatenate((x_train, x_val), axis=0)
    y_combined = np.concatenate((y_train_labels, y_val_labels), axis=0)

    # 🌲 Entrenar modelo RandomForest con parámetros por defecto simples
    print("🚀 Entrenando RandomForestClassifier...")
    rf = RandomForestClassifier(
        n_estimators=100,
        max_depth=20,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    )
    rf.fit(x_combined, y_combined)

    # 💾 Guardar modelo
    os.makedirs(models_path, exist_ok=True)
    model_path = os.path.join(models_path, "random_forest_simple.pkl")
    joblib.dump(rf, model_path)
    print(f"📦 Modelo guardado en: {model_path}")

    # 📈 Evaluación en test
    y_test_pred = rf.predict(x_test)
    print("📈 Evaluación final en conjunto de prueba:")
    metrics_values(y_test_labels, y_test_pred, class_names)

    return rf, x_test, feature_names

