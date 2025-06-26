from sklearn.svm import SVC
import joblib
import numpy as np
import os
from models.metrics import metrics_values

def run_svm(
    data_path="src/",
    models_path="models/"
):
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

    # 🚀 Entrenar modelo SVM (kernel RBF por defecto)
    clf = SVC(probability=True, kernel='rbf', C=1.0, random_state=42)
    clf.fit(x_train, y_train_labels)

    # 💾 Guardar modelo entrenado
    model_path = os.path.join(models_path, "svm_model.pkl")
    joblib.dump(clf, model_path)
    print(f"📦 Modelo SVM guardado en: {model_path}")

    # 🔍 Hacer predicciones
    y_pred = clf.predict(x_test)

    metrics_values(y_test_labels, y_pred, class_names)
    
    return clf, x_test, feature_names
