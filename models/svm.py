from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
import joblib
import numpy as np
import os
from models.metrics import metrics_values

def run_svm(
    data_path="src/",
    models_path="models/"
):
    # 📦 Cargar datos
    x_train, y_train, feature_names = joblib.load(os.path.join(data_path, "train.pkl"))
    x_val, y_val, _ = joblib.load(os.path.join(data_path, "val.pkl"))
    x_test, y_test, _ = joblib.load(os.path.join(data_path, "test.pkl"))

    # 🔁 Aplanar entradas si están expandidas
    if len(x_train.shape) > 2:
        x_train = x_train.reshape(x_train.shape[0], -1)
        x_val = x_val.reshape(x_val.shape[0], -1)
        x_test = x_test.reshape(x_test.shape[0], -1)

    # 🏷️ Cargar nombres de clases
    class_labels_path = os.path.join(data_path, "class_labels.npy")
    if os.path.exists(class_labels_path):
        class_names = np.load(class_labels_path, allow_pickle=True).tolist()
    else:
        raise FileNotFoundError("❌ No se encontró el archivo class_labels.npy con los nombres de las emociones.")

    # 🔄 Decodificar etiquetas One-Hot a índices
    y_train_labels = np.argmax(y_train, axis=1)
    y_val_labels = np.argmax(y_val, axis=1)
    y_test_labels = np.argmax(y_test, axis=1)

    # 🧪 Combinar train y val para entrenamiento
    x_combined = np.concatenate((x_train, x_val), axis=0)
    y_combined = np.concatenate((y_train_labels, y_val_labels), axis=0)

    # 🔍 Definir grid de parámetros
    param_grid = {
    'C': [1, 10],                # Menos valores
    'kernel': ['linear', 'rbf'], # Elimina 'poly' que es muy lento
    'gamma': ['scale']           # Solo una opción
    }

    print("🔍 Ejecutando GridSearchCV para SVM...")
    grid = GridSearchCV(
        estimator=SVC(probability=False, random_state=42),
        param_grid=param_grid,
        scoring='accuracy',
        cv=3,
        verbose=1,
        n_jobs=-1
    )

    grid.fit(x_combined, y_combined)
    print(f"✅ Mejores parámetros encontrados: {grid.best_params_}")

    best_svm = grid.best_estimator_

    # 💾 Guardar modelo optimizado
    model_path = os.path.join(models_path, "svm_best_model.pkl")
    joblib.dump(best_svm, model_path)
    print(f"📦 Modelo SVM guardado en: {model_path}")

    # 📊 Evaluar en test
    y_pred_test = best_svm.predict(x_test)
    print("📈 Evaluación en conjunto de prueba:")
    metrics_values(y_test_labels, y_pred_test, class_names)

    return best_svm, x_test, feature_names

