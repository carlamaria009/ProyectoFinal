from imblearn.over_sampling import SMOTE
import os
import numpy as np
import joblib

def balancear_smote(X_train, Y_train):
  """
  Aplica la técnica SMOTE para balancear el conjunto de entrenamiento.

  Args:
    X_train (pd.DataFrame or np.ndarray): Características del conjunto de entrenamiento.
    Y_train (pd.Series or np.ndarray): Etiquetas del conjunto de entrenamiento.

  Returns:
    tuple: Un tuple que contiene:
      - X_res (pd.DataFrame or np.ndarray): Características balanceadas.
      - y_res (pd.Series or np.ndarray): Etiquetas balanceadas.
  """
  smote = SMOTE(random_state=123)
  X_res, y_res = smote.fit_resample(X_train, Y_train)
  print("✅ Balanceo con SMOTE realizado:")
  print(f"Forma de X_train después de SMOTE: {X_res.shape}")
  print(f"Forma de Y_train después de SMOTE: {y_res.shape}")
  return X_res, y_res

def balancear_smotepkl():
    import os
    import joblib
    import numpy as np
    from imblearn.over_sampling import SMOTE
    from sklearn.preprocessing import OneHotEncoder

    data_path = "src/"
    
    # Cargar datos originales
    X_train, Y_train, feature_names = joblib.load(os.path.join(data_path, "train.pkl"))
    
    # Aplanar X_train si tiene más de 2 dimensiones (ej: (n_samples, n_feat, 1) -> (n_samples, n_feat))
    if len(X_train.shape) > 2:
        X_train_2d = X_train.reshape(X_train.shape[0], -1)
    else:
        X_train_2d = X_train
    
    # Convertir Y_train de one-hot a etiquetas numéricas si aplica
    if len(Y_train.shape) > 1 and Y_train.shape[1] > 1:
        Y_train_labels = np.argmax(Y_train, axis=1)
    else:
        Y_train_labels = Y_train
    
    smote = SMOTE(random_state=123)
    X_res, y_res = smote.fit_resample(X_train_2d, Y_train_labels)
    
    print("✅ Balanceo con SMOTE realizado:")
    print(f"Forma de X_train después de SMOTE: {X_res.shape}")
    print(f"Forma de Y_train después de SMOTE: {y_res.shape}")

    # Si quieres mantener las etiquetas en one-hot, conviértelas de nuevo:
    enc = OneHotEncoder(sparse_output=False)
    y_res_oh = enc.fit_transform(y_res.reshape(-1, 1))

    # Volver a dar forma 3D a X_res si originalmente tenía 3D (opcional)
    if len(X_train.shape) > 2:
        X_res = X_res.reshape(X_res.shape[0], X_train.shape[1], X_train.shape[2])

    # Guardar datos balanceados
    output_path = os.path.join(data_path, "train_balanced.pkl")
    joblib.dump((X_res, y_res_oh, feature_names), output_path)
    print(f"📦 Datos balanceados guardados en: {output_path}")
    
    return X_res, y_res_oh
