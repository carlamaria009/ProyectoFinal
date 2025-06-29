#PROGRAMA QUE HACE EL LA DIVISIÓN DEL DATASET EN TRAINING Y TEST
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd
import os
import joblib  # Para guardar los datos

#función que realiza la división. Recibe los valores de la matriz (X) y las categorías(Y) emociones
def prepare_datasets(X, Y, save_path="src/"):

    # Extraer nombres de columnas si existen
    if isinstance(X, pd.DataFrame):
        feature_names = X.columns.tolist()
        X = X.values  # convertimos a ndarray para el procesamiento
    else:
        feature_names = [f"feature_{i}" for i in range(X.shape[1])]

    # División de datos
    x_train, x_test, y_train, y_test = train_test_split(
        X, Y, test_size=0.2, random_state=0, stratify=Y
    )

    print(f"Tamaño del conjunto de entrenamiento: {x_train.shape[0]} muestras")
    print(f"Tamaño del conjunto de prueba: {x_test.shape[0]} muestras")

    # Escalado
    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)
    
    # Expandir dimensión para modelos que lo requieren
    x_train_scaled = np.expand_dims(x_train_scaled, axis=2)
    x_test_scaled = np.expand_dims(x_test_scaled, axis=2)

    # Guardar datos con nombres de features
    joblib.dump((x_train_scaled, y_train, feature_names), os.path.join(save_path, "train.pkl"))
    joblib.dump((x_test_scaled, y_test, feature_names), os.path.join(save_path, "test.pkl"))
    joblib.dump(scaler, os.path.join(save_path, "scaler.pkl"))

    print(f"✅ Datos guardados en {save_path}")

    # Mostrar primeras filas para ver que todo va bien
    x_train_flat = x_train_scaled.reshape(x_train_scaled.shape[0], -1)
    df_preview = pd.DataFrame(x_train_flat[:5], columns=feature_names)
    print("\n📋 Primeras 5 filas del set de entrenamiento:")
    print(df_preview)

    return x_train_scaled, x_test_scaled, y_train, y_test, feature_names

def prepare_datasets2(X, Y, save_path="src/"):
    import os
    import numpy as np
    import pandas as pd
    import joblib
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler

    # Obtener etiquetas crudas para estratificar (Y está en one-hot)
    Y_labels = np.argmax(Y, axis=1)

    # División en train (70%) y test+val (30%)
    x_train, x_temp, y_train, y_temp = train_test_split(
        X, Y, test_size=0.3, random_state=0, stratify=Y_labels
    )

    # División del conjunto temporal en validation (15%) y test (15%)
    y_temp_labels = np.argmax(y_temp, axis=1)
    x_val, x_test, y_val, y_test = train_test_split(
        x_temp, y_temp, test_size=0.5, random_state=0, stratify=y_temp_labels
    )

    print(f"Tamaño del conjunto de entrenamiento: {x_train.shape[0]} muestras")
    print(f"Tamaño del conjunto de validación: {x_val.shape[0]} muestras")
    print(f"Tamaño del conjunto de prueba: {x_test.shape[0]} muestras")

    # Aplanar para escalar (StandardScaler espera 2D)
    x_train_flat = x_train.reshape(x_train.shape[0], -1)
    x_val_flat = x_val.reshape(x_val.shape[0], -1)
    x_test_flat = x_test.reshape(x_test.shape[0], -1)

    # Escalar
    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train_flat)
    x_val_scaled = scaler.transform(x_val_flat)
    x_test_scaled = scaler.transform(x_test_flat)

    # Volver a forma 2D + canal para CNN2D
    x_train_scaled = x_train_scaled.reshape(-1, x_train.shape[1], x_train.shape[2], 1)
    x_val_scaled = x_val_scaled.reshape(-1, x_val.shape[1], x_val.shape[2], 1)
    x_test_scaled = x_test_scaled.reshape(-1, x_test.shape[1], x_test.shape[2], 1)

    # Expandir dimensión para modelos tipo CNN (ej. [batch, features, 1])
    #x_train_scaled = np.expand_dims(x_train_scaled, axis=2)
    #x_val_scaled = np.expand_dims(x_val_scaled, axis=2)
    #x_test_scaled = np.expand_dims(x_test_scaled, axis=2)

    # Generar nombres genéricos para las columnas
    num_features = x_train.shape[1] * x_train.shape[2]  # n_mels × time_frames
    feature_names = [f"mel_{i}" for i in range(num_features)]

    # Crear carpeta y guardar
    os.makedirs(save_path, exist_ok=True)
    joblib.dump((x_train_scaled, y_train, feature_names), os.path.join(save_path, "train.pkl"))
    joblib.dump((x_val_scaled, y_val, feature_names), os.path.join(save_path, "val.pkl"))
    joblib.dump((x_test_scaled, y_test, feature_names), os.path.join(save_path, "test.pkl"))
    joblib.dump(scaler, os.path.join(save_path, "scaler.pkl"))

    print(f"✅ Datos guardados en {save_path}")

    # Vista previa de los primeros 5 ejemplos (aplanados)
    df_preview = pd.DataFrame(x_train_scaled.reshape(x_train_scaled.shape[0], -1)[:5], columns=feature_names)
    print("\n📋 Primeras 5 filas del set de entrenamiento:")
    print(df_preview)

    return x_train_scaled, x_val_scaled, x_test_scaled, y_train, y_val, y_test, feature_names

#Función que verifica el dataset guardado en el pkl
def check_dataset(pkl_path):
    data = joblib.load(pkl_path)
    print(type(data))

def check_train(pkl_filename):
    import os
    import joblib
    import pandas as pd
    import numpy as np
    import seaborn as sns
    import matplotlib.pyplot as plt

    # Ruta a tu archivo train.pkl y class_labels.npy
    train_path = "src/"
    pkl_path = os.path.join(train_path, pkl_filename)
    class_names_path = os.path.join(train_path, "class_labels.npy")

    # Cargo los datos (X_train, y_train, feature_names)
    X_train, y_train, feature_names = joblib.load(pkl_path)

    # Cargo nombres de las clases
    class_names = np.load(class_names_path, allow_pickle=True)

    # Aplanar X_train si es multidimensional
    X_train_flat = X_train.reshape(X_train.shape[0], -1)

    # Crear DataFrame con características
    df_train = pd.DataFrame(X_train_flat, columns=feature_names)

    # Convertir y_train de one-hot a etiquetas numéricas
    if len(y_train.shape) > 1 and y_train.shape[1] > 1:
        y_labels = np.argmax(y_train, axis=1)
    else:
        y_labels = y_train

    # Añadir columna numérica de etiquetas
    df_train['Emotions_num'] = y_labels

    # Añadir columna con nombres de las emociones
    df_train['Emotions'] = df_train['Emotions_num'].apply(lambda x: class_names[x])

    print(df_train.head())

    plt.figure(figsize=(8,5))
    sns.countplot(data=df_train, x='Emotions',
                  order=df_train['Emotions'].value_counts().index)
    plt.title("Distribución de etiquetas en train.pkl")
    plt.xlabel("Emoción")
    plt.ylabel("Cantidad de muestras")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

    
