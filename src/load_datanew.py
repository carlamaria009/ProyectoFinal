# PROGRAMA QUE CONTIENE LAS FUNCIONES QUE PERMITE:
#1. CARGAR LOS 4 DATASETS
#2. OBTENER LOS FEATURES (CARACTERISTICAS DE LOS AUDIOS)
# AQUI SE MUESTRA UNA ESTRUCTURA DE LAS LLAMADAS
#run_pipeline FUNCION PRINCIPAL
#│
#├── load_ravdess_dataset (si se proporciona ravdess_path)  CARGA DATASET
#├── load_crema_dataset (si se proporciona crema_path)      CARGA DATASET
#├── load_tess_dataset (si se proporciona tess_path)        CARGA DATASET
#├── load_savee_dataset (si se proporciona savee_path)      CARGA DATASET
#│
#├── explore_data GRAFICO DE BARRAS 
#│
#├── process_dataset
#│   ├── get_features (por cada fila del DataFrame)
#│   │   └── extract_features
#│   └── get_feature_names
#└── encode_labels APLICA ONEHOT A CATEGORICAS Y ALMACENA NOMBRES


#importamos librerias requeridas
import os
import pandas as pd
import numpy as np
import librosa
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder

#cargamos los datos de la fuente de RAVDESS. Recibe como parámetro la ruta donde se encuentran los archivos.
def load_ravdess_dataset(ravdess_path):
    # Listar directorios
    ravdess_directory_list = os.listdir(ravdess_path)

    file_emotion = []
    file_path = []

    for dir in ravdess_directory_list:
        actor = os.listdir(os.path.join(ravdess_path, dir))
        for file in actor:
            part = file.split('.')[0].split('-')
            file_emotion.append(int(part[2]))
            file_path.append(os.path.join(ravdess_path, dir, file))

    # Crear DataFrame
    emotion_df = pd.DataFrame(file_emotion, columns=['Emotions'])
    path_df = pd.DataFrame(file_path, columns=['Path'])
    df = pd.concat([emotion_df, path_df], axis=1)

    # Mapear emociones numéricas a etiquetas
    df.replace({'Emotions': {
        1: 'neutral', 2: 'neutral',  # Combinar neutral (1) y calma (2)
        3: 'felicidad', 4: 'triste',
        5: 'enojado', 6: 'miedo', 7: 'desagrado', 8: 'sorprendido'
    }}, inplace=True)

    # Eliminar la emoción "sorprendido"
    df = df[df['Emotions'] != 'sorprendido']

    # Resetear índices después del filtrado
    df.reset_index(drop=True, inplace=True)

    # Estadísticas
    total = len(df)
    emociones_unicas = df['Emotions'].nunique()
    lista_emociones = df['Emotions'].unique().tolist()

    return df, total, emociones_unicas, lista_emociones

#cargamos los datos de la fuente de CREMA. Recibe como parámetro la ruta donde se encuentran los archivos.
def load_crema_dataset(crema_path):
    crema_directory_list = os.listdir(crema_path)
    file_emotion = []
    file_path = []

    for file in crema_directory_list:
        # storing file paths
        file_path.append(crema_path + file)
        # storing file emotions
        part=file.split('_')
        if part[2] == 'SAD':
            file_emotion.append('triste')
        elif part[2] == 'ANG':
            file_emotion.append('enojado')
        elif part[2] == 'DIS':
            file_emotion.append('desagrado')
        elif part[2] == 'FEA':
            file_emotion.append('miedo')
        elif part[2] == 'HAP':
            file_emotion.append('felicidad')
        elif part[2] == 'NEU':
            file_emotion.append('neutral')
        else:
            file_emotion.append('Unknown')

    # dataframe for emotion of files
    emotion_df = pd.DataFrame(file_emotion, columns=['Emotions'])

    # dataframe for path of files.
    path_df = pd.DataFrame(file_path, columns=['Path'])
    df = pd.concat([emotion_df, path_df], axis=1)

    total = len(df)
    emociones_unicas = df['Emotions'].nunique()
    lista_emociones = df['Emotions'].unique().tolist()
    return df, total, emociones_unicas, lista_emociones

#cargamos los datos de la fuente de TESS. Recibe como parámetro la ruta donde se encuentran los archivos.
def load_tess_dataset(tess_path):
    tess_directory_list = os.listdir(tess_path)

    file_emotion = []
    file_path = []

    for dir in tess_directory_list:
        directories = os.listdir(os.path.join(tess_path, dir))
        for file in directories:
            part = file.split('.')[0].split('_')[2]
            file_path.append(os.path.join(tess_path, dir, file))
            file_emotion.append(part)

    # Traducir emociones de inglés a español
    traduccion_emociones = {
        'fear': 'miedo',
        'angry': 'enojado',
        'disgust': 'desagrado',
        'neutral': 'neutral',
        'sad': 'triste',
        'ps': 'sorprendido',  # ps = pleasant surprise
        'happy': 'felicidad'
    }

    emociones_traducidas = [traduccion_emociones.get(e, e) for e in file_emotion]

    # Crear DataFrames
    emotion_df = pd.DataFrame(emociones_traducidas, columns=['Emotions'])
    path_df = pd.DataFrame(file_path, columns=['Path'])
    df = pd.concat([emotion_df, path_df], axis=1)
    
    # Eliminar registros con emoción "sorprendido"
    df = df[df['Emotions'] != 'sorprendido']
    df.reset_index(drop=True, inplace=True)
    
    # Datos resumidos
    total = len(df)
    emociones_unicas = df['Emotions'].nunique()
    lista_emociones = df['Emotions'].unique().tolist()

    return df, total, emociones_unicas, lista_emociones

#cargamos los datos de la fuente de SAVEE. Recibe como parámetro la ruta donde se encuentran los archivos.
def load_savee_dataset(savee_path):
    savee_directory_list = os.listdir(savee_path)

    file_emotion = []
    file_path = []

    for file in savee_directory_list:
        file_path.append(savee_path + file)
        part = file.split('_')[1]
        ele = part[:-6]
        if ele=='a':
            file_emotion.append('enojado')
        elif ele=='d':
            file_emotion.append('desagrado')
        elif ele=='f':
            file_emotion.append('miedo')
        elif ele=='h':
            file_emotion.append('felicidad')
        elif ele=='n':
            file_emotion.append('neutral')
        elif ele=='sa':
            file_emotion.append('triste')
        else:
            file_emotion.append('sorprendido')

    # dataframe for emotion of files
    emotion_df = pd.DataFrame(file_emotion, columns=['Emotions'])

    # dataframe for path of files.
    path_df = pd.DataFrame(file_path, columns=['Path'])
    df = pd.concat([emotion_df, path_df], axis=1)
    
    # Eliminar registros con emoción "sorprendido"
    df = df[df['Emotions'] != 'sorprendido']
    df.reset_index(drop=True, inplace=True)
    
    total = len(df)
    emociones_unicas = df['Emotions'].nunique()
    lista_emociones = df['Emotions'].unique().tolist()
    return df, total, emociones_unicas, lista_emociones

#función que explora los dato. Crea un gráfico de barras con las emociones identificadas
def explore_data(df):
    df.to_csv("src/data_path.csv", index=False)
    #print("DF en explore_data")
    #print(df.head())

    plt.figure(figsize=(10, 6))
    plt.title('Conteo de Emociones', size=16)
    sns.countplot(data=df, x='Emotions', order=df['Emotions'].value_counts().index)
    plt.ylabel('Conteo', size=12)
    plt.xlabel('Emociones', size=12)
    sns.despine()
    plt.show()

#Extrae los features de los audios. Es decir extrae las características de los audios 
def extract_features(data, sample_rate):
    result = np.array([])

    #Extrae 2 características: media y desviación estándar de esa única banda, total 2 características
    zcr = librosa.feature.zero_crossing_rate(y=data)
    result = np.hstack((result, np.mean(zcr), np.std(zcr)))

    #Extrae 12 medias y 12 desviaciones estándar, total 24 características
    stft = np.abs(librosa.stft(data))
    chroma = librosa.feature.chroma_stft(S=stft, sr=sample_rate)
    result = np.hstack((result, np.mean(chroma, axis=1), np.std(chroma, axis=1)))

    #Extrae 40 medias y 40 desviaciones estándar, total 80 características
    mfccs = librosa.feature.mfcc(y=data, sr=sample_rate, n_mfcc=40)
    result = np.hstack((result, np.mean(mfccs, axis=1), np.std(mfccs, axis=1)))

    #Extraes 2 características: media y desviación estándar, total 2 características
    rms = librosa.feature.rms(y=data)
    result = np.hstack((result, np.mean(rms), np.std(rms)))

    #Extraes 128 medias y 128 desviaciones estándar, total 256 características
    mel = librosa.feature.melspectrogram(y=data, sr=sample_rate)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    result = np.hstack((result, np.mean(mel_db, axis=1), np.std(mel_db, axis=1)))

    return result

# Esta función define los nombres de las características con el propósito de identificarlas
# y posteriormente analizar las que tienen mayor importancia
def get_feature_names():
    names = []

    # Zero Crossing Rate
    names += ["zcr_mean", "zcr_std"]

    # Chroma STFT (12 coeficientes)
    chroma_names = [f"chroma_{i}_mean" for i in range(12)] + [f"chroma_{i}_std" for i in range(12)]
    names += chroma_names

    # MFCC (40 coeficientes)
    mfcc_names = [f"mfcc_{i}_mean" for i in range(40)] + [f"mfcc_{i}_std" for i in range(40)]
    names += mfcc_names

    # RMS
    names += ["rms_mean", "rms_std"]

    # Mel Spectrogram (128 coeficientes)
    mel_names = [f"mel_{i}_mean" for i in range(128)] + [f"mel_{i}_std" for i in range(128)]
    names += mel_names

    return names

#Carga 2.5 segundos de audio a partir del segundo 0.6 desde un archivo. Esto con el objetivo de estandarizar
#Extrae características del audio usando la función extract_features definida anteriormente
#Devuelve el vector de características.·
import librosa
import numpy as np

def get_features(path, n_mels=128, n_fft=2048, hop_length=512, duration=2.5, offset=0.6):
    try:
        # 1. Cargar el audio
        y, sr = librosa.load(path, duration=duration, offset=offset)
        
        # 2. Obtener el espectrograma tipo imagen
        mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels, n_fft=n_fft, hop_length=hop_length)

        # 3. Escala logarítmica (dB)
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

        # 4. Normalización a rango 0-1
        mel_spec_db = (mel_spec_db - mel_spec_db.min()) / (mel_spec_db.max() - mel_spec_db.min())

        # 5. Asegurarnos de tener tamaño fijo (128 x 128)
        if mel_spec_db.shape[1] < 128:
            pad_width = 128 - mel_spec_db.shape[1]
            mel_spec_db = np.pad(mel_spec_db, pad_width=((0,0), (0, pad_width)), mode='constant')
        else:
            mel_spec_db = mel_spec_db[:, :128]  # recortar si es más grande

        # 6. Expandir dimensiones para CNN 2D: (128, 128, 1)
        mel_spec_db = np.expand_dims(mel_spec_db, axis=-1)

        return mel_spec_db
    
    except Exception as e:
        print(f"❌ Error en {path}: {e}")
        return np.zeros((128, 128, 1))  # fallback por si hay error


#Función que hace la llamada a las funciones que obtienen los features y los nombres de los features.
def process_dataset(df):
    X, Y = [], []
    for path, emotion in zip(df.Path, df.Emotions):
        feature = get_features(path)
        X.append(feature)
        Y.append(emotion)

    # ✅ Obtener nombres de las columnas
    feature_names = get_feature_names()

    # Crear DataFrame con nombres reales
    features_df = pd.DataFrame(X, columns=feature_names)
    features_df['Emotions'] = Y

    # Guardar a CSV
    features_df.to_csv('src/features.csv', index=False)

    return features_df

# convierte etiquetas categóricas (como emociones) a formato one-hot, que es el formato que suelen requerir 
# los modelos de machine learning y almacena las categorías en el archivo class_labels.npy
def encode_labels(labels, save_path=None):
    encoder = OneHotEncoder()
    Y_encoded = encoder.fit_transform(np.array(labels).reshape(-1, 1)).toarray()
    # Guardar las clases si se proporciona una ruta
    if save_path is not None:
        os.makedirs(save_path, exist_ok=True)
        np.save(os.path.join(save_path, "class_labels.npy"), encoder.categories_[0])
        print(f"✅ Clases guardadas en: {os.path.join(save_path, 'class_labels.npy')}")

    return Y_encoded, encoder

#función principal que hace la llamada a las funciones que cargan los dataserts
#y a las funciones de procesamiento de los audios
def run_pipeline(ravdess_path=None, crema_path=None, tess_path=None, savee_path=None):
    dfs = []

    if ravdess_path:
        df_ravdess, total_r, emociones_r, lista_emociones_r = load_ravdess_dataset(ravdess_path)
        print(f"[RAVDESS] Total de registros: {total_r}")
        print(f"[RAVDESS] Total de emociones únicas: {emociones_r}")
        print(f"[RAVDESS] Emociones presentes: {lista_emociones_r}")
        dfs.append(df_ravdess)
    
    if crema_path:
        df_crema, total_r, emociones_r, lista_emociones_r = load_crema_dataset(crema_path)
        print(f"[CREMA] Total de registros: {total_r}")
        print(f"[CREMA] Total de emociones únicas: {emociones_r}")
        print(f"[CREMA] Emociones presentes: {lista_emociones_r}")
        dfs.append(df_crema)

    if tess_path:
        df_tess, total_r, emociones_r, lista_emociones_r = load_tess_dataset(tess_path)
        print(f"[TESS] Total de registros: {total_r}")
        print(f"[TESS] Total de emociones únicas: {emociones_r}")
        print(f"[TESS] Emociones presentes: {lista_emociones_r}")
        dfs.append(df_tess)
    
    if savee_path:
        df_savee, total_r, emociones_r, lista_emociones_r = load_savee_dataset(savee_path)
        print(f"[SAVEE] Total de registros: {total_r}")
        print(f"[SAVEE] Total de emociones únicas: {emociones_r}")
        print(f"[SAVEE] Emociones presentes: {lista_emociones_r}")
        dfs.append(df_savee)

    # Combinar todos los datasets en uno solo
    full_df = pd.concat(dfs, ignore_index=True)
    
    print("Exploración de datos")
    explore_data(full_df)

    features = process_dataset(full_df)
    
    #print("Exploración de features")
    #print(features.head())
    
    X = features.iloc[:, :-1]
    Y, _ = encode_labels(features['Emotions'], 'src/')
    
    return X, Y
