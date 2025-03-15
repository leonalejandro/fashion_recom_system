import tensorflow as tf
import numpy as np
import pandas as pd
import os
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
from transformers import BertTokenizer, TFBertModel
import constants
import re
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error, mean_squared_log_error

def extract_image_features(input_path, output_path):
    """
    Extrae características de imágenes usando EfficientNetB0 y guarda los vectores en un archivo CSV.

    Parámetros:
    - input_path: Carpeta donde están las imágenes.
    - output_path: Ruta del archivo CSV donde se guardan los vectores.
    """
    
    # Cargar el modelo preentrenado sin la última capa
    base_model = EfficientNetB0(weights=constants.IMAGENET_MODEL, include_top=False, pooling=constants.POOLING_AVG)
    
    # Lista para almacenar las características
    features_list = []

    # Recorrer las imágenes en la carpeta
    for img_name in os.listdir(input_path):
        img_path = os.path.join(input_path, img_name)
        
        try:
            # Cargar imagen y preprocesar
            img = image.load_img(img_path, target_size=(224, 224))
            img_array = image.img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0)
            img_array = preprocess_input(img_array)
            
            # Extraer características
            features = base_model.predict(img_array).flatten()
            
            # Agregar a la lista
            features_list.append([os.path.splitext(img_name)[0]] + features.tolist())
        
        except Exception as e:
            print(f"Error procesando {img_name}: {e}")
    
    # Crear un DataFrame con las características
    num_features = len(features_list[0]) - 1  # Restamos 1 por la columna de imagen
    columns = [constants.COL_IMAGE_NAME] + [f"feature_{i+1}" for i in range(num_features)]
    df = pd.DataFrame(features_list, columns=columns)

    # Guardar como CSV
    df.to_csv(output_path, index=False)

    print(f"Extracción completada. Archivo guardado en {output_path}")


def get_bert_embeddings(texts, tokenizer, bert_model, projection_layer):
    """Convierte una lista de textos en embeddings usando BERT y reduce la dimensionalidad."""
    tokens = tokenizer(texts, padding=True, truncation=True, return_tensors=constants.TENSOR)
    outputs = bert_model(**tokens)
    pooled_output = outputs.pooler_output  # Vector de 768 dimensiones
    reduced_embeddings = projection_layer(pooled_output)  # Reducir a 256 dimensiones
    return reduced_embeddings

def get_bert_embeddings_batch(texts, tokenizer, bert_model, projection_layer, batch_size=constants.N_16):
    """Obtiene los embeddings de BERT en lotes para evitar consumo excesivo de memoria."""
    embeddings_list = []
    
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        batch_embeddings = get_bert_embeddings(batch, tokenizer, bert_model, projection_layer)
        embeddings_list.append(batch_embeddings.numpy())  # Convertir a numpy para eficiencia
    
    return np.vstack(embeddings_list)  # Une los lotes en una sola matriz

def clean_text(text):
    """Limpia el texto eliminando caracteres especiales y espacios extras."""
    text = text.lower()  # Convertir a minúsculas
    text = re.sub(r'\s+', ' ', text)  # Eliminar espacios extra
    text = re.sub(r'[^a-z0-9.,!? ]', '', text)  # Dejar solo letras, números y puntuación básica
    return text.strip()

def to_numpy(x):
    if isinstance(x, tf.Tensor):
        return x.numpy()  # Si es un tensor, conviértelo
    return x  # Si ya es NumPy, déjalo así

def extract_review_features(input_path, output_path):
    # Cargar el tokenizador y el modelo de BERT
    tokenizer = BertTokenizer.from_pretrained(constants.BERT_MODEL)
    bert_model = TFBertModel.from_pretrained(constants.BERT_MODEL)

    # Capa densa para reducir dimensionalidad
    projection_layer = tf.keras.layers.Dense(constants.N_256, activation=constants.ACTIVATION_RELU)
    
    # Lectura del DF y ejecución del modelo.
    df_reviews = pd.read_csv(input_path).iloc[:constants.DF_REVIEWS_QTY]
    df_reviews[constants.COL_CLEAN_REVIEW] = df_reviews[constants.COL_REVIEW_CONTENT].fillna(constants.PARAM_N_A).astype(str).apply(clean_text)
    text_embeddings = get_bert_embeddings_batch(df_reviews[constants.COL_CLEAN_REVIEW].tolist(),tokenizer, bert_model, projection_layer,batch_size=constants.N_16)
    df_embeddings = pd.DataFrame(to_numpy(text_embeddings), columns=[f"dim_{i}" for i in range(256)])
    df_embeddings[constants.COL_CUSTOMER_ID] = df_reviews[constants.COL_CUSTOMER_ID].values
    df_embeddings[constants.COL_PARENT_PRODUCT_ID] = df_reviews[constants.COL_PARENT_PRODUCT_ID].values
    df_embeddings[constants.COL_REVIEW_SCORE] = df_reviews[constants.COL_REVIEW_SCORE].values

    df_embeddings.to_csv(output_path, index=False)
    return df_embeddings

def getDataForHybridModel(pathProducts, pathReviews):

    # Cargar embeddings de imágenes
    image_embeddings_df = pd.read_csv(pathProducts)

    # Extraer columnas de embeddings de imágenes
    image_embedding_columns = [col for col in image_embeddings_df.columns if col != constants.COL_IMAGE_NAME]
    image_embeddings_df[constants.COL_EMBEDDING] = image_embeddings_df[image_embedding_columns].apply(tuple, axis=1)

    # Cargar reseñas de reseñas
    reviews_df = pd.read_csv(pathReviews)

    # Codificar customer_id como un número único
    user_encoder = LabelEncoder()
    reviews_df[constants.COL_CUSTOMER_ID] = user_encoder.fit_transform(reviews_df[constants.COL_CUSTOMER_ID])
    
    # Extraer embeddings de reseñas desde las columnas dim_1, dim_2, etc.
    embedding_columns = [col for col in reviews_df.columns if col.startswith('dim_')]
    reviews_df[constants.COL_EMBEDDING] = reviews_df[embedding_columns].apply(tuple, axis=1)

    # Fusionar embeddings (review + imagen)
    X = []
    y = []

    for _, row in reviews_df.iterrows():
        product_id = str(row[constants.COL_PARENT_PRODUCT_ID])
        img_emb_row = image_embeddings_df[image_embeddings_df[constants.COL_IMAGE_NAME].str.startswith(product_id)]
        
        if not img_emb_row.empty:
          img_emb = np.array(img_emb_row[constants.COL_EMBEDDING].values[0])
          review_emb = np.array(row[constants.COL_EMBEDDING])
          user_emb = np.array([row[constants.COL_CUSTOMER_ID]])
          combined_emb = np.concatenate([review_emb, img_emb, user_emb])
          X.append(combined_emb)
          y.append(row[constants.COL_REVIEW_SCORE])
            
    X = np.array(X)
    y = np.array(y)

    # Normalizar datos
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # Particionar datos
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42) 

    return X_train, X_test, y_train, y_test


def generateModel(X_train, X_test, y_train, y_test):
  model = keras.Sequential([
      layers.Dense(128, activation=constants.ACTIVATION_RELU, input_shape=(X_train.shape[1],)),
      layers.Dense(64, activation=constants.ACTIVATION_RELU),
      layers.Dense(32, activation=constants.ACTIVATION_RELU),
      layers.Dense(1, activation='linear')  # Salida continua para predecir calificación
  ])
  model.compile(optimizer='adam', loss='mse', metrics=['mae'])

  # Entrenar el modelo
  model.fit(X_train, y_train, epochs=20, batch_size=32, validation_data=(X_test, y_test))

  # Evaluación del modelo
  y_pred = model.predict(X_test)
  r2 = r2_score(y_test, y_pred)
  mae = mean_absolute_error(y_test, y_pred)
  mse = mean_squared_error(y_test, y_pred)
  rmse = np.sqrt(mse)

  print(f'R2 Score en datos de prueba: {r2}')
  print(f'MAE (Error Absoluto Medio): {mae}')
  print(f'MSE (Error Cuadrático Medio): {mse}')
  print(f'RMSE (Raíz del Error Cuadrático Medio): {rmse}')

  model.save(constants.OUTPUT_PATH_MODEL)


def obtener_titulos_productos(csv_path, productos_lista):
    """
    Busca los títulos de productos en un archivo CSV basado en una lista de IDs.

    Parámetros:
    - csv_path: Ruta del archivo CSV que contiene 'parent_product_id' y 'product_title'.
    - productos_lista: Lista de tuplas con (ID del producto, puntuación).

    Retorna:
    - Lista de tuplas con (ID del producto, título, puntuación).
    """
    # Cargar el DataFrame desde el CSV
    df_products = pd.read_csv(csv_path)

    # Convertir el DataFrame en un diccionario para acceso rápido
    producto_dict = df_products.set_index(constants.COL_PARENT_PRODUCT_ID)[constants.COL_PRODUCT_TITLE].to_dict()

    # Crear la nueva lista con los títulos de los productos
    resultado = [(id_, producto_dict.get(id_, "Título no encontrado"), float(valor)) for id_, valor in productos_lista]

    return resultado


# Función para recomendar productos a un usuario
def recomendar_productos(user_id, reviews_path, products_path, model_path='model.keras', top_n=5):

    # Cargar el modelo
    model = load_model(model_path)
    
    # Cargar datasets
    reviews_df = pd.read_csv(reviews_path)
    image_embeddings_df = pd.read_csv(products_path)
    
    # Codificar customer_id
    user_encoder = LabelEncoder()
    reviews_df[constants.COL_CUSTOMER_ID] = user_encoder.fit_transform(reviews_df[constants.COL_CUSTOMER_ID])
    user_id_encoded = user_encoder.transform([user_id])[0]

    # Extraer columnas de embeddings
    embedding_columns = [col for col in reviews_df.columns if col.startswith('dim_')]
    image_embedding_columns = [col for col in image_embeddings_df.columns if col != constants.COL_IMAGE_NAME]
    
    # Convertir embeddings en tuplas
    reviews_df[constants.COL_EMBEDDING] = reviews_df[embedding_columns].apply(lambda x: tuple(x.values), axis=1)
    image_embeddings_df[constants.COL_EMBEDDING] = image_embeddings_df[image_embedding_columns].apply(lambda x: tuple(x.values), axis=1)
    
    # Filtrar reseñas del usuario
    user_reviews = reviews_df[reviews_df[constants.COL_CUSTOMER_ID] == user_id_encoded]
    user_reviewed_products = set(user_reviews[constants.COL_PARENT_PRODUCT_ID])
    recomendaciones = []

    for _, row in image_embeddings_df.iterrows():
        product_id = row[constants.COL_IMAGE_NAME].split('.')[0]  # Quitar extensión
        if product_id not in user_reviewed_products:
            img_emb = np.array(row[constants.COL_EMBEDDING], dtype=np.float32)
            if not user_reviews.empty:
                review_emb = np.mean(np.stack(user_reviews[constants.COL_EMBEDDING].apply(lambda x: np.array(x, dtype=np.float32))), axis=0)
            else:
                review_emb = np.zeros_like(img_emb)  # Si no hay reseñas, usar ceros
            user_emb = np.array([user_id_encoded], dtype=np.float32)
            combined_emb = np.concatenate([review_emb, img_emb, user_emb]).reshape(1, -1)
            predicted_rating = model.predict(combined_emb)[0][0]
            recomendaciones.append((product_id, predicted_rating))
    
    recomendaciones.sort(key=lambda x: x[1], reverse=True)
    return obtener_titulos_productos(constants.INPUT_PATH_PRODUCTS, recomendaciones[:top_n])