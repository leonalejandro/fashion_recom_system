import pandas as pd
import numpy as np
import ast
import os
import requests 

# Función para extraer la URL de "large" del primer elemento
def extract_first_large(image_data):
    """
    Extrae la URL de la primera imagen en tamaño "large" de una lista de imágenes.

    Parámetros:
    image_data (str o list): Puede ser una lista de diccionarios con información de imágenes 
                             o una cadena de texto que representa una lista en formato JSON.

    Retorna:
    str o None: La URL de la primera imagen en tamaño "large" si está disponible, 
                o None en caso de error o si no se encuentra la clave.

    Excepciones manejadas:
    - ValueError, SyntaxError: Si la conversión desde string a lista falla.
    """
    try:
        # Convertir el string a lista de diccionarios (si es necesario)
        images = ast.literal_eval(image_data) if isinstance(image_data, str) else image_data
        if isinstance(images, list) and images:
            return images[0].get("large", None)  # Obtener la URL de "large"
    except (ValueError, SyntaxError):
        return None  # Si hay error en la conversión, devolver None
    return None

def sample_top_rated_products(df: pd.DataFrame, n: int) -> pd.DataFrame:
    """
    Devuelve una muestra de 'n' productos priorizando aquellos con muchas calificaciones
    y asegurando variedad en la variable "average_rating".
    
    Parámetros:
    df (pd.DataFrame): DataFrame con columnas "rating_count" y "average_rating".
    n (int): Número de productos a muestrear.
    
    Retorna:
    pd.DataFrame: DataFrame con la muestra seleccionada.
    """
    # Ordenar por cantidad de calificaciones de mayor a menor
    df_sorted = df.sort_values(by="rating_count", ascending=False)
    
    # Crear intervalos de average_rating
    df_sorted["rating_bin"] = pd.cut(df_sorted["average_rating"], bins=np.linspace(0, 5, 6))
    
    # Calcular cantidad de muestras por grupo
    num_bins = df_sorted["rating_bin"].nunique()
    per_bin_sample = max(n // num_bins, 1)  # Asegurar al menos 1 por bin
    
    # Muestreo estratificado
    sampled_df = df_sorted.groupby("rating_bin",observed=False).apply(lambda x: x.head(per_bin_sample)).reset_index(drop=True)
    
    # Si hay menos de 'n' productos, rellenar con los más calificados restantes
    if len(sampled_df) < n:
        remaining = df_sorted[~df_sorted.index.isin(sampled_df.index)]
        extra_sample = remaining.head(n - len(sampled_df))
        sampled_df = pd.concat([sampled_df, extra_sample])
    
    # Tomar exactamente 'n' elementos aleatorios para balance final
    sampled_df = sampled_df.sample(n=n, random_state=42)
    
    # Eliminar la columna auxiliar
    sampled_df = sampled_df.drop(columns=["rating_bin"], errors="ignore")
    
    return sampled_df

def download_images(df, save_path):
    """
    Descarga imágenes de la columna 'image_1' y las guarda con el nombre de 'parent_product_id'.
    
    Parámetros:
    - df: DataFrame de pandas que contiene las columnas 'image_1' y 'parent_product_id'.
    - save_path: Ruta donde se guardarán las imágenes.
    
    Si la imagen no se puede descargar, simplemente se ignora.
    """
    # Crear el directorio si no existe
    os.makedirs(save_path, exist_ok=True)

    for _, row in df.iterrows():
        url = row["image_1"]
        filename = f"{row['parent_product_id']}.jpg"
        filepath = os.path.join(save_path, filename)

        if not url or pd.isna(url):  # Si no hay URL, saltar
            continue
        
        try:
            response = requests.get(url, timeout=5) 
            response.raise_for_status()  # Lanza un error si la descarga falla

            with open(filepath, "wb") as f:
                f.write(response.content)
            
            print(f"Imagen guardada: {filename}")

        except requests.RequestException:
            print(f"No se pudo descargar la imagen: {url}")
            continue

    print("Descarga de imágenes completada.")


def filter_dataframe_by_images(image_folder, df, column_name="parent_product_id"):
    """
    Filtra un DataFrame manteniendo solo los registros donde 'column_name'
    coincide con los nombres de archivos en 'image_folder' (sin extensión).

    Parámetros:
    - image_folder: Ruta de la carpeta con imágenes.
    - df: DataFrame a filtrar.
    - column_name: Nombre de la columna en el DataFrame que debe coincidir con los nombres de archivo.

    Retorna:
    - Un nuevo DataFrame con los registros filtrados.
    """
    # Obtener la lista de nombres de archivos sin extensión
    image_names = {os.path.splitext(f)[0] for f in os.listdir(image_folder) if os.path.isfile(os.path.join(image_folder, f))}
    
    # Filtrar el DataFrame
    filtered_df = df[df[column_name].isin(image_names)].copy()

    return filtered_df