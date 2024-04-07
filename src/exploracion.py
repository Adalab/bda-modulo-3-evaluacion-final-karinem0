
#%%
# exploracion.py
import pandas as pd
import numpy as np 
from sklearn.impute import SimpleImputer

def cargar_dataframes(ruta_csv1, ruta_csv2):
    """Carga los dataframes desde archivos CSV.

    Args:
    - ruta_flight (str): La ruta al archivo CSV de Customer Flight Activity.
    - ruta_loyalty (str): La ruta al archivo CSV de Customer Loyalty History.

    Returns:
    - df_flight (DataFrame): El dataframe cargado desde el archivo CSV de Customer Flight Activity.
    - df_loyalty (DataFrame): El dataframe cargado desde el archivo CSV de Customer Loyalty History.
    """
    df_flight = pd.read_csv(ruta_csv1, index_col=0)
    df_loyalty = pd.read_csv(ruta_csv2, index_col=0)
    return df_flight, df_loyalty


def combinar_dataframes(df_flight, df_loyalty):
    """Combina los dataframes df_flight y df_loyalty utilizando la función merge.

    Args:
    - df_flight (DataFrame): El dataframe de Customer Flight Activity.
    - df_loyalty (DataFrame): El dataframe de Customer Loyalty History.

    Returns:
    - df(DataFrame): El dataframe resultante de la combinación de df_flight y df_loyalty.
    """
    df_merge = pd.merge(df_flight, df_loyalty, left_index=True, right_index=True)
    return df_merge


def configurar_visualizacion():
    """Configura la visualización de pandas para mostrar todas las columnas y formato de los números."""
    pd.set_option('display.max_columns', None)
    pd.set_option('display.float_format', '{:.2f}'.format)


def exploracion_dataframe(dataframe, columna_control):
    """Realiza un análisis exploratorio básico de un DataFrame.

    Args:
    - df_merge(DataFrame): El DataFrame que se va a explorar.
    - columna_control (str): El nombre de la columna que se utilizará como control para dividir el DataFrame.
    """
    print(f"Los duplicados que tenemos en el conjunto de datos son: {dataframe.duplicated().sum()}")
    print("\n ..................... \n")
    
    # generamos un DataFrame para los valores nulos
    print("Los nulos que tenemos en el conjunto de datos son:")
    df_nulos = pd.DataFrame(dataframe.isnull().sum() / dataframe.shape[0] * 100, columns = ["%_nulos"])
    display(df_nulos[df_nulos["%_nulos"] > 0])
    
    print("\n ..................... \n")
    print(f"Los tipos de las columnas son:")
    display(pd.DataFrame(dataframe.dtypes, columns = ["tipo_dato"]))
    
    
    print("\n ..................... \n")
    print("Los valores que tenemos para las columnas categóricas son: ")
    dataframe_categoricas = dataframe.select_dtypes(include = "O")
    
    for col in dataframe_categoricas.columns:
        print(f"La columna {col.upper()} tiene las siguientes valore únicos:")
        display(pd.DataFrame(dataframe[col].value_counts()).head())    
    
    # Sacamos los principales estadísticos del dataframe 
    
    for categoria in dataframe[columna_control].unique():
        
        dataframe_filtrado = dataframe[dataframe[columna_control] == categoria]
    
        print("\n ..................... \n")
        print(f"Los principales estadísticos de las columnas categóricas para el {categoria.upper()} son: ")
        display(dataframe_filtrado.describe(include = "O").T)
        
        print("\n ..................... \n")
        print(f"Los principales estadísticos de las columnas numéricas para el {categoria.upper()} son: ")
        display(dataframe_filtrado.describe().T)
        
    return dataframe


def imputar_valores_nulos(dataframe, columna, estrategia='median'):
    """Imputa valores nulos en una columna del DataFrame utilizando SimpleImputer.

    Args:
    - dataframe (DataFrame): El DataFrame en el que se imputarán los valores nulos.
    - columna (str): El nombre de la columna en la que se imputarán los valores nulos.
    - estrategia (str, optional): La estrategia a utilizar para la imputación. Por defecto es 'median'.
    """
    
    imputer = SimpleImputer(strategy=estrategia)
    columna_imputada = imputer.fit_transform(dataframe[[columna]])
    dataframe[columna] = columna_imputada

def eliminar_columnas(dataframe, columnas):
    """Elimina columnas del DataFrame.

    Args:
    - dataframe (DataFrame): El DataFrame del que se eliminarán las columnas.
    - columnas (list): La lista de nombres de las columnas que se eliminarán.
    """
    dataframe.drop(columns=columnas, inplace=True)

def transformar_nombres_columnas(dataframe):
    """Transforma los nombres de las columnas del DataFrame.

    Args:
    - dataframe (DataFrame): El DataFrame cuyos nombres de columnas se transformarán.
    """
    nuevas_columnas = [col.replace(" ", '_').lower() for col in dataframe.columns]
    dataframe.columns = nuevas_columnas

def transformar_salary(valor):
    """Transforma el valor de la columna 'Salary'.

    Args:
    - valor: El valor que se transformará.

    Returns:
    - int: El valor transformado.
    """
    if valor != np.nan:
    #if pd.notnull(valor):
        valor = str(valor).replace('-', '')  # Elimina los guiones
        valor = valor.split('.')[0]  # Obtiene la parte entera antes del punto
        return int(valor)
    else:
        return np.nan
    

def analisis_frecuencia_cancelaciones(dataframe, columnas):
    """Realiza un análisis de frecuencia de cancelaciones según meses y años.

    Args:
    - dataframe (DataFrame): El DataFrame en el que se realizará el análisis.
    - columnas (list): La lista de nombres de las columnas a analizar.
    """
    for columnas_analizar in columnas:
        for columna in columnas_analizar:
            frecuencia_valores = dataframe[columna].value_counts()
            print(f"La frecuencia de cancelaciones para la columna '{columna}' es:\n {frecuencia_valores}")

# %%
