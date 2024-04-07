#%%
# main.py

# Importar funciones de los tres archivos:
from exploracion import (cargar_dataframes,
                         combinar_dataframes,
                         configurar_visualizacion,
                         exploracion_dataframe,
                         imputar_valores_nulos,
                         eliminar_columnas,
                         transformar_nombres_columnas,
                         transformar_salary,
                         analisis_frecuencia_cancelaciones)

# Importar funciones de visualización
from visualizacion import (
    scatterplot_vuelos_por_mes_anio,
    boxplot_relacion_distancia_puntos,
    violinplot_distancia_puntos,
    bar_clientes_provincia,
    barplot_educacion_salary,
    fidelidad_clientes,
    gender_marital)

# Importar función de ab_testing.py
from ab_testing import exploracion_ab

#%%


if __name__ == "__main__":
    
    # FASE 1: EDA (Exploratory Data Analysis)
    
    # Rutas de los archivos CSV
    ruta_flight = "data/Customer Flight Activity.csv"
    ruta_loyalty = "data/Customer Loyalty History.csv"

    # Cargar los dataframes desde los archivos CSV
    df_flight, df_loyalty = exploracion.cargar_dataframes(ruta_flight, ruta_loyalty)
    
    # Combinar los dataframes
    df_merge = exploracion.combinar_dataframes(df_flight, df_loyalty)

    # Configurar la visualización
    exploracion.configurar_visualizacion()
    
    # Realizar análisis exploratorio del dataframe combinado
    exploracion.exploracion_dataframe(df_merge, "Education")
    
    # Imputar valores nulos en la columna "Salary" utilizando la mediana
    exploracion.imputar_valores_nulos(df_merge, "Salary")
    
    # Analizar frecuencia de cancelaciones según los meses y los años
    exploracion.analisis_frecuencia_cancelaciones(df_merge, [["Cancellation Month", "Cancellation Year"]])
    
    # Eliminar columnas no necesarias
    columnas_eliminar = ["Country", "Cancellation Month", "Cancellation Year"]
    exploracion.eliminar_columnas(df_merge, columnas_eliminar)
    
    # Transformar nombres de columnas para tener un formato uniforme
    exploracion.transformar_nombres_columnas(df_merge)
    
    # Aplicar una transformación a la columna "Salary"
    df_merge["Salary"] = df_merge["Salary"].apply(exploracion.transformar_salary)

    # Imprimir la información del dataframe final
    print(df_merge.info())

#%%

    # FASE 2: VISUALIZACIÓN DE DATOS

    # Aquí debes cargar tu dataframe o importarlo desde donde se encuentre
    # por ejemplo, df = pd.read_csv("nombre_archivo.csv")

    # Llamadas a las funciones de visualización con el dataframe como argumento
    scatterplot_vuelos_por_mes_anio(df_merge)
    boxplot_relacion_distancia_puntos(df_merge)
    violinplot_distancia_puntos(df_merge)
    bar_clientes_provincia(df_merge)
    barplot_educacion_salary(df_merge)
    fidelidad_clientes(df_merge)
    gender_marital(df_merge)
    
    
    # FASE 3: A/B TESTING

    # Aquí debes cargar tu dataframe o importarlo desde donde se encuentre
    # por ejemplo, df = pd.read_csv("nombre_archivo.csv")

    # Llamada a la función de exploración A/B Testing con el dataframe como argumento
    exploracion_ab(df_merge)

