# Realizamos todas las importaciones necesarias
import pandas as pd
import numpy as np 
from sklearn.impute import SimpleImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.impute import KNNImputer
import seaborn as sns
import matplotlib.pyplot as plt

## FASE 1: EXPLORACIÓN, TRANSFORMACIÓN Y LIMPIEZA DE DATOS 

# Configuración para visualizar todas las columnas y formato de los números
pd.set_option('display.max_columns', None)
pd.set_option('display.float_format', '{:.2f}'.format)

# Carga de los dataframes desde archivos CSV
df_flight = pd.read_csv("files/Customer Flight Activity.csv", index_col=0)
df_loyalty = pd.read_csv("files/Customer Loyalty History.csv", index_col=0)

# Visualización de las primeras 10 filas del dataframe df_flight
df_flight.head(10)

# Combinación de los dataframes df_flight y df_loyalty utilizando la función merge
df_merge = pd.merge(df_flight, df_loyalty, left_index=True, right_index=True)

# Visualización de las primeras 10 filas del dataframe combinado
df_merge.head(10)

# Dimensiones de los dataframes antes y después de la unión
print(f"En el dataframe 'flights', el número de filas es {df_flight.shape[0]} y el número de columnas es {df_flight.shape[1]}")
print(f"En el dataframe 'loyalty', el número de filas es {df_loyalty.shape[0]} y el número de columnas es {df_loyalty.shape[1]}")
print(f"En el dataframe 'mergeado', el número de filas es {df_merge.shape[0]} y el número de columnas es {df_merge.shape[1]}")

# Función para realizar un análisis exploratorio del dataframe
def exploracion_dataframe(dataframe, columna_control):
    """
    Realiza un análisis exploratorio básico de un DataFrame, mostrando información sobre duplicados,
    valores nulos, tipos de datos, valores únicos para columnas categóricas y estadísticas descriptivas
    para columnas categóricas y numéricas, agrupadas por la columna de control.

    Parámetros:
    - dataframe (DataFrame): El DataFrame que se va a explorar.
    - columna_control (str): El nombre de la columna que se utilizará como control para dividir el DataFrame.

    Returns: 
    No devuelve nada directamente, pero imprime en la consola la información exploratoria.
    """
    
    # Duplicados en el conjunto de datos
    print(f"Los duplicados que tenemos en el conjunto de datos son: {dataframe.duplicated().sum()}")
    print("\n ..................... \n")
    
    # Valores nulos en el conjunto de datos
    print("Los nulos que tenemos en el conjunto de datos son:")
    df_nulos = pd.DataFrame(dataframe.isnull().sum() / dataframe.shape[0] * 100, columns = ["%_nulos"])
    display(df_nulos[df_nulos["%_nulos"] > 0])
    
    print("\n ..................... \n")
    
    # Tipos de las columnas en el conjunto de datos
    print(f"Los tipos de las columnas son:")
    display(pd.DataFrame(dataframe.dtypes, columns = ["tipo_dato"]))
    
    print("\n ..................... \n")
    
    # Valores únicos para las columnas categóricas en el conjunto de datos
    print("Los valores que tenemos para las columnas categóricas son: ")
    dataframe_categoricas = dataframe.select_dtypes(include = "O")
    
    for col in dataframe_categoricas.columns:
        print(f"La columna {col.upper()} tiene las siguientes valore únicos:")
        display(pd.DataFrame(dataframe[col].value_counts()).head())    
    
    # Estadísticas descriptivas para columnas categóricas y numéricas, agrupadas por la columna de control
    for categoria in dataframe[columna_control].unique():
        dataframe_filtrado = dataframe[dataframe[columna_control] == categoria]
    
        print("\n ..................... \n")
        print(f"Los principales estadísticos de las columnas categóricas para el {categoria.upper()} son: ")
        display(dataframe_filtrado.describe(include = "O").T)
        
        print("\n ..................... \n")
        print(f"Los principales estadísticos de las columnas numéricas para el {categoria.upper()} son: ")
        display(dataframe_filtrado.describe().T)
        
# Llamada a la función exploracion_dataframe con la columna de control "Education"
exploracion_dataframe(df_merge, "Education")

# Visualización de los nombres de las columnas del dataframe mergeado
df_merge.columns

# Selección de columnas categóricas y numéricas
col_categoricas = df_merge.select_dtypes(include=["object"])
print(f"Las columnas categóricas son:{list(col_categoricas)}")

col_numericas = df_merge.select_dtypes(include=["int","float"])
print(f"Las columnas categóricas son:{list(col_numericas)}")

# Análisis de frecuencia de cancelaciones según meses y años
columnas = ["Cancellation Month", "Cancellation Year"]

for columna in columnas:  
    valores_unicos = df_merge[columna].unique()
    frecuencia_valores = df_merge[columna].value_counts()
    print(f"La frecuencia de cancelaciones según los meses es:\n {frecuencia_valores}")
    print(f"La frecuencia de cancelaciones según los años es:\n {frecuencia_valores}")
    
# Creación de un DataFrame de valores nulos y su porcentaje
df_nulos = pd.DataFrame((df_merge.isnull().sum() / df_merge.shape[0]) * 100, columns = ["%_nulos"])
# Filtrado de columnas con valores nulos
df_nulos[df_nulos["%_nulos"] > 0]

# Imputación de valores nulos en la columna "Salary" usando la mediana como estrategia
imputer = SimpleImputer(strategy='median') 
salary_imput = imputer.fit_transform(df_merge[["Salary"]])
df_merge["Salary"] = salary_imput
print(f"Después del 'SimpleImputer' tenemos {df_merge['Salary'].isnull().sum()} nulos")

# Función para eliminar columnas del dataframe
def eliminacion_columnas(dataframe, columnas):
    dataframe.drop(columns=columnas, inplace=True)

# Función para transformar los datos del dataframe
def transformacion_datos(dataframe):
    nuevas_columnas = [col.replace(" ", '_').lower() for col in dataframe.columns]
    dataframe.columns = nuevas_columnas  

# Función para transformar la columna "salary"
def transformacion_salary(valor):
    if valor != np.nan:
        valor = str(valor).replace('-', '')  # Elimina los guiones
        valor = valor.split('.')[0]  # Obtiene la parte entera antes del punto
        return int(valor)
    else:
        return np.nan

# Llamadas a las funciones para aplicar los cambios
columnas_eliminar = ["Country", "Cancellation Month", "Cancellation Year"]
eliminacion_columnas(df_merge, columnas_eliminar)
transformacion_datos(df_merge)
df_merge["Salary"] = df_merge["Salary"].apply(transformacion_salary)
df_merge.info()


## FASE 2: VISUALIZACIÓN DE DATOS

# Creamos un DataFrame que suma la cantidad de vuelos reservados por mes y año
vuelos_anio_mes = df_merge.groupby(['year', 'month'])['flights_booked'].sum().reset_index(name="flights_booked")

# Scatter plot para visualizar la distribución de vuelos reservados por mes y año
sns.scatterplot(x="month", y="flights_booked", data=vuelos_anio_mes, hue="year", palette="pastel")
plt.title("Distribución vuelos reservados por mes")
plt.xlabel("Mes del año")
plt.ylabel("Cantidad de vuelos reservados")
plt.show()

# Creamos un DataFrame que suma los puntos acumulados por distancia de vuelos
df_puntos = df_merge.groupby("distance")["points_accumulated"].sum().reset_index(name= "Puntos totales")

# Box plot para visualizar la relación entre distancia de vuelos y puntos acumulados
sns.boxplot(y = "distance", data = df_puntos, width = 0.5, color = "turquoise")
plt.xlabel("Distancia entre vuelos")
plt.ylabel("Puntos acumulados por los clientes")
plt.title("Relación entre distancia vuelos y puntos acumulados")
plt.show()

# Violin plot para visualizar la distribución de puntos acumulados por distancia de vuelos
sns.violinplot(y = "distance", data = df_puntos, width = 0.5, color = "turquoise", linewidth = 2)
plt.xlabel("Distancia entre vuelos")
plt.ylabel("Puntos acumulados por los clientes")
plt.title("Relación entre distancia vuelos y puntos acumulados")
plt.show()

# Creamos un DataFrame que cuenta el número de clientes por provincia
df_clientes_provincia = df_merge.groupby("province")["province"].count().reset_index(name="num_clientes")
df_clientes_provincia = df_clientes_provincia.sort_values(by="num_clientes", ascending=False)

# Gráfico de barras para visualizar la distribución de clientes por provincia
plt.figure(figsize=(10, 6))
plt.bar(df_clientes_provincia['province'], df_clientes_provincia['num_clientes'], color='coral')
plt.xlabel('Provincia')
plt.ylabel('Número clientes')
plt.title('Distribución clientes por provincia')
plt.xticks(rotation=45, ha='right')
plt.show()

# Creamos un DataFrame que calcula el salario promedio por nivel educativo
df_educacion = df_merge.groupby("education")["salary"].mean().reset_index()

# Gráfico de barras para visualizar el salario promedio por nivel educativo
sns.barplot(x = "education", y = "salary", data = df_educacion, palette = "pink")
plt.xlabel("Educación")
plt.ylabel("Salario promedio")
plt.title("Salario promedio entre los diferentes niveles educativos")
plt.xticks(rotation=45, ha='right')
plt.show()

# Creamos un DataFrame que cuenta el número de clientes por tipo de tarjeta de fidelidad
df_fidelidad = df_merge["loyalty_card"].value_counts().to_frame().reset_index()

# Gráfico de pastel para visualizar la distribución de tipos de tarjeta de fidelidad entre los clientes
colores = ["deeppink", "purple", "plum"]
plt.pie("count", labels= "loyalty_card", data = df_fidelidad, autopct=  '%1.1f%%', colors = colores, textprops={'fontsize': 8}, startangle=90)
plt.show()

# Creamos un DataFrame que cuenta el número de clientes por género y estado civil
df_estado = df_merge.groupby(["marital_status", "gender"]).size().reset_index(name="count")

# Gráfico de barras para visualizar la distribución de clientes por género y estado civil
sns.barplot(x = "gender", y = "count", hue = "marital_status", data = df_estado, palette = "pastel")
plt.xticks(rotation = 90)
plt.xlabel("Género")
plt.ylabel("Cantidad de clientes")
plt.title("Distribución de clientes por género y estado civil")
plt.tight_layout()
plt.show()

plt.show()


## FASE 3: A/B TESTING 

# Abrimos el dataframe clasificado solo por las columnas de interés
df_education_flights = df_merge[['flights_booked', 'education']]
print(df_education_flights.head())

# Análisis Descriptivo
grupo_educativo = df_merge.groupby('education')['flights_booked']
print(grupo_educativo.head())

# Estadísticas descriptivas del grupo educativo
estadisticas_descriptivas = grupo_educativo.describe()
print(estadisticas_descriptivas)

# Función para calcular la tasa de conversión por nivel educativo
def tasa_conversion(df, nivel_estudio): 
    grupo = df[df["education"] == nivel_estudio]
    conversion_rate = grupo["flights_booked"].sum() / grupo["flights_booked"].count()
    return conversion_rate

grupos_educativos = ["Bachelor", "College", "Doctor", "Master", "High School or Below"]
for grupo in grupos_educativos:
    conversion_rate = tasa_conversion(df_merge, grupo)
    print(f"Tasa de conversión para el grupo educativo {grupo}: {conversion_rate}")

# Test ANOVA para diferentes grupos de educación
anova_resultado = f_oneway(df_merge[df_merge["education"] == "Bachelor"]["flights_booked"],
                            df_merge[df_merge["education"] == "College"]["flights_booked"],
                            df_merge[df_merge["education"] == "Doctor"]["flights_booked"],
                            df_merge[df_merge["education"] == "Master"]["flights_booked"],
                            df_merge[df_merge["education"] == "High School or Below"]["flights_booked"])

print("Estadístico F:", anova_resultado.statistic)
print("Valor p:", anova_resultado.pvalue)

alpha = 0.05
if anova_resultado.pvalue < alpha:
    print("Hay diferencias significativas en el número de vuelos reservados entre al menos dos grupos.")
    print("\n ---------- \n")
    print("""
          Los resultados sugieren que existe evidencia estadística para afirmar que las medias de las muestras son distintas. 
          Por lo tanto, nuestro nuevo sistema tiene los efectos deseados y deberíamos cambiar la nueva versión de anuncios   
          """)
else:
    print("No hay evidencia de diferencias significativas en el número de vuelos reservados entre los grupos.")
    print("\n ---------- \n")
    print(""" 
          Los resultados sugieren que no existe evidencia estadística para afirmar que las medias de las muestras son distintas,
          por lo que la nueva campaña no está ayudando a nuestro problema. 
          """)

# Gráfico de barras para visualizar los diferentes grupos y sus vuelos reservados
plt.figure(figsize=(10, 6))
sns.barplot(x="education", 
            y="flights_booked", 
            data=df_merge, 
            palette="muted")
plt.title("Vuelos Reservados por Grupo Educativo")
plt.xlabel("Nivel Educativo")
plt.ylabel("Vuelos Reservados")
plt.xticks(rotation=45)

# Tabla de contingencia y prueba de chi-cuadrado
contingency_table = pd.crosstab(df_merge['education'], df_merge['flights_booked'])
chi2, p, dof, expected = chi2_contingency(contingency_table)
print("Estadístico Chi-cuadrado:", chi2)
print("Valor p:", p)

# Correlación de rango de Kendall
kendall_corr, kendall_p = kendalltau(df_merge['flights_booked'], df_merge['education'])
print("Correlación de rango de Kendall:", kendall_corr)
print("Valor p de Kendall:", kendall_p)




