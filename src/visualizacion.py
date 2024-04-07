#%%

# visualizacion.py
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def scatterplot_vuelos_por_mes_anio(df):
    # Creamos un DataFrame que suma la cantidad de vuelos reservados por mes y año
    vuelos_anio_mes = df.groupby(['year', 'month'])['flights_booked'].sum().reset_index(name="flights_booked")
    sns.scatterplot(x="month", y="flights_booked", data=vuelos_anio_mes, hue="year", palette="pastel")
    
    # Scatter plot para visualizar la distribución de vuelos reservados por mes y año
    plt.title("Distribución vuelos reservados por mes")
    plt.xlabel("Mes del año")
    plt.ylabel("Cantidad de vuelos reservados")
    plt.show()

def boxplot_relacion_distancia_puntos(df):
    # Creamos un DataFrame que suma los puntos acumulados por distancia de vuelos
    df_puntos = df.groupby("distance")["points_accumulated"].sum().reset_index(name= "Puntos totales")
    
    # Box plot para visualizar la relación entre distancia de vuelos y puntos acumulados
    sns.boxplot(y = "distance", data = df_puntos, width = 0.5, color = "turquoise")
    plt.xlabel("Distancia entre vuelos")
    plt.ylabel("Puntos acumulados por los clientes")
    plt.title("Relación entre distancia vuelos y puntos acumulados")
    plt.show()

def violinplot_distancia_puntos(df):
    # Creamos un DataFrame que agrupa distancia vuelos y puntos acumulados 
    df_puntos = df.groupby("distance")["points_accumulated"].sum().reset_index(name= "Puntos totales")
    
    # Violin plot para visualizar la distribución de puntos acumulados por distancia de vuelos
    sns.violinplot(y = "distance", data = df_puntos, width = 0.5, color = "turquoise", linewidth = 2)
    plt.xlabel("Distancia entre vuelos")
    plt.ylabel("Puntos acumulados por los clientes")
    plt.title("Relación entre distancia vuelos y puntos acumulados")
    plt.show()
    
def bar_clientes_provincia(df):   
    
    # Creamos un DataFrame que cuenta el número de clientes por provincia
    df_clientes_provincia = df.groupby("province")["province"].count().reset_index(name="num_clientes")
    df_clientes_provincia = df_clientes_provincia.sort_values(by="num_clientes", ascending=False)

    # Gráfico de barras para visualizar la distribución de clientes por provincia
    plt.figure(figsize=(10, 6))
    plt.bar(df_clientes_provincia['province'], df_clientes_provincia['num_clientes'], color='coral')
    plt.xlabel('Provincia')
    plt.ylabel('Número clientes')
    plt.title('Distribución clientes por provincia')
    plt.xticks(rotation=45, ha='right')
    plt.show()

def barplot_educacion_salary(df):
    # Creamos un DataFrame que calcula el salario promedio por nivel educativo
    df_educacion = df.groupby("education")["salary"].mean().reset_index()

    # Gráfico de barras para visualizar el salario promedio por nivel educativo
    sns.barplot(x = "education", y = "salary", data = df_educacion, palette = "pink")
    plt.xlabel("Educación")
    plt.ylabel("Salario promedio")
    plt.title("Salario promedio entre los diferentes niveles educativos")
    plt.xticks(rotation=45, ha='right')
    plt.show()

def fidelidad_clientes(df):
    # Creamos un DataFrame que cuenta el número de clientes por tipo de tarjeta de fidelidad
    df_fidelidad = df["loyalty_card"].value_counts().to_frame().reset_index()

    # Gráfico de pastel para visualizar la distribución de tipos de tarjeta de fidelidad entre los clientes
    colores = ["deeppink", "purple", "plum"]
    plt.pie("count", labels= "loyalty_card", data = df_fidelidad, autopct=  '%1.1f%%', colors = colores, textprops={'fontsize': 8}, startangle=90)
    plt.show()

def gender_marital(df):
    # Creamos un DataFrame que cuenta el número de clientes por género y estado civil
    df_estado = df.groupby(["marital_status", "gender"]).size().reset_index(name="count")

    # Gráfico de barras para visualizar la distribución de clientes por género y estado civil
    sns.barplot(x = "gender", y = "count", hue = "marital_status", data = df_estado, palette = "pastel")
    plt.xticks(rotation = 90)
    plt.xlabel("Género")
    plt.ylabel("Cantidad de clientes")
    plt.title("Distribución de clientes por género y estado civil")
    plt.tight_layout()
    plt.show()

    plt.show()

# %%
