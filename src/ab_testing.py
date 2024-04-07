#%%

# ab_testing.py
import pandas as pd
from scipy.stats import f_oneway, chi2_contingency
from scipy.stats import kendalltau
import seaborn as sns
import matplotlib.pyplot as plt


def exploracion_ab(df):
    """
        Realiza un análisis A/B Testing que incluye:
        - Análisis descriptivo de vuelos reservados por nivel educativo.
        - Test ANOVA para comparar las medias de varias muestras.
        - Tabla de contingencia y prueba de chi-cuadrado.
        - Correlación de rango de Kendall entre vuelos reservados y nivel educativo.

        Parámetros:
        - df(DataFrame): DataFrame combinado de los datos de vuelo y lealtad.

        Returns:
        No devuelve nada directamente, pero imprime en la consola los resultados del análisis.
        """
    # Abrimos el dataframe clasificado solo por las columnas de interés
    df_education_flights = df[['flights_booked', 'education']]
    print(df_education_flights.head())

    # Análisis Descriptivo
    grupo_educativo = df.groupby('education')['flights_booked']
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
        conversion_rate = tasa_conversion(df, grupo)
        print(f"Tasa de conversión para el grupo educativo {grupo}: {conversion_rate}")


    # Test ANOVA para diferentes grupos de educación
    anova_resultado = f_oneway(df[df["education"] == "Bachelor"]["flights_booked"],
                                df[df["education"] == "College"]["flights_booked"],
                                df[df["education"] == "Doctor"]["flights_booked"],
                                df[df["education"] == "Master"]["flights_booked"],
                                df[df["education"] == "High School or Below"]["flights_booked"])

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
                data=df, 
                palette="muted")
    plt.title("Vuelos Reservados por Grupo Educativo")
    plt.xlabel("Nivel Educativo")
    plt.ylabel("Vuelos Reservados")
    plt.xticks(rotation=45)

    # Tabla de contingencia y prueba de chi-cuadrado
    contingency_table = pd.crosstab(df['education'], df['flights_booked'])
    chi2, p, dof, expected = chi2_contingency(contingency_table)
    print("Estadístico Chi-cuadrado:", chi2)
    print("Valor p:", p)

    # Correlación de rango de Kendall
    kendall_corr, kendall_p = kendalltau(df['flights_booked'], df['education'])
    print("Correlación de rango de Kendall:", kendall_corr)
    print("Valor p de Kendall:", kendall_p)
    
    #%%
