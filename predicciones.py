# Importar librerías
import mysql.connector
import pandas as pd
import plotly as plt
import plotly.graph_objects as go
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
import pickle
import os
import kaleido

# Colocarse en el directorio actual
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# Ruta de guardado de las imágenes
ruta = '../assets/'

# Crear conexión a MySQL
mydb = mysql.connector.connect(
  host="tripulaciones_sql",
  port="3306",
  user="admin",
  password="Tripulaciones1.",
  database="tripulaciones_backend" 
)

crsr = mydb.cursor()
print(mydb)

# Con esta función leemos los datos y lo pasamos a un DataFrame de Pandas
def sql_query(query):

    # Ejecuta la query
    crsr.execute(query)

    # Almacena los datos de la query 
    ans = crsr.fetchall()

    # Obtenemos los nombres de las columnas de la tabla
    names = [description[0] for description in crsr.description]

    return pd.DataFrame(ans,columns=names)

query = '''
SELECT * FROM `vote` 
INNER JOIN `user` 
ON user_userid = user.userid 
INNER JOIN `dept` 
ON user.dept_deptid = dept.deptid; 
'''

df = sql_query(query)

df.drop(columns=['user_userid', 'email', 'password', 'role', 'firstname', 
                 'lastname', 'active', 'dept_deptid', 'terminationdate'], inplace = True)

df.set_index('voteid', inplace= True)

df['total_votes'] = df['clockinvote'] + df['clockoutvote'] / 2

media_general = df[['clockinvote', 'clockoutvote']].mean(axis=1)

# Configura el objeto de la figura
fig = go.Figure(go.Indicator(
    mode="gauge+number",
    value=float(media_general.iloc[0]),
    title={'text': "Índice de felicidad"},
    gauge={'axis': {'range': [0, 5]}, 'bar': {'color': "#FF145A"}},
    domain={'x': [0, 1], 'y': [0, 1]},
))

# Agrega un subtítulo usando update_layout
fig.update_layout(annotations=[dict(text="Toda la empresa", x=0.5, y=1.2, showarrow=False)])

# Guarda el gráfico
fig.write_image(ruta + 'grafico_1.png')

media_clockinvote = df['clockinvote'].mean()

fig = go.Figure(go.Indicator(
    mode = "gauge+number",
    value = media_clockinvote,
    title = {'text': "Índice de felicidad"},
    gauge={'axis': {'range': [0, 5]}, 'bar': {'color': "#FF145A"}},
    domain = {'x': [0, 1], 'y': [0, 1]}
))

fig.update_layout(annotations=[dict(text="Entrada", x=0.5, y=1.2, showarrow=False)])

# Guarda el gráfico
fig.write_image(ruta + 'grafico_2.png')

media_clocoutvote = df['clockoutvote'].mean()

fig = go.Figure(go.Indicator(
    mode = "gauge+number",
    value = media_clocoutvote,
    title = {'text': "Índice de felicidad"},
    gauge={'axis': {'range': [0, 5]}, 'bar': {'color': "#FF145A"}},
    domain = {'x': [0, 1], 'y': [0, 1]}
))

fig.update_layout(annotations=[dict(text="Salida", x=0.5, y=1.2, showarrow=False)])

# Guarda el gráfico
fig.write_image(ruta + 'grafico_3.png')


total_votes = pd.concat([df['clockinvote'], df['clockoutvote']]).value_counts().sort_index()

colores_escala = ['#FF145A', '#2B2B2B', '#545454', '#7C7B7A', '#ACACAC', 'f0F0F0']

# Crear el gráfico de donut con colores personalizados
fig = go.Figure(data=[go.Pie(labels=total_votes.index, values=total_votes, 
                             hole=0.4, marker=dict(colors=colores_escala))])

# Personalizar el diseño con título y subtítulo
fig.update_layout(title_text='Toda la empresa | Votos totales')

# Guarda el gráfico
fig.write_image(ruta + 'grafico_4.png')


votos_contados_entrada = df['clockinvote'].value_counts()

# Definir una escala de colores según la descripción
colores_escala = ['#FF145A', '#2B2B2B', '#545454', '#7C7B7A', '#ACACAC', 'f0F0F0']

# Crear el gráfico de donut con colores personalizados
fig = go.Figure(data=[go.Pie(labels=votos_contados_entrada.index, values=votos_contados_entrada, 
                             hole=0.4, marker=dict(colors=colores_escala))])

# Personalizar el diseño con título y subtítulo
fig.update_layout(title_text='Entrada | Votos totales')

# Guarda el gráfico
fig.write_image(ruta + 'grafico_5.png')


votos_contados_salida = df['clockoutvote'].value_counts()

# Definir una escala de colores según la descripción
colores_escala = ['#FF145A', '#2B2B2B', '#545454', '#7C7B7A', '#ACACAC', 'f0F0F0']

# Crear el gráfico de donut con colores personalizados
fig = go.Figure(data=[go.Pie(labels=votos_contados_salida.index, values=votos_contados_salida, 
                             hole=0.4, marker=dict(colors=colores_escala))])

# Personalizar el diseño con título y subtítulo
fig.update_layout(title_text='Salida | Votos totales')

# Guarda el gráfico
fig.write_image(ruta + 'grafico_6.png')


df['date'] = pd.to_datetime(df['date'])
df['fecha'] = df['date'].dt.date

media_diaria = df.groupby('fecha')[['clockinvote', 'clockoutvote']].mean().reset_index()
media_diaria['media_diaria'] = media_diaria[['clockinvote', 'clockoutvote']].mean(axis=1)

fig = px.line(media_diaria, x='fecha', y='media_diaria', title='Evolución diaria del índice de satisfacción')

fig.update_traces(line_color='#FF145A', line_shape='linear', fill='tozeroy', fillcolor='rgba(255, 20, 90, 0.3)')
fig.update_yaxes(title_text='índice de satisfacción', range=[0, 5])
fig.update_layout(plot_bgcolor='#f0f0f0')

# Guarda el gráfico
fig.write_image(ruta + 'grafico_7.png')


df['date'] = pd.to_datetime(df['date'])
df['fecha'] = df['date'].dt.date

media_diaria_entrada = df.groupby('fecha')['clockinvote'].mean().reset_index(name='media_diaria_entrada')

fig = px.line(media_diaria_entrada, x='fecha', y='media_diaria_entrada', title='Entrada | Evolución diaria del índice de satisfacción')

fig.update_traces(line_color='#FF145A', line_shape='linear', fill='tozeroy', fillcolor='rgba(255, 20, 90, 0.3)')
fig.update_yaxes(title_text='índice de satisfacción', range=[0, 5])
fig.update_layout(plot_bgcolor='#f0f0f0')

# Guarda el gráfico
fig.write_image(ruta + 'grafico_8.png')


df['date'] = pd.to_datetime(df['date'])
df['fecha'] = df['date'].dt.date

media_diaria_salida = df.groupby('fecha')['clockoutvote'].mean().reset_index(name='media_diaria_salida')

fig = px.line(media_diaria_salida, x='fecha', y='media_diaria_salida', title='Salida | Evolución diaria del índice de satisfacción')

fig.update_traces(line_color='#FF145A', line_shape='linear', fill='tozeroy', fillcolor='rgba(255, 20, 90, 0.3)')
fig.update_yaxes(title_text='índice de satisfacción', range=[0, 5])
fig.update_layout(plot_bgcolor='#f0f0f0')

# Guarda el gráfico
fig.write_image(ruta + 'grafico_9.png')


# Crear un nuevo DataFrame combinando las columnas relevantes
new_df = pd.DataFrame({
    'vote': pd.concat([df['clockinvote'], df['clockoutvote']]),
    'tag': pd.concat([df['clockintag'], df['clockouttag']])
})

# Calcular el porcentaje en lugar de contar las ocurrencias
heatmap_data = new_df.groupby(['tag', 'vote']).size().unstack(fill_value=0)
heatmap_data_percentage = heatmap_data.div(heatmap_data.sum(axis=1), axis=0) * 100

# Convertir los valores a cadenas con el símbolo de porcentaje
heatmap_data_percent_str = heatmap_data_percentage.applymap(lambda x: f"{x:.2f}%")

# Crear el mapa de calor con Seaborn y anotar los valores como cadenas
plt.figure(figsize=(10, 8))
sns.heatmap(heatmap_data_percentage, annot=heatmap_data_percent_str.values, fmt="", cbar_kws={'label': 'Porcentaje'})

# Actualizar el diseño del mapa de calor
plt.title('Razones')
plt.xlabel('Índice de satisfacción')
plt.ylabel('Motivos')

# Guarda el gráfico
plt.savefig(ruta + 'grafico_10.png')


heatmap_data = df.groupby(['clockintag', 'clockinvote']).size().unstack(fill_value=0)

# Normaliza los valores para obtener porcentajes basados en el total de votos
heatmap_data_entrada = heatmap_data.div(heatmap_data.sum(axis=0), axis=1) * 100

# Configura el tamaño del gráfico
plt.figure(figsize=(10, 8))

# Crea el heatmap con Seaborn
sns.heatmap(heatmap_data_entrada, annot=True, fmt=".1f", cmap="rocket", cbar_kws={'label': 'Porcentaje'})

# Configura los títulos y etiquetas
plt.title('Entrada | Razones del voto')
plt.xlabel('Índice de satisfacción')
plt.ylabel('Motivos')

# Guarda el gráfico
plt.savefig(ruta + 'grafico_11.png')


heatmap_data = df.groupby(['clockouttag', 'clockoutvote']).size().unstack(fill_value=0)

# Normaliza los valores para obtener porcentajes basados en el total de votos
heatmap_data_salida = heatmap_data.div(heatmap_data.sum(axis=0), axis=1) * 100

# Configura el tamaño del gráfico
plt.figure(figsize=(10, 8))

# Crea el heatmap con Seaborn
sns.heatmap(heatmap_data_salida, annot=True, fmt=".1f", cmap="rocket", cbar_kws={'label': 'Porcentaje'})

# Configura los títulos y etiquetas
plt.title('Salida | Razones del voto')
plt.xlabel('Índice de satisfacción')
plt.ylabel('Motivos')

# Guarda el gráfico
plt.savefig(ruta + 'grafico_12.png')


df_means = df.groupby('name')['total_votes'].mean().reset_index()

fig = px.bar(df_means, x='name', y='total_votes', title='Índice de Satisfacción por departamentos',
             color_discrete_sequence=['#FF145A'], labels={'name': 'Departamentos', 'total_votes': 'Índice de Satisfacción'})

# Guarda el gráfico
fig.write_image(ruta + 'grafico_13.png')


clf_terminated = pickle.load(open('./pred_flight.pkl', 'rb'))


media_predicciones = clf_terminated.predict(df[['total_votes']]).mean()
media_predicciones_formatted = "{:.1f}".format(media_predicciones)


# Definir los umbrales para los niveles de riesgo
umbral_bajo = 0.2
umbral_medio = 0.5
umbral_alto = 0.8

# Determinar el nivel de riesgo según los umbrales
if media_predicciones <= umbral_bajo:
    nivel_riesgo = "Riesgo muy bajo"
elif umbral_bajo < media_predicciones <= umbral_medio:
    nivel_riesgo = "Riesgo bajo"
elif umbral_medio < media_predicciones <= umbral_alto:
    nivel_riesgo = "Riesgo medio"
else:
    nivel_riesgo = "Riesgo alto"

# Configurar el objeto de la figura
fig = go.Figure(go.Indicator(
    mode="gauge+number",
    value=float(media_predicciones_formatted),  # Usar la cadena del nivel de riesgo como valor
    title={'text': "Riesgo de fuga"},
    gauge={'axis': {'range': [0, 1]}, 'bar': {'color': "#FF145A"}},
    domain={'x': [0, 1], 'y': [0, 1]},
))

# Agregar un subtítulo utilizando update_layout
fig.update_layout(
    annotations=[
        dict(text=nivel_riesgo, x=0.5, y=0.3, font=dict(size=30), showarrow=False)
    ]
)

# Guarda el gráfico
fig.write_image(ruta + 'grafico_14.png')


df['predicted_fuga'] = clf_terminated.predict(df[['total_votes']])
df_means = df.groupby('name')['predicted_fuga'].mean().reset_index()

umbral_bajo = 0.2
umbral_medio = 0.5
umbral_alto = 0.8

df_means['nivel_riesgo'] = pd.cut(df_means['predicted_fuga'], bins=[-float('inf'), umbral_bajo, umbral_medio, umbral_alto, float('inf')],
                                  labels=["Riesgo muy bajo", "Riesgo bajo", "Riesgo medio", "Riesgo alto"])

# category_order = ["Riesgo muy bajo", "Riesgo bajo", "Riesgo medio", "Riesgo alto"]

fig = px.bar(df_means, x='name', y='predicted_fuga', color='nivel_riesgo',
             title='Riesgo de fuga por departamentos',
             color_discrete_map={'Riesgo muy bajo': '#7C7B7A', 'Riesgo bajo': '#545454', 'Riesgo medio': '#2B2B2B', 'Riesgo alto': '#FF145A'},
             labels={'name': 'Departamentos', 'predicted_fuga': 'Riesgo de fuga', 'nivel_riesgo': 'Nivel de riesgo'})

fig.update_yaxes(range=[0, 1], showline=False, showgrid=False)

# Guarda el gráfico
fig.write_image(ruta + 'grafico_15.png')


clf_absences = pickle.load(open('./pred_abs.pkl', 'rb'))


media_predicciones_abs = clf_absences.predict(df[['total_votes']]).mean()
media_predicciones_absences_formatted = "{:.1f}".format(media_predicciones_abs)


# Definir los umbrales para los niveles de riesgo
umbral_bajo = 5
umbral_medio = 10
umbral_alto = 15

# Determinar el nivel de riesgo según los umbrales
if media_predicciones_abs <= umbral_bajo:
    nivel_riesgo = "Riesgo muy bajo"
elif umbral_bajo < media_predicciones_abs <= umbral_medio:
    nivel_riesgo = "Riesgo bajo"
elif umbral_medio < media_predicciones_abs <= umbral_alto:
    nivel_riesgo = "Riesgo medio"
else:
    nivel_riesgo = "Riesgo alto"

# Configurar el objeto de la figura
fig = go.Figure(go.Indicator(
    mode="gauge+number",
    value=float(media_predicciones_absences_formatted),  # Usar la cadena del nivel de riesgo como valor
    title={'text': "Riesgo de absentismo"},
    gauge={'axis': {'range': [0, 20]}, 'bar': {'color': "#FF145A"}},
    domain={'x': [0, 1], 'y': [0, 1]},
))

# Agregar un subtítulo utilizando update_layout
fig.update_layout(
    annotations=[
        dict(text=nivel_riesgo, x=0.5, y=0.3, font=dict(size=30), showarrow=False)
    ]
)

# Guarda el gráfico
fig.write_image(ruta + 'grafico_16.png')


df['predicted_abs'] = clf_absences.predict(df[['total_votes']])

df_means = df.groupby('name')['predicted_abs'].mean().reset_index()

umbral_bajo = 5
umbral_medio = 10
umbral_alto = 15

# Agrega el argumento 'categories' para asegurar todas las etiquetas
df_means['nivel_riesgo'] = pd.cut(df_means['predicted_abs'], bins=[-float('inf'), umbral_bajo, umbral_medio, umbral_alto, float('inf')],
                                  labels=["Riesgo muy bajo", "Riesgo bajo", "Riesgo medio", "Riesgo alto"])

# category_order = ["Riesgo muy bajo", "Riesgo bajo", "Riesgo medio", "Riesgo alto"]

# Restringe las categorías a solo aquellas presentes en df_means
categories_presentes = df_means['nivel_riesgo'].unique()

color_map={'Riesgo muy bajo': '#7C7B7A', 'Riesgo bajo': '#545454', 'Riesgo medio': '#2B2B2B', 'Riesgo alto': '#FF145A'}

# Define el mapeo de colores solo para las categorías presentes
color_discrete_map = {categoria: color_map[categoria] for categoria in categories_presentes}

fig = px.bar(df_means, x='name', y='predicted_abs', c
             # color='nivel_riesgo',
             title='Riesgo de absentismo por departamentos',
             # color_discrete_map=color_discrete_map,
             labels={'name': 'Departamentos', 'predicted_abs': 'Riesgo de absentismo', 'nivel_riesgo': 'Nivel de riesgo'})

fig.update_yaxes(range=[0, 15], showline=False, showgrid=False)

# Guarda el gráfico
fig.write_image(ruta + 'grafico_17.png')
