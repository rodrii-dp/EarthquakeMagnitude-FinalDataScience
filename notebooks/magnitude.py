import os
import requests
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import sweetviz as sv
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import r2_score, mean_squared_error
import gradio as gr

if not os.path.exists('results'):
    os.makedirs('results')

url = 'https://earthquake.usgs.gov/fdsnws/event/1/query'

params = {
    'format': 'geojson',
    'starttime': '2014-10-21',
    'endtime': '2015-10-21',
    'minmagnitude': 0,
    'limit': 20000,
}

response = requests.get(url, params=params)

if response.status_code == 200:
    data = response.json()

    earthquakes = []
    for feature in data['features']:
        properties = feature['properties']
        geometry = feature['geometry']

        earthquake_info = {
            'place': properties['place'],
            'magnitude': properties['mag'],
            'time': pd.to_datetime(properties['time'], unit='ms'),
            'latitude': geometry['coordinates'][1],
            'longitude': geometry['coordinates'][0],
            'depth': geometry['coordinates'][2],
        }
        earthquakes.append(earthquake_info)

    df = pd.DataFrame(earthquakes)

    csv_path = 'results/earthquakes_last_10_years.csv'
    df.to_csv(csv_path, index=False)
    print(f"Datos guardados en '{csv_path}'.")

    excel_path = 'results/earthquakes_last_10_years.xlsx'
    df.to_excel(excel_path, index=False)
    print(f"Datos guardados en '{excel_path}'.")

else:
    print(f"Error en la solicitud: {response.status_code}")

# Exploración de datos
print(df.head())
print(df.describe())
print(df.info())

# Preparar datos
df['magnitude'] = df['magnitude'].astype(float)
df['time'] = pd.to_datetime(df['time'])
df['year'] = df['time'].dt.year
df['month'] = df['time'].dt.month
df['day'] = df['time'].dt.day
df['latitude'] = df['latitude'].astype(float)
df['longitude'] = df['longitude'].astype(float)

# Datos de entrenamiento
X = df[['latitude', 'longitude', 'depth']]
y = df['magnitude']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

params = {
    'n_estimators': [10, 50, 100, 200, 500],
    'max_depth': [None, 5, 10, 20, 50, 100],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
}

rf = RandomForestRegressor(random_state=42)
rs = RandomizedSearchCV(rf, params, n_iter=10, cv=5, n_jobs=-1, random_state=42)
rs.fit(X_train, y_train)

X2_train, X_val, y2_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

best_model = rs.best_estimator_
best_model.fit(X2_train, y2_train)

y_pred = best_model.predict(X_val)
r2 = r2_score(y_val, y_pred)
mse = mean_squared_error(y_val, y_pred)
rmse = mse ** 0.5
print(f"R2: {r2:.2f}")
print(f"MSE: {mse:.2f}")
print(f"RMSE: {rmse:.2f}")

y_pred = best_model.predict(X_test)
r2 = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = mse ** 0.5
print(f"R2: {r2:.2f}")
print(f"MSE: {mse:.2f}")
print(f"RMSE: {rmse:.2f}")

sns.scatterplot(x=y_test, y=y_pred)
plt.xlabel('Magnitud real')
plt.ylabel('Magnitud predicha')
plt.savefig('results/scatter_plot.png')
plt.close()

df = df[['magnitude', 'latitude', 'longitude', 'depth']]
corr_matrix = df.corr()
sns.heatmap(corr_matrix, annot=True)
plt.savefig('results/corr_heatmap.png')
plt.close()

# Análisis con Sweetviz
report = sv.analyze(df)
report.show_html('results/earthquakes_report.html')

# Interfaz de Gradio para predicciones
def predict_magnitude(latitude, longitude, depth):
    data = {'latitude': [latitude], 'longitude': [longitude], 'depth': [depth]}
    df_input = pd.DataFrame(data)
    magnitude = best_model.predict(df_input)[0]
    return magnitude

latitude = gr.Number(label='Latitud')
longitude = gr.Number(label='Longitud')
depth = gr.Number(label='Profundidad')
output = gr.Number(label='Magnitud')

gr.Interface(fn=predict_magnitude, inputs=[latitude, longitude, depth], outputs=output).launch()
