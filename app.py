from flask import Flask, jsonify, request, send_file
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from pandas import DataFrame
import matplotlib.pyplot as plt
import io
app = Flask(__name__)

# Funciones auxiliares
def train_val_test_split(df, rstate=42, shuffle=True, stratify=None):
    strat = df[stratify] if stratify else None
    train_set, test_set = train_test_split(
        df, test_size=0.4, random_state=rstate, shuffle=shuffle, stratify=strat)
    strat = test_set[stratify] if stratify else None
    val_set, test_set = train_test_split(
        test_set, test_size=0.5, random_state=rstate, shuffle=shuffle, stratify=strat)
    return (train_set, val_set, test_set)

def remove_labels(df, label_name):
    X = df.drop(label_name, axis=1)
    y = df[label_name].copy()
    return (X, y)

# Cargar y procesar el dataset
df = pd.read_csv('../forest/TotalFeatures-ISCXFlowMeter.csv')
df['calss'], _ = pd.factorize(df['calss'])
train_set, val_set, test_set = train_val_test_split(df)
X_train, y_train = remove_labels(train_set, 'calss')
X_val, y_val = remove_labels(val_set, 'calss')
X_test, y_test = remove_labels(test_set, 'calss')

# Escalado de los datos
scaler = RobustScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)
X_train_scaled = DataFrame(X_train_scaled, columns=X_train.columns, index=X_train.index)

# Configuración del modelo
rf_model = RandomForestRegressor(
    n_estimators=5,
    max_depth=10,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42
)
rf_model.fit(X_train, y_train)

# Ruta de inicio
@app.route('/')
def home():
    return """
    <h1>Bienvenido al API de predicción</h1>
    <p>Rutas disponibles:</p>
    <ul>
        <li><a href="/metrics">/metrics</a> - Ver métricas del modelo</li>
        <li><a href="/plot">/plot</a> - Ver gráfica de predicciones vs valores reales</li>
    </ul>
    """

# Ruta para las métricas y la importancia de características
@app.route('/metrics', methods=['GET'])
def get_metrics():
    y_pred = rf_model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    importancia = pd.DataFrame({
        'caracteristica': X_train.columns,
        'importancia': rf_model.feature_importances_
    }).sort_values('importancia', ascending=False)
    
    metrics = {
        "mse": mse,
        "r2": r2,
        "importancia": importancia.to_dict(orient='records')
    }
    
    return jsonify(metrics)

# Ruta para visualizar predicciones vs valores reales
@app.route('/plot', methods=['GET'])
def plot_predictions():
    y_pred = rf_model.predict(X_test)
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred, alpha=0.5)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    plt.xlabel('Valores reales')
    plt.ylabel('Predicciones')
    plt.title('Predicciones vs Valores Reales')
    plt.tight_layout()
    
    # Guardar la imagen en un objeto de bytes
    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plt.close()
    
    return send_file(img, mimetype='image/png')

if __name__ == '__main__':
    app.run(debug=True)
