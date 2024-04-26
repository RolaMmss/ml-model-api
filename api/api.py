import sqlite3
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler, PolynomialFeatures
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import recall_score, accuracy_score, f1_score
import mlflow
from fastapi import FastAPI
import os

app = FastAPI()

# Load data and train the model
def load_data_and_train_model():
        
   # Get the absolute path to the cars.db file located in the parent directory
    db_path = os.path.abspath("../cars.db")

    # Connect to the existing cars.db file
    connection = sqlite3.connect(db_path)

    df = pd.read_sql_query("SELECT * FROM Cars",connection)

    connection.close()
    
    
    y = df['prix']
    X = df.drop(['prix'], axis=1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=42)

    categorical_features = ['etat_de_route', 'carburant','turbo','nombre_portes', 'type_vehicule', 'roues_motrices', 'emplacement_moteur','type_moteur', 'nombre_cylindres', 'systeme_carburant', 'marque', 'modèle']
    numeric_features = ['empattement', 'longueur' , 'largeur','hauteur', 'poids_vehicule', 'taille_moteur', 'taux_alésage', 'course', 'taux_compression','chevaux','tour_moteur', 'consommation_ville', 'consommation_autoroute']

    categorical_transformer = Pipeline(steps=[
        ('onehotencoder', OneHotEncoder(handle_unknown='ignore'))
    ])

    numeric_transformer = Pipeline([
            ('min_max', MinMaxScaler()),
            ('poly', PolynomialFeatures(degree=2, include_bias=False))
            ])

    preprocessor = ColumnTransformer(transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

    model = Pipeline([
        ('preprocess', preprocessor),
        ('random_forest', RandomForestRegressor(n_estimators=50, max_depth=20, min_samples_split=2, min_samples_leaf=1,random_state=42))
    ])
    model.fit(X_train,y_train)

    return model, X_test, y_test

# Log parameters and metrics to MLflow
def log_to_mlflow(model, X_test, y_test):
    with mlflow.start_run() as run:
        # Log parameters
        mlflow.log_param("n_estimators", model.named_steps['random_forest'].n_estimators)
        mlflow.log_param("max_depth", model.named_steps['random_forest'].max_depth)
        mlflow.log_param("min_samples_split", model.named_steps['random_forest'].min_samples_split)
        mlflow.log_param("min_samples_leaf", model.named_steps['random_forest'].min_samples_leaf)

        # Evaluate model
        y_pred = model.predict(X_test)
        recall = recall_score(y_test, y_pred)
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)

        # Log metrics
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("f1", f1)

        # Log model
        mlflow.sklearn.log_model(model, "random_forest_model")

        return run.info.run_id

@app.post("/train/")
def train_model():
    model, X_test, y_test = load_data_and_train_model()
    run_id = log_to_mlflow(model, X_test, y_test)
    return {"message": "Model trained and logged to MLflow", "mlflow_run_id": run_id}

@app.post("/predict/")
def predict_price(data: dict):
    # Implement prediction logic here using the trained model
    # Example:
    # prediction = model.predict(data)
    # return {"predicted_price": prediction}
    return {"message": "Prediction endpoint is under construction"}

if __name__ == "__main__":
    mlflow.set_tracking_uri("http://localhost:5000")  # Set the MLflow tracking URI
    mlflow.set_experiment("car_price_prediction")     # Set the MLflow experiment name
    app.run(debug=True)
