# # # main.py script
# # from fastapi import FastAPI, Depends
# # import api.predict
# # from api.utils import has_access
# # from fastapi import FastAPI
# # from fastapi.params import Depends
# # from api.utils import has_access, SinglePredictionInput, SinglePredictionOutput, predict_single, get_model_path
# # from typing import List, Annotated

# # app = FastAPI()

# # # routes
# # PROTECTED = [Depends(has_access)]

# # app.include_router(
# #     api.predict.router,
# #     prefix="/predict",
# #     dependencies=PROTECTED
# # )





# # Import necessary libraries
# from fastapi import FastAPI
# import sqlite3
# import pandas as pd
# from sklearn.pipeline import Pipeline
# from sklearn.compose import ColumnTransformer
# from sklearn.preprocessing import OneHotEncoder, MinMaxScaler, PolynomialFeatures
# from sklearn.ensemble import RandomForestRegressor
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import recall_score, accuracy_score, f1_score
# import mlflow

# # Initialize FastAPI
# app = FastAPI()
# # Load data and train the model
# def load_data_and_train_model():
#     connection = sqlite3.connect("cars.db")
#     df = pd.read_sql_query("SELECT * FROM TrainingDataset", connection)
#     connection.close()
    
#     y = df['prix']
#     X = df.drop(['prix'], axis=1)
#     X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=42)

#     categorial_features = ['etat_de_route', 'carburant','turbo','nombre_portes', 'type_vehicule', 'roues_motrices', 'emplacement_moteur','type_moteur', 'nombre_cylindres', 'systeme_carburant', 'marque', 'modèle']
#     numeric_features = ['empattement', 'longueur' , 'largeur','hauteur', 'poids_vehicule', 'taille_moteur', 'taux_alésage', 'course', 'taux_compression','chevaux','tour_moteur', 'consommation_ville', 'consommation_autoroute']

#     categorical_transformer = Pipeline(steps=[
#         ('onehotencoder', OneHotEncoder(handle_unknown='ignore'))
#     ])

#     numeric_transformer = Pipeline([
#             ('min_max', MinMaxScaler()),
#             ('poly', PolynomialFeatures(degree=2, include_bias=False))
#             ])

#     preprocessor = ColumnTransformer(transformers=[
#         ('num', numeric_transformer, numeric_features),
#         ('cat', categorical_transformer, categorial_features)
#     ])

#     model = Pipeline([
#         ('preprocess', preprocessor),
#         ('random_forest', RandomForestRegressor(n_estimators=50, max_depth=20, min_samples_split=2, min_samples_leaf=1,random_state=42))
#     ])
#     model.fit(X_train,y_train)

#     return model, X_test, y_test

# # Endpoint for predicting car price
# @app.post("/predict_price/")
# def predict_price(data: dict):
#     # Load trained model and test data
#     model, X_test, y_test = load_data_and_train_model()
    
#     # Get features from request
#     features = pd.DataFrame(data, index=[0])
    
#     # Make prediction
#     prediction = model.predict(features)
    
#     return {"predicted_price": prediction[0]}

# # Main function
# if __name__ == "__main__":
#     import sqlite3
#     connection = sqlite3.connect("cars.db")
#     # You may want to do more here, like setting up your database schema if it's not already set up.
#     connection.close()


