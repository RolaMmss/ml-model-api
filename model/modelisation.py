def modelisation(connection,run_name):
    import pandas as pd
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import recall_score, accuracy_score, f1_score
    import mlflow
    
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, PolynomialFeatures
    from sklearn.compose import ColumnTransformer

    modelisation_query = f"""
    SELECT etat_de_route, carburant, turbo, nombre_portes, type_vehicule, roues_motrices, emplacement_moteur, empattement, longueur, largeur, hauteur, poids_vehicule, type_moteur, nombre_cylindres, taille_moteur, systeme_carburant, taux_alésage, course, taux_compression, chevaux, tour_moteur, consommation_ville, consommation_autoroute, prix, marque, modèle
    FROM {run_name}_CleanDataset
    """


    df = pd.read_sql_query(modelisation_query,connection)
    # df = df.dropna()

    y = df['prix']
    X = df.drop(['prix'], axis=1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=42)

    categorial_features = ['etat_de_route', 'carburant','turbo','nombre_portes', 'type_vehicule', 'roues_motrices', 'emplacement_moteur','type_moteur', 'nombre_cylindres', 'systeme_carburant', 'marque', 'modèle']
    numeric_features = ['empattement', 'longueur' , 'largeur','hauteur', 'poids_vehicule', 'taille_moteur', 'taux_alésage', 'course', 'taux_compression','chevaux','tour_moteur', 'consommation_ville', 'consommation_autoroute']


    categorical_transformer = Pipeline(steps=[
        ('onehotencoder', OneHotEncoder(handle_unknown='ignore'))
    ])
    # define the numeric transformer
    numeric_transformer = Pipeline([
            ('min_max', MinMaxScaler()),  #StandardScaler():moyenne 0 et écart type = 1 -> Reg, SVM, PCA
            ('poly', PolynomialFeatures(degree=2, include_bias=False))
            ])


    # use the ColumnTransformer to apply the appropriate transformers to each column
    preprocessor = ColumnTransformer(transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorial_features)
    ])

    ###############################################################################################
   

    # Set the experiment name or ID
    experiment_name = "predict_price" if run_name != "test_run" else "test_experiment"

    # Get or create the experiment
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        experiment_id = mlflow.create_experiment(experiment_name)
    else:
        experiment_id = experiment.experiment_id

    with mlflow.start_run(experiment_id=experiment_id, run_name=run_name) as run:
        # Create the Random Forest model with 50 trees

        model = Pipeline([
            ('preprocess', preprocessor),  # ajouter des étapes de prétraitement
            ('random_forest', RandomForestRegressor(n_estimators=50, max_depth=20, min_samples_split=2, min_samples_leaf=1,random_state=42))  #score: 0.9210
        ])
        model.fit(X_train,y_train)

        # mlflow.set_tag("start_date", start_date)
        # mlflow.set_tag("end_date", end_date)

        # recall_train = round(recall_score(y_train, model.predict(X_train)),4)
        # acc_train = round(accuracy_score(y_train, model.predict(X_train)),4)
        # f1_train = round(f1_score(y_train, model.predict(X_train)),4)

        # mlflow.log_metric("recall_train", recall_train)
        # mlflow.log_metric("accuracy_train", acc_train)
        # mlflow.log_metric("f1_train", f1_train)

        # print(f"Pour le jeu d'entrainement: \n le recall est de {recall_train}, \n l'accuracy de {acc_train} \n le f1 score de {f1_train}")

        # recall_test = round(recall_score(y_test, model.predict(X_test)),4)
        # acc_test = round(accuracy_score(y_test, model.predict(X_test)),4)
        # f1_test = round(f1_score(y_test, model.predict(X_test)),4)

        # mlflow.log_metric("recall_test", recall_test)
        # mlflow.log_metric("accuracy_test", acc_test)
        # mlflow.log_metric("f1_test", f1_test)
        
        # Predict on the test data
        y_pred = model.predict(X_test)

        # Calculate the R^2 score on the test data
        score = model.score(X_test, y_test)

        # Log parameters
        mlflow.log_param("n_estimators", 15)
        mlflow.log_param("max_depth", 20)

        # Log metrics
        mlflow.log_metric("R2 score", score)
        print(f"The R^2 score on the test data is {score}")

        # Save the model to MLflow
        mlflow.sklearn.log_model(model,run_name)
        run_id = run.info.run_id
    return run_id
    

if __name__== "main":
    import sqlite3
    connection = sqlite3.connect("cars.db")