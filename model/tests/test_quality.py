import pytest
import pandas as pd
from model.data_cleaning import data_cleaning
# from model.feature_engineering import feature_engineering
from model.modelisation import modelisation
from unittest.mock import patch  # Import patch from unittest.mock
import mlflow

def pytest_addoption(parser):
    parser.addoption("--run_name", action="store", default="test_run")
    parser.addoption("--start_date", action="store", default="2017-01-01")
    parser.addoption("--end_date", action="store", default="2018-01-01")

@pytest.fixture
def run_name(request):
    return request.config.getoption("--run_name")

@pytest.fixture
def start_date(request):
    return request.config.getoption("--start_date")

@pytest.fixture
def end_date(request):
    return request.config.getoption("--end_date")

@pytest.fixture
def connection(run_name):
    if run_name == 'test_run':
        # # Create a mock connection
        # class MockConnection:
        #     def __init__(self):
        #         self.mock_reviews_data = pd.DataFrame({
        #             'order_id': [0, 1, 2, 3, 3, 4],
        #             'review_score': [1, 4, 5, 3, 3, 4],
        #             'review_creation_date': ['2016-01-01 12:00:00','2017-01-01 12:00:00', '2017-01-02 12:00:00', '2017-01-03 12:00:00',  '2017-01-03 12:00:00', pd.NaT],
        #         })

        #         # Mock data for the "orders" table
        #         self.mock_orders_data = pd.DataFrame({
        #             'order_id': [0, 1, 2, 3, 4],
        #             'order_purchase_timestamp': ['2016-01-01 12:00:00', '2016-01-02 12:00:00', '2016-01-03 12:00:00', '2016-01-04 12:00:00', '2016-01-05 12:00:00'],
        #             'order_delivered_customer_date': ['2016-01-05 12:00:00', '2016-01-07 12:00:00', '2016-01-09 12:00:00', '2016-01-11 12:00:00', '2016-01-12 12:00:00'],
        #             'order_estimated_delivery_date': ['2016-01-07 12:00:00', '2016-01-09 12:00:00', '2016-01-11 12:00:00', '2016-01-13 12:00:00', '2016-01-14 12:00:00']
        #         })

        #         self.mock_clean_data = pd.DataFrame({
        #             'order_id': [1, 2, 3],
        #             'review_score': [4, 5, 3],
        #             'review_creation_date': ['2017-01-01 12:00:00', '2017-01-02 12:00:00', '2017-01-03 12:00:00'],
        #             'order_purchase_timestamp': [ '2016-01-02 12:00:00', '2016-01-03 12:00:00', '2016-01-04 12:00:00'],
        #             'order_delivered_customer_date': [ '2016-01-07 12:00:00', '2016-01-09 12:00:00', '2016-01-11 12:00:00'],
        #             'order_estimated_delivery_date': [ '2016-01-09', '2016-01-11', '2016-01-13']
        #         })

        #         self.mock_training_data = pd.DataFrame({
        #             'score': [0, 1, 1, 0, 0],
        #             'produit_recu': [1, 1, 0, 0, 1],
        #             'temps_livraison': [2, 3, 1, 4, 2]
        #         })

        # return MockConnection()
        class MockConnection:
            def __init__(self):
                self.mock_car_data = pd.DataFrame({
                    'etat_de_route': [3, 3, 1, 2, 2],
                    'carburant': ['essence', 3, 'essence', 'essence', 'essence'],
                    'turbo': ['atmosphérique', 'atmosphérique', 'atmosphérique', 'atmosphérique', 'atmosphérique'],
                    'nombre_portes': ['deux', 'deux', 'deux', 'quatre', 'quatre'],
                    'type_vehicule': ['convertible', 'convertible', 'berline compacte', 'berline', 'berline'],
                    'roues_motrices': ['propulsion', 'propulsion', 'propulsion', 'traction', 'quatre roues motrices'],
                    'emplacement_moteur': ['avant', 'avant', 'avant', 'avant', 'avant'],
                    'empattement': [2.25, 2.25, 2.4, 2.53, 2.52],
                    'longueur': [4.28752, 4.28752, 4.34848, 4.48564, 4.48564],
                    'largeur': [1.62814, 1.62814, 1.6637, 1.68148, 1.68656],
                    'hauteur': [1.23952, 1.23952, 1.33096, 1.37922, 1.37922],
                    'poids_vehicule': [1155.752416, 1155.752416, 1280.490216, 1060.044504, 1280.943808],
                    'type_moteur': ['dohc', 'dohc', 'ohcv', 'ohc', 'ohc'],
                    'nombre_cylindres': ['three', 'four', 'six', 'four', 'five'],
                    'taille_moteur': [2.130323, 2.130323, 2.4908392, 1.7861939, 2.2286456],
                    'systeme_carburant': ['mpfi', 'mpfi', 'mpfi', 'mpfi', 'mpfi'],
                    'taux_alésage': [88.138, 88.138, 68.072, 81.026, 81.026],
                    'course': [68.072, 68.072, 88.138, 86.36, 86.36],
                    'taux_compression': [9.0, 9.0, 9.0, 10.0, 8.0],
                    'chevaux': [111, -10, 154, 102, 115],
                    'tour_moteur': [5000, 5000, 5000, 5500, 5500],
                    'consommation_ville': [11.200666666666667, 11.200666666666667, 12.379684210526316, 9.800583333333334, 13.067444444444444],
                    'consommation_autoroute': [8.711629629629629, 8.711629629629629, 9.046692307692307, 7.840466666666667, 10.691545454545455],
                    'marque': ['alfa-romero', 'alfa-romero', 'alfa-romero', 'audi', 'audi'],
                    'modèle': ['giulia', 'stelvio', 'Quadrifoglio', '100ls', '100ls']
                })
                self.mock_clean_data = pd.DataFrame({
                    'order_id': [1, 2, 3],
                    'review_score': [4, 5, 3],
                    'review_creation_date': ['2017-01-01 12:00:00', '2017-01-02 12:00:00', '2017-01-03 12:00:00'],
                    'order_purchase_timestamp': [ '2016-01-02 12:00:00', '2016-01-03 12:00:00', '2016-01-04 12:00:00'],
                    'order_delivered_customer_date': [ '2016-01-07 12:00:00', '2016-01-09 12:00:00', '2016-01-11 12:00:00'],
                    'order_estimated_delivery_date': [ '2016-01-09', '2016-01-11', '2016-01-13']
                })

                self.mock_training_data = pd.DataFrame({
                    'score': [0, 1, 1, 0, 0],
                    'produit_recu': [1, 1, 0, 0, 1],
                    'temps_livraison': [2, 3, 1, 4, 2]
                })

        #     # Introducing inconsistencies
        #     self.mock_car_data.loc[0, 'prix'] = 100000  # Introduce a very high price
        #     self.mock_car_data.loc[1, 'empattement'] = 3.0  # Introduce an unrealistic empattement value
        #     self.mock_car_data.loc[2, 'nombre_cylindres'] = 'three'  # Introduce an invalid value for nombre_cylindres
        #     self.mock_car_data.loc[3, 'chevaux'] = -10  # Introduce a negative value for chevaux

        # def get_mock_data(self):
        #     return self.mock_car_data
        return MockConnection()

    else:
        # Create a real connection
        import sqlite3
        connection = sqlite3.connect("cars.db")
        return connection
        

# def test_data_cleaning(connection,run_name, start_date, end_date):
def test_data_cleaning(connection,run_name):

    if run_name == 'test_run':
            # Mocking the SQL query execution
        def mock_read_sql_query(query, connection):
            if 'cars' in query:
                return connection.mock_car_data
            
        with patch('pandas.read_sql_query', side_effect=mock_read_sql_query):
            df_clean = data_cleaning(connection,run_name)

        assert len(df_clean) == 24
    else:
        df_clean = pd.read_sql_query(f"SELECT * FROM {run_name}_CleanDataset",connection)
        # df_clean.review_creation_date = pd.to_datetime(df_clean['review_creation_date'])

    assert len(df_clean) > 0
    
    assert set(df_clean.columns) == {'etat_de_route', 'carburant','turbo','nombre_portes', 'type_vehicule', 'roues_motrices', 'emplacement_moteur',
                                     'type_moteur', 'nombre_cylindres', 'systeme_carburant', 'marque', 'modèle','empattement', 'longueur' , 'largeur',
                                     'hauteur', 'poids_vehicule', 'taille_moteur', 'taux_alésage', 'course', 'taux_compression','chevaux','tour_moteur', 
                                     'consommation_ville', 'consommation_autoroute'} # Check if all expected columns are present

    # Assert that there are no raw before the start_date
    assert df_clean['review_creation_date'].min() >= pd.to_datetime(start_date)
    assert df_clean['review_creation_date'].max() <= pd.to_datetime(end_date)

    # Add more assertions to check specific cleaning steps based on your function's logic
    # For example, you can check if the date format conversion is correct
    assert df_clean['review_creation_date'].dtype == '<M8[ns]'
    
    assert not df_clean[['order_id', 'review_score','review_creation_date']].isnull().values.any(), "DataFrame contains missing values"

    # Check for duplicated rows
    assert not df_clean.duplicated().any(), "DataFrame contains duplicated rows"

# def test_feature_engineering(connection,run_name):

#     if run_name == 'test_run':
#             # Mocking the SQL query execution
#         def mock_read_sql_query(query, connection):
#             return connection.mock_clean_data
        
#         with patch('pandas.read_sql_query', side_effect=mock_read_sql_query):
#             df_feat = feature_engineering(connection,run_name)
#     else:
#         df_feat = pd.read_sql_query(f"SELECT * FROM {run_name}_TrainingDataset",connection)

#     assert set(df_feat.columns) == {'order_id', 'review_score','review_creation_date',
#                                      'order_purchase_timestamp','order_delivered_customer_date','order_estimated_delivery_date',
#                                      'score', 'temps_livraison', 'produit_recu', 'retard_livraison'
#                                      }  # Check if all expected columns are present

#     assert df_feat['score'].isin([0, 1]).all()
#     assert df_feat['produit_recu'].isin([0, 1]).all()
#     assert df_feat['temps_livraison'].dtype in [int, 'int64'], f"Column 'temps_livraison' is not of integer type"

def test_modelisation(connection,run_name):
    import pickle
    import os
    from sklearn.linear_model import LogisticRegression
    if run_name == 'test_run':
            # Mocking the SQL query execution
        def mock_read_sql_query(query, connection):
            return connection.mock_training_data
        
        with patch('pandas.read_sql_query', side_effect=mock_read_sql_query):
            run_id = modelisation(connection,run_name)
    else:
        experiment = mlflow.get_experiment_by_name("predict_price")
        runs = mlflow.search_runs(experiment_ids=experiment.experiment_id)
        # Filter runs by run name
        filtered_runs = runs[runs['tags.mlflow.runName'] == run_name]
        run_id = filtered_runs.iloc[0]['run_id']

    
    run = mlflow.get_run(run_id)
    assert run
    # Get run metrics
    metrics = run.data.metrics
    assert set(metrics.keys()) == {'accuracy_test', 'accuracy_train', 'f1_test', 'f1_train','recall_train','recall_test'}
    

    if run_name != 'test_run':
        # test for accuracy
        assert metrics['accuracy_test'] > 0.6
        # test for overfitting
        assert metrics['accuracy_train']- metrics['accuracy_test'] < 0.2

    artifact_uri = run.info.artifact_uri

    # Construct the path to the model pickle file within the artifacts directory
    modelel_pickle_path = os.path.join(artifact_uri.replace("file://", ""), run_name, "model.pkl")

    with open(modelel_pickle_path, 'rb') as f:
        model = pickle.load(f)
        assert type(model) == LogisticRegression
    
    if run_name == 'test_run':
        mlflow.delete_run(run_id)