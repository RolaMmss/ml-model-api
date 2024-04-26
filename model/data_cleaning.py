import pandas as pd
import os
import sqlite3
import numpy as np

def data_cleaning(connection,run_name):
# # Connect to the existing cars.db file
# connection = sqlite3.connect("cars.db")


    
    df = pd.read_sql_query("SELECT * FROM Cars",connection)


    # Separate column "CarName" into two distinct columns: "marque" and "modèle"
    df[['marque', 'modèle']] = df['CarName'].str.split(' ', n=1, expand=True)

    # Drop the 'CarName' column
    df.drop('CarName', axis=1, inplace=True)

    # df.to_csv('prix_voiture.csv', index=False)

    # Nettoyer et modifier les données pour qu'il soit conforme aux normes françaises:
    # Modify the values of the "fueltype" column
    df["fueltype"] = df["fueltype"].replace({"gas": "essence", "diesel": "diesel"})

    # Modify the values of the "aspiration" column
    df["aspiration"] = df["aspiration"].replace({"std": "atmosphérique", "turbo": "turbo"})

    # Modify the values of the "doornumber" column
    df["doornumber"] = df["doornumber"].replace({"two": "deux", "four": "quatre"})

    # Modify the values of the "carbody" column
    df["carbody"] = df["carbody"].replace({"hatchback": "berline compacte", "sedan": "berline", "wagon": "break"})

    # Modify the values of the "drivewheel" column
    df["drivewheel"] = df["drivewheel"].replace({"rwd": "propulsion", "fwd": "traction", "4wd": "quatre roues motrices"})

    # Modify the values of the "enginelocation" column
    df["enginelocation"] = df["enginelocation"].replace({"front": "avant", "rear": "arrière"})

    # Rename the columns "carlength", "carwidth", and "carheight"
    df = df.rename(columns={"carlength": "longueur", "carwidth": "largeur", "carheight": "hauteur"})
    #######################################################

    df = df.rename(columns={'car_ID': 'identifiant', 'symboling': 'etat_de_route', 'fueltype': 'carburant', 'aspiration': 'turbo', 'doornumber': 'nombre_portes', 'carbody': 'type_vehicule', 'drivewheel': 'roues_motrices', 'enginelocation': 'emplacement_moteur', 'wheelbase': 'empattement', 'carlength': 'longueur_voiture', 'carwidth': 'largeur_voiture', 'carheight': 'hauteur_voiture', 'curbweight': 'poids_vehicule', 'enginetype': 'type_moteur', 'cylindernumber': 'nombre_cylindres', 'enginesize': 'taille_moteur', 'fuelsystem': 'systeme_carburant', 'boreratio': 'taux_alésage', 'stroke': 'course', 'compressionratio': 'taux_compression', 'horsepower': 'chevaux', 'peakrpm': 'tour_moteur', 'citympg': 'consommation_ville', 'highwaympg': 'consommation_autoroute', 'price': 'prix'})


    # Delete column car_id
    df = df.drop('identifiant', axis=1)

    # Missing values #
    # There are two missing values in column "modèle". Delete these two rows.
    df = df.dropna()


    if df.shape[1]!= 26:
        raise ValueError("Le nombre de colonnes ne correspond pas")
    else:
        print("Valeurs manquantes: OK")

    # Select numerical columns
    numeric_cols = df.select_dtypes(include=['int', 'float']).columns

    # Convert american units into french units
    df['empattement'] = (df['empattement'] * 0.0254).round(2)
    df['longueur'] = df['longueur'] * 0.0254
    df['largeur'] = df['largeur'] * 0.0254
    df['hauteur'] = df['hauteur'] * 0.0254
    df['poids_vehicule'] = df['poids_vehicule'] * 0.453592
    df['taille_moteur'] = df['taille_moteur'] * 0.0163871
    df['taux_alésage'] = df['taux_alésage'] * 25.4
    df['course'] = df['course'] * 25.4
    df['consommation_ville'] = 235.214 / df['consommation_ville']
    df['consommation_autoroute'] = 235.214 / df['consommation_autoroute']

    # Duplicates :
    df = df.drop_duplicates()

    # Unify names. Correction of writing mistakes
    df["marque"] = df["marque"].replace({"porcshce": "porsche", "toyouta": "toyota", "vokswagen": "volkswagen", "maxda": "mazda", "Nissan": "nissan"})
    df["modèle"] = df["modèle"].replace({"100 ls": "100ls"})

    return df


if __name__ == "__main__":
    connection = sqlite3.connect("cars.db")
    df = data_cleaning(connection,"first_run_2017")
    df.to_sql("first_run_2017" +'_CleanDataset', connection, index=False, if_exists='replace')