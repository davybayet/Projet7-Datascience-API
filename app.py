import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from flask import Flask, jsonify, request, make_response
import pickle

app = Flask(__name__)
# charger le dataset et le model
FILE_TEST_SET = 'resources/data/test_set.pickle'
with open(FILE_TEST_SET, 'rb') as df_test_set:
            test_set = pickle.load(df_test_set)
FILE_BEST_MODELE = 'resources/modele/best_model.pickle'
with open(FILE_BEST_MODELE, 'rb') as model_lgbm:
            best_model = pickle.load(model_lgbm)

print("API ready")

def convert_types(df, print_info = False):
    original_memory = df.memory_usage().sum()
    # Iterate through each column
    for c in df:
        # Convert ids and booleans to integers
        if ('SK_ID' in c):
            df[c] = df[c].fillna(0).astype(np.int32)
        # Convert objects to category
        elif (df[c].dtype == 'object') and (df[c].nunique() < df.shape[0]):
            df[c] = df[c].astype('category')
        # Booleans mapped to integers
        elif list(df[c].unique()) == [1, 0]:
            df[c] = df[c].astype(bool)
        # Float64 to float32
        elif df[c].dtype == float:
            df[c] = df[c].astype(np.float32)
        # Int64 to int32
        elif df[c].dtype == int:
            df[c] = df[c].astype(np.int32)
    new_memory = df.memory_usage().sum()
    if print_info:
        print(f'Original Memory Usage: {round(original_memory / 1e9, 2)} gb.')
        print(f'New Memory Usage: {round(new_memory / 1e9, 2)} gb.')
    return df

test_set = convert_types(test_set, print_info=True)

def generateMetrics():# page d'accueil
    description= "Bienvenue sur l'API du projet Implémentez un modèle de scoring:\n\nVoici les endpoints de cette API\n\n:<int:client_id> requête pour prédire le score d'un client sélectionné(saisir l'ID du client dans la barre URL)"
    return(description)

@app.route("/", methods=["GET"])# afficher la description en format texte
def home():
    response = make_response(generateMetrics(), 200)
    response.mimetype = "text/plain"
    return response

#def predict(client_id):
    #X_test=test_set[test_set['SK_ID_CURR']==int(client_id)]
    #y_proba=best_model.predict_proba(X_test.drop(['SK_ID_CURR'],axis=1))[:, 1]
    #return y_proba
    # return jsonify({'result': str(np.around(y_proba[0]*100,2))+'%'})

#prédit le score de faillite d'un client sélectionné
@app.route('/<int:client_id>/', methods=["GET"])
def predict(client_id):
    if client_id not in list(test_set['SK_ID_CURR']):
        return 'Ce client n\'est pas dans la base de donnée'
    else:
        # client_id = request.args.get(client_id)
        X_test=test_set[test_set['SK_ID_CURR']==client_id]
        y_proba=best_model.predict_proba(X_test.drop(['SK_ID_CURR'],axis=1))[:, 1]
        return jsonify({'result': y_proba[0]})


if __name__ == '__main__':
    app.run(debug=True)
    pass
