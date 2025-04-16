import pickle
from flask import Flask, request, json, jsonify
import numpy as np
import pandas as pd
import json

app = Flask(__name__)

#---the filename of the saved model--
filename = 'pulsar_star_svm_model.pkl'
scalar_filename = 'pulsar_star_pp.pkl'

# load the scaler
loaded_scalar = pickle.load(open(scalar_filename, 'rb'))

 #---load the saved model--
loaded_model = pickle.load(open(filename, 'rb'))

@app.route('/diabetes/v1/predict', methods=['POST'])
def predict():
    #---get the features to predict--
    features = request.json
    # feature_json = json.dump(features, indent=4)
    
    # features_df = pd.read_json(feature_json)
    # #---scale the features--
    # scaled_features = loaded_scalar.transform(features_df)
    #---make the prediction--
    # IP Mean	IP Sd	IP Kurtosis	IP Skewness	DM-SNR Mean	DM-SNR Sd	DM-SNR Kurtosis	DM-SNR Skewness

    #---get the prediction class--
    prediction = loaded_model.predict(scaled_features)
    print(prediction)
    #---get the prediction probabilities--
    # confidence = loaded_model.predict_proba([features_list])
    # print(confidence)
    #---formulate the response to return to client--
    response = {}
    response['prediction'] = int(prediction[0])
    # response['confidence'] = str(round(np.amax(confidence[0]) * 100 ,2))
    return  jsonify(response)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)