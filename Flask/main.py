import pickle
from flask import Flask, request, json, jsonify
import numpy as np

app = Flask(__name__)

#---the filename of the saved model--
filename = 'pulsar_star_svm_model.pkl'

 #---load the saved model--
loaded_model = pickle.load(open(filename, 'rb'))

@app.route('/pulsarstar/v1/predict', methods=['POST'])
def predict():
    #---get the features to predict--
    features = request.json
    #---create the features list for prediction--
    # IP Mean	IP Sd	IP Kurtosis	IP Skewness	DM-SNR Mean	DM-SNR Sd	DM-SNR Kurtosis	DM-SNR Skewness
    features_list = [features["IP Mean"],
                     features["IP Sd"],
                     features["IP Kurtosis"],
                     features["IP Skewness"],
                     features["DM-SNR Mean"],
                     features["DM-SNR Sd"],
                     features["DM-SNR Kurtosis"],
                     features["DM-SNR Skewness"]
                    ]
    
    #---get the prediction class--
    prediction = loaded_model.predict([features_list])
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