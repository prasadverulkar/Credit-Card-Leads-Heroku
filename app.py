import json

import numpy as np
from flask import Flask, request, jsonify, render_template
from flask import Response
import pickle
import json

app = Flask(__name__)
# For model prediction
model = pickle.load(open('model.pkl','rb'))
# For Label Encode
le = pickle.load(open('le_encode.pkl','rb'))
# For Target Encode
tar = pickle.load(open('tar_enc.pkl','rb'))

@app.route('/',methods = ['GET'])
def home():
    return render_template('index.html')

@app.route('/predict',methods = ['POST'])
def predict():
    try:
        features = [x for x in request.form.values()]
        features[1] = int(features[1])
        features[5] = int(features[5])
        features[7] = int(features[7])
        
        # feature Engineering
        
        if ((features[1] >= 37.5)|(features[1] <= 65)) & (features[3] =="Salaried"):
            features.append(1)
        else:
            features.append(0)

        
        # Apply log to avg_account_balance
        features[7] = np.log(features[7])
        
        # Target Encoding
        values = tar.to_dict()
        features[2] = values[features[2]]

        
        final_features = np.array(features).reshape(1,10)
            
        prediction = model.predict(final_features)
        if prediction == 0:
            output = 'No'
        else:
            output = 'Yes'
            
        return render_template('index.html', prediction_text='Potential Lead for Credit Card?, {}'.format(output))
    
    except ValueError:
        return Response("Error Occurred! %s" %ValueError)
    except KeyError:
        return Response("Error Occurred! %s" %KeyError)
    except Exception as e:
        return Response("Error Occurred! %s" %e)

predict()
if __name__ == "__main__":
    app.run(debug=True)
