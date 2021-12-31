import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)
model = pickle.load(open('ML_LoanPrediction_Model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    float_features = [float(x) for x in request.form.values()]
    final_features = [np.array(float_features)]
    lp = model.predict(final_features)

    if lp == 0:
        LPred = print(f'Sorry, you do not qualify for a loan at this time.')
    elif lp == 1:
        LPred = print(f'Congralutaions, you qualify for a loan.')
    else:
        LPred = print('please speak with a customser care repersentative.')


    return render_template('index.html', LPred = LPred )

@app.route('/predict_api',methods=['POST'])
def predict_api():
    '''
    For direct API calls throught request
    '''
    data = request.get_json(force=True)
    lp = model.predict([np.array(list(data.values()))])

    if lp == 0:
        LPred = print(f'Sorry, you do not qualify for a loan at this time.')
    elif lp == 1:
        LPred = print(f'Congralutaions, you qualify for a loan.')
    else:
        LPred = print('please speak with a customser care repersentative.')
       
    return jsonify(LPred)

if __name__ == "__main__":
    app.run(debug=True)