import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
import os
import json


app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index1.html')

@app.route('/AboutUs')
def about():
    return render_template('About.html')

@app.route('/Linear')
def Linear():
    return render_template('Linear.html') 

@app.route('/KNN')
def KNN():
    return render_template('KNN.html')

@app.route('/Decision')
def Decision():
    return render_template('Decision.html')

@app.route('/RForest')
def RForest():
    return render_template('RForest.html')

@app.route('/ANN')
def ANN():
    return render_template('ANN.html')

@app.route('/Lasso')
def Lasso():
    return render_template('Lasso.html')

# For Rendering Results 

@app.route('/PredictionSuccess')
def PLinear():
    return render_template('PLinear.html')


@app.route('/predictLinear',methods=['POST'])
def predictLinear():
    '''
    For rendering results on HTML GUI
    '''
    model = pickle.load(open('LinearModel.pkl', 'rb'))
    int_features =[]
    int_features.append(float(request.form['T']))
    int_features.append(float(request.form['TM']))
    int_features.append(float(request.form['Tm']))
    int_features.append(float(request.form['SLP']))
    int_features.append(float(request.form['H']))
    int_features.append(float(request.form['VV']))
    int_features.append(float(request.form['V']))
    int_features.append(float(request.form['VM']))
    # int_features = [float(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)

    output = abs(round(prediction[0], 2))
    if output>=300:
        output=output/3.55;
    output = round(output, 2)
    # return json.dumps({'AQi':output});
    return render_template('PLinear.html', prediction_text='PM2.5 = {} µg/m3'.format(output),
    temp='Temperature = {} °C'.format(int_features[0]),
    maxtemp='Maximum Temperature = {} °C'.format(int_features[1]),
    mintemp='Minimum Temperature = {} °C'.format(int_features[2]),
    atm='Atmospheric Pressure = {} mm Hg'.format(int_features[3]),
    hum='Humidity = {} %'.format(int_features[4]),
    visi='Visibility = {} km'.format(int_features[5]),
    ws='Wind Speed = {} m/s'.format(int_features[6]),
    mws='Maximum Wind Speed = {} m/s'.format(int_features[7]))


@app.route('/predictANN',methods=['POST'])
def predictANN():
    '''
    For rendering results on HTML GUI
    '''
    model = pickle.load(open('ANNModel.pkl', 'rb'))
    int_features =[]
    int_features.append(float(request.form['T']))
    int_features.append(float(request.form['TM']))
    int_features.append(float(request.form['Tm']))
    int_features.append(float(request.form['SLP']))
    int_features.append(float(request.form['H']))
    int_features.append(float(request.form['VV']))
    int_features.append(float(request.form['V']))
    int_features.append(float(request.form['VM']))
    # int_features = [float(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)

    output = round(prediction[0], 2)

    return render_template('ANN.html', prediction_text='PM2.5 = {} µg/m3'.format(output))


@app.route('/predictKNN',methods=['POST'])
def predictKNN():
    '''
    For rendering results on HTML GUI
    '''
    model = pickle.load(open('KNNModel.pkl', 'rb'))
    int_features =[]
    int_features.append(float(request.form['T']))
    int_features.append(float(request.form['TM']))
    int_features.append(float(request.form['Tm']))
    int_features.append(float(request.form['SLP']))
    int_features.append(float(request.form['H']))
    int_features.append(float(request.form['VV']))
    int_features.append(float(request.form['V']))
    int_features.append(float(request.form['VM']))
    # int_features = [float(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)

    output = round(prediction[0], 2)

    return render_template('KNN.html', prediction_text='PM2.5 = {} µg/m3'.format(output))


@app.route('/predictDecision',methods=['POST'])
def predictDecision():
    '''
    For rendering results on HTML GUI
    '''
    model = pickle.load(open('DecisionModel.pkl', 'rb'))
    int_features =[]
    int_features.append(float(request.form['T']))
    int_features.append(float(request.form['TM']))
    int_features.append(float(request.form['Tm']))
    int_features.append(float(request.form['SLP']))
    int_features.append(float(request.form['H']))
    int_features.append(float(request.form['VV']))
    int_features.append(float(request.form['V']))
    int_features.append(float(request.form['VM']))
    # int_features = [float(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)

    output = round(prediction[0], 2)

    return render_template('Decision.html', prediction_text='PM2.5 = {} µg/m3'.format(output))

@app.route('/predictRF',methods=['POST'])
def predictRF():
    '''
    For rendering results on HTML GUI
    '''
    model = pickle.load(open('RFModel.pkl', 'rb'))
    int_features =[]
    int_features.append(float(request.form['T']))
    int_features.append(float(request.form['TM']))
    int_features.append(float(request.form['Tm']))
    int_features.append(float(request.form['SLP']))
    int_features.append(float(request.form['H']))
    int_features.append(float(request.form['VV']))
    int_features.append(float(request.form['V']))
    int_features.append(float(request.form['VM']))
    # int_features = [float(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)

    output = round(prediction[0], 2)

    return render_template('RForest.html', prediction_text='PM2.5 = {} µg/m3'.format(output))

@app.route('/predictLasso',methods=['POST'])
def predictLasso():
    '''
    For rendering results on HTML GUI
    '''
    model = pickle.load(open('LassoModel.pkl', 'rb'))
    int_features =[]
    int_features.append(float(request.form['T']))
    int_features.append(float(request.form['TM']))
    int_features.append(float(request.form['Tm']))
    int_features.append(float(request.form['SLP']))
    int_features.append(float(request.form['H']))
    int_features.append(float(request.form['VV']))
    int_features.append(float(request.form['V']))
    int_features.append(float(request.form['VM']))
    # int_features = [float(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)

    output = round(prediction[0], 2)

    return render_template('Lasso.html', prediction_text='PM2.5 = {} µg/m3'.format(output),
    temp='Temperature = {} °C'.format(int_features[0]),
    maxtemp='Maximum Temperature = {} °C'.format(final_features[1]),
    mintemp='Minimum Temperature = {} °C'.format(final_features[2]),
    atm='Atmospheric Pressure = {} mm Hg'.format(final_features[3]),
    hum='Humidity = {} %'.format(final_features[4]),
    visi='Visibility = {} km'.format(final_features[5]),
    ws='Wind Speed = {} m/s'.format(final_features[6]),
    mws='Maximum Wind Speed = {} m/s'.format(final_features[7]))

# @app.route('/predict_api',methods=['POST'])
# def predict_api():
#     '''
#     For direct API calls trought request
#     '''
#     data = request.get_json(force=True)
#     prediction = model.predict([np.array(list(data.values()))])

#     output = prediction[0]
#     return jsonify(output)

if __name__ == "__main__":
    app.run(debug=True)