from flask import Flask, jsonify, request
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import  pickle
import numpy as np

app = Flask(__name__)

@app.route("/api/predict", methods=['POST'])
def predict():
    crim = request.json['crim']
    zn = request.json['zn']
    indus = request.json['indus']
    chars = request.json['chars']
    nox = request.json['nox']
    rm = request.json['rm']
    age = request.json['age']
    dis = request.json['dis']
    rad = request.json['rad']
    tax = request.json['tax']
    ptratio = request.json['ptratio']
    b = request.json['b']
    lstat = request.json['lstat']

    data = np.array([[crim, zn, indus, chars, nox, rm, age,  dis, rad, tax, ptratio, b, lstat]], dtype=float); 

    std = StandardScaler()
    data_std = std.fit_transform(data)

    with open('model.sav', 'rb') as f: 
    	u = pickle._Unpickler(f) 
    	u.encoding = 'latin1' 
    	model = u.load() 

    resultado = model.predict(data_std);

    return jsonify({'resultado': resultado[0]}), 201

if __name__ == "__main__":
    app.run(debug=True)
