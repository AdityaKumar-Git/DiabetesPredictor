from flask import Flask, render_template, request
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score

model = pickle.load(open('classifier.pkl','rb'))
diabetes_dataset = pd.read_csv('diabetes.csv')
X = diabetes_dataset.drop(columns = 'Outcome', axis=1)

app = Flask(__name__)

@app.route('/')
def home():
    result = ""
    return render_template('index.html', **locals())

@app.route('/predict', methods=['POST'])
def pred():
    data1 = request.form['fname']
    data2 = request.form['pregnancies']
    data3 = request.form['glucose']
    data4 = request.form['bloodPressure']
    data5 = request.form['skinThickness']
    data6 = request.form['insulin']
    data7 = request.form['BMI']
    data8 = request.form['diabetesPedigreeFunction']
    data9 = request.form['age']
    
    if(data2==""):
        data2="0"
        
    input_data = (data2,data3,data4,data5,data6,data7,data8,data9)
    
    for i in input_data:
        if(i=="" or i==None):
            result=""
            return render_template('index.html',**locals())

    # changing the input_data to numpy array
    input_data_as_numpy_array = np.asarray(input_data)

    # reshape the array as we are predicting for one instance
    input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

    # standardize the input data
    scaler = StandardScaler()
    scaler.fit(X)
    
    std_data = scaler.transform(input_data_reshaped)
    prediction = model.predict(std_data)
    
    # print(prediction)
    
    if(prediction[0] == 1):
        result = f"{data1} is predicted to be diabetic"
    else:
        result = f"{data1} is predicted to be non-diabetic"
    return render_template('index.html',**locals())
        
if __name__ == "__main__":
    app.run(debug=False,host='0.0.0.0')