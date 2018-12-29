from flask import Flask, request, render_template
import pickle
import numpy as np
import pandas as pd
from sklearn.externals import joblib

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict',methods=['POST','GET'])
def get_delay():

    if request.method=='POST':

        result=request.form
        gender=result['gender']
        Occupation=result['Occupation']
        Marital_Status=result['Marital_Status']
        Age_group=result['Age_group']
        product_category_1=result['product_category_1']
        product_category_2=result['product_category_2']
        product_category_3=result['product_category_3']
        City_Category_A=result['City_Category_A']
        City_Category_B=result['City_Category_B']
        City_Category_C=result['City_Category_C']
        Stay_In_Current_City_Years_0=result['Stay_In_Current_City_Years_0']
        Stay_In_Current_City_Years_1=result['Stay_In_Current_City_Years_1']
        Stay_In_Current_City_Years_2=result['Stay_In_Current_City_Years_2']
        Stay_In_Current_City_Years_3=result['Stay_In_Current_City_Years_3']
        Stay_In_Current_City_Years_4_above=result['Stay_In_Current_City_Years_4_above']
        
        lst=[]
        
        lst.append(Occupation)
        lst.append(Marital_Status)
        lst.append(Age_group)
        lst.append(product_category_1)
        lst.append(product_category_2)
        lst.append(product_category_3)
        lst.append(gender)
        lst.append(City_Category_A)
        lst.append(City_Category_B)
        lst.append(City_Category_C)
        lst.append(Stay_In_Current_City_Years_0)
        lst.append(Stay_In_Current_City_Years_1)
        lst.append(Stay_In_Current_City_Years_2)
        lst.append(Stay_In_Current_City_Years_3)
        lst.append(Stay_In_Current_City_Years_4_above)

        df=pd.DataFrame(lst).T

        loaded_model = joblib.load('Black_Friday2.pkl')
        prediction = loaded_model.predict(df)

    return render_template('result.html',prediction=prediction)

    
if __name__ == '__main__':
    app.debug = True
    app.run()