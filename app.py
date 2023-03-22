from flask import Flask,render_template,request
import pickle
import numpy as np

model=pickle.load(open("model.pkl","rb"))

app=Flask(__name__)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict",methods=["POST"])
def predict_wine_quality():
    fixed_acidity=float(request.form.get("fixed acidity"))
    volatile_acidity=float(request.form.get("volatile acidity"))
    citric_acid=float(request.form.get("citric acid"))
    residual_sugar=float(request.form.get("residual sugar"))
    chlorides=float(request.form.get("chlorides"))
    free_sulfur_dioxide=float(request.form.get("free sulfur dioxide"))
    total_sulfur_dioxide=float(request.form.get("total sulfur dioxide"))
    density=float(request.form.get("density"))
    pH=float(request.form.get("pH"))
    sulphates=float(request.form.get("sulphates"))
    alcohol=float(request.form.get("alcohol"))
    
    
    result=model.predict(np.array([[fixed_acidity,volatile_acidity,citric_acid,residual_sugar,chlorides,free_sulfur_dioxide,total_sulfur_dioxide,density,pH,sulphates,alcohol]]))
    
    if result[0]==0:
        return "<h1 style='color:red'>Bad Quality</h1>"
    else:
        return "<h1 style='color:green'>Good Quality</h1>"
    
  
app.run(host="0.0.0.0",port=8080)