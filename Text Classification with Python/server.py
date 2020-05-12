from flask import Flask, render_template, request
from main import predict
import json

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html', name='test')

@app.route('/predict', methods=['POST'])
def predictMe():
    result = predict(request.form['text'], request.form['name'], request.form['email'], request.form['phone'])
    return render_template('predict.html', pred=result)


