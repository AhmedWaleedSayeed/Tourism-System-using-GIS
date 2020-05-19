import MySQLdb
import cursor as cursor

from flask import Flask, render_template, request, redirect, url_for, flash
from flask_mysqldb import MySQL
from main import predict
import json

app = Flask(__name__)
app.secret_key = 'many random bytes'

app.config['MYSQL_HOST'] = 'localhost'
app.config['MYSQL_USER'] = 'root'
app.config['MYSQL_PASSWORD'] = ''
app.config['MYSQL_DB'] = 'travel'

mysql = MySQL(app)

@app.route('/')
def home():
    return render_template('index.html', name='Comment')

@app.route('/insert', methods = ['POST'])
def insert():

    if request.method == "POST":
        flash("Data Inserted Successfully")
        name = request.form['name1']
        email = request.form['email1']
        phone = request.form['phone1']
        text = request.form['text1']
        cur = mysql.connection.cursor()
        cur.execute("INSERT INTO rate (name, email, phone, text) VALUES (%s, %s, %s, %s)", (name, email, phone, text))
        mysql.connection.commit()
        return redirect(url_for('home'))

@app.route('/predict', methods=['POST'])
def predictMe():


    result = predict(request.form['text'],request.form['name'], request.form['email'], request.form['phone']) #can remove name email phone
    # test = json.load(result)
    return render_template('predict.html', pred=result)