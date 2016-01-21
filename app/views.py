from app import app
from flask import render_template
from flask import request
from model import predictPrice

@app.route('/index')
def index():
   return "Hello, World!"

@app.route('/')
@app.route('/input')
def input():
    return render_template("input.html")

@app.route('/output')
def output():
	price=predictPrice()
	return render_template("output.html", price=price)