from app import app
from flask import render_template
import request

@app.route('/index')
def index():
   return "Hello, World!"

@app.route('/')
@app.route('/input')
def cesareans_input():
    return render_template("input.html")