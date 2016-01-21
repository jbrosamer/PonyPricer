from app import app
from flask import render_template
from flask import request
import model as m
import numpy as np


@app.route('/index')
def index():
   return "Hello, World!"

@app.route('/')
@app.route('/input')
def input():
    return render_template("input.html")

@app.route('/output')
def output():
	print request.args
	inputDict={'lnPrice':[None]}
	askingPrice=float(request.args['price'])
	#need to format into dict with list for values
	for k in request.args.keys():
		if k=='price':
			continue
		inputDict[k]=[request.args[k]]
	pred=np.exp(m.runPrediction(inputDict))
	perDiff=int(100.*pred[0]/askingPrice)
	outStr="The estimated price is $%i, %i%% of the $%i asking price!"%(pred, perDiff, askingPrice)
	if perDiff > 100:#prediction is more than asking
		outStr+="\nProbably a good deal!"
	else:
		outStr+="\nProbably not a good deal!"
	return render_template("output.html", outStr=outStr)