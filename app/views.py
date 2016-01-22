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
	
	askingPrice=float(request.args['price'])

	#need to format into dict with list for values. Have gender switch for first test, and age +/ 50% for next two elements
	nVars=6
	inputDict={'lnPrice':[None]*nVars}
	for k in request.args.keys():
		if k=='price':
			continue
		inputDict[k]=[request.args[k]]*nVars
	print "inputDict",inputDict
	if inputDict['gender'][0]=="Mare":
		inputDict['gender'][1]="Gelding"
	else:
		inputDict['gender'][1]="Mare"
	inputDict['age'][2]=float(inputDict['age'][0])*0.5
	inputDict['age'][3]=float(inputDict['age'][0])*1.5
	inputDict['inches'][4]=float(inputDict['inches'][0])+4
	inputDict['inches'][5]=float(inputDict['inches'][0])-4
	print "inputDict",inputDict
	pred=np.exp(m.runPrediction(inputDict))
	perDiff=int(100.*pred[0]/askingPrice)
	#outStr="The estimated price for a %s year old %s %s inch tall %s is $%s."%(inputDict['age'][0], "s", 0, "s" ,0)
	outStr="The estimated price for a %s year old %s inch tall %s %s is $%i."%(int(inputDict['age'][0]),int(inputDict['inches'][0]), inputDict['breed'][0],  inputDict['gender'][0], int(pred[0]))
	if perDiff > 100:#prediction is more than asking
		headerStr="$%i is probably a good deal"%askingPrice
	else:
		headerStr="$%i is probably not a good deal"%askingPrice
	genderStr="A similar %s would cost $%i."%(inputDict['gender'][1], int(pred[1]))
	ageStr="A similar %s year old would cost $%i. A similar %s year old would cost $%i."%(int(inputDict['age'][2]), int(pred[2]), int(inputDict['age'][3]), int(pred[3]))
	htStr="A similar %s inch tall horse would cost $%i. A similar %s inch tall horse would cost $%i."%(inputDict['inches'][4], int(pred[4]), inputDict['inches'][5], int(pred[5]))
	return render_template("index.html", outStr=outStr, headerStr=headerStr, genderStr=genderStr, ageStr=ageStr, htStr=htStr)