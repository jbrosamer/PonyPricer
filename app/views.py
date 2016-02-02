from app import app
from flask import render_template
from flask import request
import model as m
import numpy as np
import seaborn as sns
import sys
import math
import categories as cat

# style=pygal.style.CleanStyle


# @app.route('/index')
# def index():
#    return "Hello, World!"

@app.route('/')
@app.route('/input')
def input():
    return render_template("input.html", colors=cat.colors, breeds=cat.breeds)

@app.route('/output')
def output():
	print request.args
	
	askingPrice=float(request.args['price'])

	#need to format into dict with list for values. Have gender switch for first test, and age +/ 50% for next two elements
	ageRange=range(1, 20, 2)
	genderRange=["Gelding", "Mare", "Stallion"]
	intHeight=int(math.floor(float(request.args['inches'])))
	heightRange=range(intHeight-8, intHeight+9, 2)
	ageStart=1
	genderStart=ageStart+len(ageRange)
	heightStart=genderStart+len(genderRange)
	nVars=heightStart+1+len(heightRange)

	inputDict={'logprice':[None]*nVars}
	for k in request.args.keys():
		if k=='price':
			continue
		inputDict[k]=[request.args[k]]*nVars
	print "inputDict",inputDict
	for n, val in enumerate(ageRange):
		inputDict['age'][n+ageStart]=val
	for n, val in enumerate(genderRange):
		inputDict['gender'][n+genderStart]=val
	for n, val in enumerate(heightRange):
		inputDict['inches'][n+heightStart]=val

	# print "inputDict",inputDict

	pred=np.power(10, m.predForWeb(inputDict))
	print "pred",pred

	ageData=list(pred[ageStart:ageStart+len(ageRange)])
	ageRange=list(ageRange)

	genderData=list(pred[genderStart:genderStart+len(genderRange)])

	heightRange=list(heightRange)
	heightData=list(pred[heightStart:heightStart+len(heightRange)])

	
	outStrs="Age: %.1f years \nBreed: %s\nHeight: %.1f in.\nGender: %s\nColor: %s"%(float(inputDict['age'][0]), inputDict['breed'][0], float(inputDict['inches'][0]), inputDict['gender'][0], inputDict['color'][0])
	if pred[0] > askingPrice:#prediction is more than asking
		headerStr="Estimated price $%i, $%i more than asking price"%(int(pred[0]), abs(int(pred[0]-askingPrice)))
	else:
		headerStr="Estimated price $%i, $%i less than asking price"%(int(pred[0]), abs(int(pred[0]-askingPrice)))
	ageStr=genderStr=htStr=""
	params={"estimate":pred[0], "askingPrice":askingPrice, "outStrs": outStrs.split("\n"), "headerStr": headerStr, "ageRange":ageRange, "ageData": ageData, "genderData":genderData, "heightRange":heightRange, "heightData":heightData}
	return render_template("index.html", **params)