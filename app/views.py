from app import app
from flask import render_template
from flask import request
import model as m
import numpy as np
import seaborn as sns
import pygal, sys

style=pygal.style.CleanStyle


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
	ageRange=np.linspace(1, 15, 15)
	genderRange=["Gelding", "Mare", "Stallion"]
	heightRange=np.linspace(float(request.args['inches'])-8.0, float(request.args['inches'])+8.0, 9)
	ageStart=1
	genderStart=ageStart+len(ageRange)
	heightStart=genderStart+len(genderRange)
	nVars=heightStart+1+len(heightRange)

	inputDict={'lnPrice':[None]*nVars}
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

	print "inputDict",inputDict
	pred=np.exp(m.runPrediction(inputDict))

	agePlt=pygal.StackedLine(disable_xml_declaration=True, style=style, interpolate='cubic', fill=True)
	agePlt.title="Estimated price with age"
	agePlt.x_labels=["%i"%i for i in ageRange]
	agePlt.add("Price ($)", pred[ageStart:ageStart+len(ageRange)])
	agePlt.x_title="Age (years)"
	agePlt.y_title="Estimated price ($)"

	genderPlt=pygal.Bar(disable_xml_declaration=True, style=style)
	genderPlt.title="Estimated price with gender"
	genderPlt.x_title="Gender"
	genderPlt.y_title="Estimated price ($)"
	for g, v in zip(genderRange, pred[genderStart:genderStart+len(genderRange)]):
		genderPlt.add(g, v)



	htPlt=pygal.StackedLine(disable_xml_declaration=True, style=style, interpolate='cubic', fill=True)
	htPlt.title="Estimated price with height"
	htPlt.x_labels=["%i in."%i for i in heightRange]
	htPlt.add("Price ($)", pred[heightStart:heightStart+len(heightRange)])
	htPlt.x_title="Height (inches)"
	htPlt.y_title="Estimated price ($)"
	#agePlt.render_to_file("agechart.svg")

	perDiff=int(100.*pred[0]/askingPrice)
	#outStr="The estimated price for a %s year old %s %s inch tall %s is $%s."%(inputDict['age'][0], "s", 0, "s" ,0)
	outStr="The estimated price for a %s year old %s inch tall %s %s is $%i."%(int(inputDict['age'][0]),inputDict['inches'][0], inputDict['breed'][0],  inputDict['gender'][0], int(pred[0]))
	if perDiff > 100:#prediction is more than asking
		headerStr="$%i is probably a good deal"%askingPrice
	else:
		headerStr="$%i is probably not a good deal"%askingPrice
	ageStr=genderStr=htStr=""
	# genderStr="A similar %s would cost $%i."%(inputDict['gender'][1], int(pred[1]))
	# ageStr="A similar %s year old would cost $%i. A similar %s year old would cost $%i."%(int(inputDict['age'][2]), int(pred[2]), int(inputDict['age'][3]), int(pred[3]))
	# htStr="A similar %s inch tall horse would cost $%i. A similar %s inch tall horse would cost $%i."%(inputDict['inches'][4], int(pred[4]), inputDict['inches'][5], int(pred[5]))
	return render_template("index.html", outStr=outStr, headerStr=headerStr, genderStr=genderStr, ageStr=ageStr, htStr=htStr, ageChart=agePlt, htChart=htPlt, genderChart=genderPlt)