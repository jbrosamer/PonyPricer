from mechanize import Browser
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
import re, pickle, sys, traceback
from datetime import datetime

url="http://www.dreamhorse.com/d/5/dressage/horses-for-sale.html"
br = Browser()
br.set_handle_robots(False)
br.set_handle_robots(False)
br.set_handle_equiv(False)
br.addheaders = [('User-agent', 'Mozilla/5.0 (X11; U; Linux i686; en-US; rv:1.9.0.1) Gecko/2008071615 Fedora/3.0.1-1.fc9 Firefox/3.0.1')]
columns=['id','breed', 'breedStr', 'price', 'color','location', 'age', 'zip', 'height', 'temp', 'warmblood', 'sold', 'soldhere']
columns+=['forsale', 'forlease', 'registered', 'skills', 'desc', 'gender']
cellColMap={32: 'for lease', 34: 'for sale', 2: 'zip', 36: 'price', 38: 'skills', 6: 'age', 8: 'gender', 10: 'height', 14: 'color', 20: 'warmblood', 22: 'temp', 4: 'breed', 26: 'registered', 40: 'desc'}
# mech = Browser()
# page = br.open(url)
# html = page.read()
# soup = BeautifulSoup(html)
# numInSearch=None
#can just get ids then use to look at ads
fromPickle=True
pickleIds=None
outputPath="/Users/jbrosamer/PonyPricer/Batch/"
if fromPickle:
	pickleIds=pickle.load(open("/Users/jbrosamer/PonyPricerFiles/ScrapedIds.p", 'rb'))

def dateStrToAge(cellTxt):
	try:
		bd=datetime.strptime(cellTxt.split("\n")[0],"%b %Y")
	except:
		bd=datetime.strptime("JAN "+cellTxt.split("\n")[0],"%b %Y")
	age=(datetime.today()-bd).days/365.
	return age

def heightStrToIn(cellTxt):
	heightStr=cellTxt.split("\n")[1]
	if "hh" in heightStr:
		hhHeight=heightStr.split(" ")[0].split(".")
		inHeight=float(hhHeight[0])*4+float(hhHeight[1])
	else:
		inHeight=float(heightStr.split(" ")[0])
	return heightStr


def scrapeSearch(url, batchStart=0, batchSize=1000):
	if fromPickle:
		print "Scraping ",batchStart, "to ",batchSize+batchStart
		ids=pickleIds[batchStart:batchStart+batchSize]
		print "Loaded %i Ids"%len(ids)
	else:
	    page = br.open(url)
	    html = page.read()
	    ids=[]
	    while html != "":
	    	ids+=extractIds(html)
	    	html=getNextPage(br)
	    print "Done with len(ids)",len(ids)
	    pickle.dump(ids, open("DressageIds.p", 'wb'))
	df=pd.DataFrame(columns=columns, index=ids)

	startTime = datetime.now()
	badIds=open("/Users/jbrosamer/PonyPricer/Batch/BadIds%iTo%i.txt"%(batchStart, batchSize+batchStart), 'wb')
	for n, i in enumerate(ids):
		try:
			srs=scrapeAd(i)
			df.loc[i]=scrapeAd(i)
		except Exception as e:
			print "BadId",i," exception ",str(e)
			print '-'*60
			traceback.print_exc(file=sys.stdout)
			print '-'*60
			badIds.write("%i\n"%i)
		if n%10==0:
			print "ID number:",n
	elapsedSec=(datetime.now()-startTime).seconds
	print "Time to scrape %i ids: %f minutes"%(len(ids), elapsedSec/60.)
	try:
		pickle.dump(df, open("/Users/jbrosamer/PonyPricer/Batch/DressageId%iTo%i.p"%(batchStart, batchSize+batchStart), "wb"))
		df.to_csv("/Users/jbrosamer/PonyPricer/Batch/DressageId%iTo%i.csv"%(batchStart, batchSize+batchStart))
		print "df",df
	except Exception as e:
		badIds.write(str(e)+"\n")
	badIds.close()
    
    
def getNextPage(br):
	for nr, form in enumerate(br.forms()):
		if "http://www.dreamhorse.com/show_list_pages.php"==form.action:
			for control in form.controls:
				if "Next" in repr(control.value):
						print "NR Next",nr
						br.select_form(nr=nr)
						response = br.submit()
						content = response.read()
						return content
	return ""

def extractIds(html):
	soup = BeautifulSoup(html, "html.parser")
	ids=list()
	for input_el in soup.findAll('input'):
		if input_el.has_attr('name') and (input_el['name']=="form_horse_id"):
			ids.append(int(input_el['value']))
	return ids


def scrapeAd(id):
	page = br.open("http://www.dreamhorse.com/ad/%i.html"%id)
	html = page.read()
	soup = BeautifulSoup(html, "html.parser")
	tables=soup.findAll(name="table")
	adDict=dict.fromkeys(columns)
	adDict['id']=id
	cells=tables[3].findAll("td")
	soldCell=soup.findAll("td", class_="navy", limit=2)[-1]
	if "SOLD" in soldCell.text:
		adDict['sold']=True
		if "SOLD HERE" in soldCell.text:
			adDict['soldhere']=False
	else:
		adDict['sold']=adDict['soldhere']=False

	for x in cellColMap.keys():
		colName=cellColMap[x]
		if not adDict['sold']:
			cellTxt=cells[x].text
		else:
			cellTxt=cells[x+1].text
		if colName=='zip':
			adDict['location']=cellTxt.split("\n")[1]
			try:
				zipStr=int(cellTxt.split("\n")[1].split(" ")[-1])
				adDict['zip']=zipStr
			except Exception as e:
				print "Zip exception ",str(e)
				adDict['zip']=0
				continue
		elif "breed"==colName:
			adDict['breed']=""
			adDict['breedStr']=cellTxt
			for t in cellTxt.split("\n"):
				thisCell=str(t.replace(u'\xa0', '').replace("Related Searches by Breed", "")).replace(" Cross", "").strip()
				if len(thisCell) > 5:
					adDict['breed']=thisCell
					continue
		elif "age"==colName:
			adDict['age']=dateStrToAge(cellTxt)
		elif "gender"==colName:
			adDict['gender']=cellTxt.split("\n")[0]
		elif "height"==colName:
			adDict['height']=heightStrToIn(cellTxt)
		elif "color"==colName:
			adDict['color']=cellTxt.split("\n")[0]
		elif "warmblood"==colName:
			if (cellTxt.split("\n")[0].replace(r'\xa0','')=="Yes"):
				adDict['warmblood']=True
			else:
				adDict['warmblood']=False
		elif "temp"==colName:
		   
			try:
				adDict['temp']=int(cellTxt.split("\n")[0])
			except:
				#print "Temp cell",cellTxt.split("\n")," isn't int!"
				adDict['temp']=11
		elif "registered" ==colName:
			if (cellTxt.split("\n")[0].replace(r'\xa0','')=="Yes"):
				adDict['registered']=True
			else:
				adDict['registered']=False
		elif "for sale"==colName:
			if (cellTxt.split("\n")[0].replace(r'\xa0','')=="No"):
				adDict['for sale']=False
			else:
				adDict['for sale']=True
		elif "for lease" ==colName:
			if (cellTxt.split("\n")[0].replace(r'\xa0','')=="Yes"):
				adDict['for lease']=True
			else:
				adDict['for lease']=False
		elif "price"==colName:
			badChars="(),USD$"
			price=cellTxt.split("\n")[0]
			for b in badChars:
				price=price.replace(b, "")
			try:
				adDict['price']=int(price)
			except:
				adDict['price']=0
		elif "skills"==colName:
			adDict['skills']=[]
			for t in cellTxt.split("\n"):
				thisCell=str(t.replace(u'\xa0', '')).strip()
				if len(thisCell) > 3:
					adDict['skills'].append(thisCell)
		elif "desc"==colName:
			adDict['desc']=cellTxt


	# for x in range(len(cells)):
	# 	text=cells[x].text
	# 	if "Location:" in text:
	# 		adDict['location']=cells[x+1].text.split("\n")[1]
	# 		try:
	# 			zipStr=int(cells[x+1].text.split("\n")[1].split(" ")[-1])
	# 			adDict['zip']=zipStr
	# 		except:
	# 			adDict['zip']=0
	# 	elif "Breed:" in text:
	# 		adDict['breed']=""
	# 		for t in cells[x+1].text.split("\n")[:-1]:
	# 			thisCell=str(t.replace(u'\xa0', '').replace("Related Searches by Breed", "")).replace(" Cross", "").strip()
	# 			if len(thisCell) > 5:
	# 				adDict['breed']=thisCell
	# 	elif "Date Foaled:" in text:
	# 		adDict['age']=dateStrToAge(cells[x+1].text)
	# 	elif "Gender:" in text:
	# 		adDict['gender']=cells[x+1].text.split("\n")[0]
	# 	elif "Height:" in text:
	# 		adDict['height']=heightStrToIn(cells[x+1].text)
	# 	elif "Color:" in text:
	# 		adDict['color']=cells[x+1].text.split("\n")[0]
	# 	elif "Warmblood:" in text:
	# 		if (cells[x+1].text.split("\n")[0].replace(r'\xa0','')=="Yes"):
	# 			adDict['warmblood']=True
	# 		else:
	# 			adDict['warmblood']=False
	# 	elif "Temperament:" in text:
		   
	# 		try:
	# 			adDict['temp']=int(cells[x+1].text.split("\n")[0])
	# 		except:
	# 			#print "Temp cell",cells[x+1].text.split("\n")," isn't int!"
	# 			adDict['temp']=11
	# 	elif "Registered?" in text:
	# 		if (cells[x+1].text.split("\n")[0].replace(r'\xa0','')=="Yes"):
	# 			adDict['registered']=True
	# 		else:
	# 			adDict['registered']=False
	# 	elif "For Sale:" in text:
	# 		if (cells[x+1].text.split("\n")[0].replace(r'\xa0','')=="No"):
	# 			adDict['for sale']=False
	# 		else:
	# 			adDict['for sale']=True
	# 	elif "For Lease" in text:
	# 		if (cells[x+1].text.split("\n")[0].replace(r'\xa0','')=="Yes"):
	# 			adDict['for lease']=True
	# 		else:
	# 			adDict['for lease']=False
	# 	elif "Askinto scg Price:" in text:
	# 		badChars="(),USD$"
	# 		price=cells[x+1].text.split("\n")[0]
	# 		for b in badChars:
	# 			price=price.replace(b, "")
	# 		try:
	# 			adDict['price']=int(price)
	# 		except:
	# 			adDict['price']=0
	# 	elif "Horse Skills" in text:
	# 		adDict['skills']=[]
	# 		for t in cells[x+1].text.split("\n"):
	# 			thisCell=str(t.replace(u'\to scxa0', '')).strip()
	# 			if len(thisCell) > 3:
	# 				adDict['skills'].append(thisCell)
	return pd.Series(adDict)


if __name__ == "__main__":
	nPages=5203
	batchSize=500
	start=0
	for x in range(start, nPages, batchSize):
		scrapeSearch(url, x, batchSize)
