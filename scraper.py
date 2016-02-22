from mechanize import Browser
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
import re, pickle, sys, traceback, os
from datetime import datetime
from progressbar import ProgressBar
from collections import OrderedDict

url="http://www.dreamhorse.com/d/5/dressage/horses-for-sale.html"
#initialize browser
br = Browser()
br.set_handle_robots(False)
br.set_handle_robots(False)
br.set_handle_equiv(False)
br.addheaders = [('User-agent', 'Mozilla/5.0 (X11; U; Linux i686; en-US; rv:1.9.0.1) Gecko/2008071615 Fedora/3.0.1-1.fc9 Firefox/3.0.1')]
columns=['id','breed', 'breedStr', 'price', 'color','location', 'age', 'zip', 'height', 'temp', 'warmblood', 'sold', 'soldhere', 'gender']
columns+=['forsale', 'forlease', 'registered', 'skills', 'desc']
cellColMap={32: 'for lease', 34: 'for sale', 2: 'zip', 36: 'price', 38: 'skills', 6: 'age', 8: 'gender', 10: 'height', 14: 'color', 20: 'warmblood', 22: 'temp', 4: 'breed', 26: 'registered', 40: 'desc'}
fromPickle=True
scrapedIds=[]
urlDict=OrderedDict([('Dressage',"http://www.dreamhorse.com/d/5/dressage/horses-for-sale.html"), ( "Jumping","http://www.dreamhorse.com/d/12/jumping/horses-for-sale.html"), ( "Eventing","http://www.dreamhorse.com/d/8/eventing/horses-for-sale.html"), ( "Hunter","http://www.dreamhorse.com/d/11/hunter/horses-for-sale.html"), ("Warmblood", "http://www.dreamhorse.com/list-horses/warmbloods.html"), ("AllAround", "http://www.dreamhorse.com/d/33/all-around/horses-for-sale.html")])
urlDict=OrderedDict([("GreatLakesUSA", "http://www.dreamhorse.com/a/horses-for-sale-in-the-great-lakes-usa-area.html") ,("MountainUSA", "http://www.dreamhorse.com/a/horses-for-sale-in-the-mountain-usa-area.html") ,("NortheastUSA", "http://www.dreamhorse.com/a/horses-for-sale-in-the-northeast-usa-area.html") ,("NorthwestUSA", "http://www.dreamhorse.com/a/horses-for-sale-in-the-northwest-usa-area.html") ,("PrairieUSA", "http://www.dreamhorse.com/a/horses-for-sale-in-the-prairie-usa-area.html") ,("SoutheastUSA", "http://www.dreamhorse.com/a/horses-for-sale-in-the-southeast-usa-area.html") ,("SouthwestUSA", "http://www.dreamhorse.com/a/horses-for-sale-in-the-southwest-usa-area.html") ,("EasternCANADA", "http://www.dreamhorse.com/a/horses-for-sale-in-the-eastern-canada-area.html") ,("WesternCANADA", "http://www.dreamhorse.com/a/horses-for-sale-in-the-western-canada-area.html") ])
dataDir="/Users/jbrosamer/PonyPricer/BatchArea/"
scrapedIdFile=dataDir+"ScrapedIds.p"
pbar=ProgressBar()
if fromPickle and os.path.isfile(scrapedIdFile):
	scrapedIds=sorted(list(pickle.load(open(scrapedIdFile, 'rb'))))

def dateStrToAge(cellTxt):
	"""
		Parse date string from web page and convert to years old
	"""
	try:
		bd=datetime.strptime(cellTxt.split("\n")[0],"%b %Y")
	except:
		bd=datetime.strptime("JAN "+cellTxt.split("\n")[0],"%b %Y")
	age=(datetime.today()-bd).days/365.
	return age

def heightStrToIn(cellTxt):
	"""
		Take height table cell with hh or in and convert to float inches
	"""
	heightStr=cellTxt.split("\n")[1]
	if "hh" in heightStr:
		hhHeight=heightStr.split(" ")[0].split(".")
		inHeight=float(hhHeight[0])*4+float(hhHeight[1])
	else:
		inHeight=float(heightStr.split(" ")[0])
	return heightStr

def IdsFromKey(key, excludeScraped=True):
	"""
		Take ad region and load all pages to scrape ids values. Each ad will be scraped by ad later.
	"""
	page = br.open(urlDict[key])
	html = page.read()
	ids=[]
	while html != "":
		ids+=extractIds(html)
		html=getNextPage(br)
		print "Found %s %i ids"%(key, len(ids))
	print "Done with len(ids)",len(ids)
	keyIds=set(ids)
	if excludeScraped and os.path.isfile(scrapedIdFile):
		scrapedIds=sorted(list(pickle.load(open(scrapedIdFile, 'rb'))))
	else:
		scrapedIds=[]
	keyIds=sorted(list(set(keyIds-set(scrapedIds))))
	scrapedIds=set(scrapedIds+keyIds)
	pickle.dump(keyIds, open(dataDir+"%sIds.p"%key, 'wb'))
	pickle.dump(scrapedIds, open(scrapedIdFile, 'wb'))
	return keyIds
	


def scrapeSearch(key, batchSize=1000, batchStart=0):
	"""
		Open a pickle file and scrape ads with all of the ids in that file
	"""

	ids=list(sorted(pickle.load(open(dataDir+"%sIds.p"%key, "rb"))))
	print "Loaded %i ids from %s"%(len(ids), dataDir+"%sIds.p"%key)
	allDf=[]
	for x in range(batchStart, len(ids), batchSize):
		df=pd.DataFrame(columns=columns, index=ids)
		startTime = datetime.now()
		batchIds=ids[x:x+batchSize]
		badIds=open(dataDir+"%sBadIds%iTo%i.txt"%(key,batchStart, batchSize+batchStart), 'wb')
		print "Start scraping %i to %i len(%i)"%(x, x+batchSize, len(batchIds))
		for n, i in enumerate(batchIds):
			try:
				srs=scrapeAd(i)
				df.loc[i]=scrapeAd(i)
			except Exception as e:
				print "BadId",i," exception ",str(e)
				print '-'*60
				traceback.print_exc(file=sys.stdout)
				print '-'*60
				badIds.write("%i\n"%i)
		try:
			print "Writing ",dataDir+"%sId%iTo%i.p"%(key,x, batchSize+x)
			pickle.dump(df, open(dataDir+"%sId%iTo%i.p"%(key,batchStart, x, batchSize+x), "wb"))
			df.to_csv(dataDir+"%sId%iTo%i.csv"%(key,x, batchSize+x))
		except Exception as e:
			badIds.write(str(e)+"\n")
		badIds.close()
		allDf+=[df]
	elapsedSec=(datetime.now()-startTime).seconds
	print "Time to scrape %i ids: %f minutes"%(len(ids), elapsedSec/60.)
	allDf=pd.concat(allDf)
	allDf = allDf.reset_index().drop('index', axis = 1)
	pickle.dump(allDf, open(dataDir+"All%sAds.p"%(key), "wb"))
	
	
    
    
def getNextPage(br):
	"""
		Go to next page with mechanize and return html
	"""
	for nr, form in enumerate(br.forms()):
		if "http://www.dreamhorse.com/show_list_pages.php"==form.action:
			for control in form.controls:
				if "Next" in repr(control.value):
						br.select_form(nr=nr)
						response = br.submit()
						content = response.read()
						return content
	return ""

def extractIds(html):
	"""
		Get ids of all ads listed on a page
	"""
	soup = BeautifulSoup(html, "html.parser")
	ids=list()
	for input_el in soup.findAll('input'):
		if input_el.has_attr('name') and (input_el['name']=="form_horse_id"):
			ids.append(int(input_el['value']))
	return ids


def scrapeAd(id):
	"""
		Take an id and scrape for all the features in adDict. Return a dataframe.
	"""
	page = br.open("http://www.dreamhorse.com/ad/%i.html"%id)
	html = page.read()
	soup = BeautifulSoup(html, "html.parser")
	tables=soup.findAll(name="table")
	adDict=dict.fromkeys(columns)
	adDict['id']=id
	cells=tables[3].findAll("td")
	soldCell=soup.findAll("td", class_="navy", limit=2)[-1]
	if "SALE PEND" in soldCell.text or "SOLD" in soldCell.text:
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
				zipStr=str(cellTxt.split("\n")[1].split(" ")[-1])
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
	return pd.Series(adDict)


if __name__ == "__main__":
	#IdsFromKey("Warmblood", excludeScraped=False)
	for k in urlDict.keys()[-3:]:
	 	scrapeSearch(k)
