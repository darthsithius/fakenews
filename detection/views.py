from django.shortcuts import render
from django.http import HttpResponse
import pickle
from GooseExtract import *
from bs4 import BeautifulSoup
from urllib.request import urlopen
import os,sys
source={'occupydemocrats': '-10', 'buzzfeed': '-9', 'breitbart': '-9', 'infowars': '-10', 'yahoo': '-7', 'huffpost': '0', 'theblaze': '-6', 'foxnews': '2', 'abc': '3', 'msnbc': '1', 'drudgereport': '-6', 'nbc': '6', 'cnn': '0', 'cbs': '6', 'theatlantic': '6', 'usatoday': '7', 'nytimes': '8', 'time': '7', 'washingtonpost': '8', 'apnews': '7', 'politico': '8', 'latimes': '8', 'wsj': '8', 'theguardian': '9', 'PBS': '9', 'NPR': '9', 'reuters': '10', 'economist': '10', 'medium': '9', 'onion.com': '0', 'empire': '0'}

class prediction:
	def detecting_fake_news(self,variable):
		var = variable
		html = urlopen(var).read().decode('utf-8')
		soup = BeautifulSoup(html, features='lxml')
		a_title= soup.title.text
		print("You entered: " + str(var))
		t=0
		load_model = pickle.load(open('final_model.sav', 'rb'))
		url = Scrape(var)
		data = url.cleaned_text
		#print("data",data)
		prediction = load_model.predict([data])
		prob = load_model.predict_proba([data])
		HeaderBodyComp = Comparison(var)
		for i in source.keys():
			if i in var:
				t=int(source[i])/10
				break
			else:
				t=0.0
		print("Hello",HeaderBodyComp)
		#print(type(HeaderBodyComp))
		FinalScore = (prob[0][1] * 0.75) + (HeaderBodyComp * 0.20) + (t*0.05)
		#print(FinalScore)
		#print(type(FinalScore))
		if FinalScore>0.5:
			result="True"
		else:
			result="Fake"
		#print("Actual prediction=",prediction[0],prob[0][1])
		#return (print("The given statement is ",result),
		#    print("The truth probability score is ",FinalScore))
		return result,FinalScore,data

pre=prediction()


def index(request):
	if request.method=="POST":
		link=request.POST.get('url')
		pred,prob,restext=pre.detecting_fake_news(link)
		print("ARTICLE\n",restext)
		return render(request,"detection/result_new.html",{'pred':pred,'prob':prob,'restext':restext})
	
	return render(request,"detection/main_page.html")


def trying(request):
	return render(request,"detection/result_new.html")