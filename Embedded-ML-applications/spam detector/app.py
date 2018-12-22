#import section
from flask import Flask,render_template,url_for,request
import numpy as np
import pandas as pd
import sqlite3
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.externals import joblib
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.naive_bayes import MultinomialNB

app = Flask(__name__)

@app.route('/')
def home():
	return render_template('home.html')

#prediction algorithm
@app.route('/predict', methods=['POST'])
def predict():
	#retreiving data
	conn = sqlite3.connect('email.sqlite')
	email= pd.read_sql_query("""SELECT * FROM emails""",conn)
	conn.close()
	#applying bag of words
	bow_vect=CountVectorizer(binary=True)
	bow = bow_vect.fit_transform(email['Cleaned_email'].values)
	X=bow
	y=email["class"].values
	#test and train data
	X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,shuffle=False)
	#finding optimal hyperparameter
	alpha= np.linspace(0.1,1,10)
	cv_scores=[]

	for a in alpha:
	    clf=MultinomialNB(alpha=a)
	    scores=cross_val_score(clf,X_train,y_train,cv=10,scoring='accuracy')
	    cv_scores.append(scores.mean())

	MSE=[1 - x for x in cv_scores]

	optimal_alpha= alpha[MSE.index(min(MSE))]
	#building the classifier
	clf=MultinomialNB(alpha=optimal_alpha)
	clf.fit(X_train,y_train)
	clf.score(X_test,y_test)
	#saving the model
	#model = open("naivebayes_spam_classifier.pkl","rb")
	#clf = joblib.load(model)

	if request.method == 'POST':
		email = request.form['email']
		data = [email]
		vect = bow_vect.transform(data).toarray()
		prediction = clf.predict(vect)

	return render_template('prediction.html',prediction=prediction)


if __name__ == '__main__':
	app.run(debug=True)