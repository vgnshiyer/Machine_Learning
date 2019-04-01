# Machine-Learning
### This repository consists of different machine learning case studies that I have solved.

### * Amazon Case Study

Amazon Fine Food Reviews dataset (https://www.kaggle.com/snap/amazon-fine-food-reviews). It is a classification problem. Machine Learning models used: K-nearest neighbours(KNN) classifier and logistic regression. The problem statement is to predict whether the given review is a positive review or a negative review (ie. score 0 (negative) or (positive) 1). I have used Bag-of-Words(BOW),TFIDF and Word-to-Vector(W2V) NLP techniques. If you want to see how Bag-of-Words(BOW),TFIDF and Word-to-Vector(W2V) work check BOW-TFIDF-W2V-DUMMY.ipynb. EDA_Cleaning_Preprocessing.ipynb contains cleaning, text preprocessing, exploratory data analysis (EDA) in detail.

### * Customer Segments Dataset

Our client is an online retailer based in the UK. They sell all-occasion gifts, and many of their customers are wholesalers.

- Most of their customers are from the UK, but they have a small percent of customers from other countries.
- They want to create groups of these international customers based on their previous purchase patterns.
- Their goal is to provide more tailored services and improve the way they market to these international customers.

The retailer has hired us to help them create customer clusters, a.k.a "customer segments" through a data-driven approach.They've provided us a dataset of past purchase data at the transaction level.Our task is to build a clustering model using that dataset.Our clustering model should factor in both aggregate sales patterns and specific items purchased.

### * MNIST hand written digit classification

The MNIST Database(Modified National Institute of Standards and Technology database) is a large database of handwritten digits that is widely used for training and testing in machine learning. The MNIST contains 60000 training and 10000 test images. As MNIST is high dimemsion data, visualizing it may be challenging. Here I have used different dimensionality reduction techniques (PCA and T-SNE) which are techniques for reducing the dimension of high deimesion data while retaining most of its information.

 ### * Embedded ML Applications
 
 This directory consists of different machine learning projects deployed as a web based application using Flask( Web Framework ).
  -BlackFridaySale-Amount Predictor: Total amount predictor for the black friday sale( https://www.kaggle.com/mehdidag/black-friday).
  -Spam Detector: Spam Detector for emails.
  
 ### * Employee Retention
 
 Our client is the HR department at a large software company.

- They are rolling out a new initiative that they call "Proactive Retention."
- The idea is to use data to predict whether and employee is likely to leave.
- Once these employees are identified, HR can be more proactive in reaching out to them before it's too late.
- For this initiative, they only care about permanent (non-temp) employees.
The HR department has hired us as data science consultants. They want to supplement their exit interviews with a more proactive approach.

### * Facebook Recuiting Competition

Link prediction on Facebook's social network.

### * Linear Regression implementation

Here I have used the SKLEARN Boston house-price dataset to implement Linear Regression Model to predic the price. I have used gradient descent optimizing algorithm on linear regression model.

### * Real Estate Case Study

Our client is a large Real Estate Investment Trust (REIT).

- They invest in houses, apartments, and condos(complex of buildings) within a small county in New York state.
- As part of their business, they try to predict the fair transaction price of a property before it's sold.
- They do so to calibrate their internal pricing models and keep a pulse on the market.
Our task is to build a real-estate pricing model using that dataset.

### * Used Cars Dataset

- This is basically a website which holds the database of all the cars registered for selling.
- Over 370000 used cars scraped with Scrapy from Ebay-Kleinanzeigen.
- The content of the data is in german, so one has to translate it first if one can not speak german.
Our task is to build a pricing model based on the given dataset.
