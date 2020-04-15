from flask import Flask,render_template,url_for,request,send_file
from flask_bootstrap import Bootstrap 
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
import nltk

from nltk.tokenize import sent_tokenize

nltk.download('punkt')
nltk.download('stopwords')


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import statsmodels.api as sm
import math
# Check for the VIF values of the feature variables. 
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.metrics import r2_score 
import io

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

from jinja2 import nodes
from jinja2.ext import Extension
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.probability import FreqDist
import re
from sklearn import svm

app = Flask(__name__)
app.jinja_env.add_extension('jinja2.ext.loopcontrols')
app.config.from_object(__name__)
Bootstrap(app)


@app.route('/')

def index():
    df = pd.read_csv('data/grad.csv')
    X = df.drop(['state','admitted_or_not','interest_in_sports','sr_no','full_name','gender','phone_numbers',],axis=1)
    Y= df['admitted_or_not']
    X.shape[0]
    Y.shape[0]
    X.info()
    X_train, X_test, y_train, y_test = train_test_split(X, Y,random_state = 45, test_size = 0.25)
   
    classify_decisionTree = DecisionTreeClassifier(criterion = 'entropy')
    classify_RandomForest = RandomForestClassifier(criterion = 'entropy')
    classify_Addboost = AdaBoostClassifier(n_estimators=50,learning_rate=1)
    classify_Svm = svm.SVC()
    model_dectree = classify_decisionTree.fit(X_train, y_train)
    model_randomForest = classify_RandomForest.fit(X_train, y_train)
    model_adaboost= classify_Addboost.fit(X_train, y_train)
    model_svm = classify_Svm.fit(X_train,y_train)
    y_pred_dectree =  model_dectree.predict(X_test)
    y_pred_randomforest = model_randomForest.predict(X_test)
    y_pred_Addboost = model_adaboost.predict(X_test)
    y_pred_svm = model_svm.predict(X_test)  
    dectree_accuracy = accuracy_score(y_true=y_test, y_pred=y_pred_dectree)
    random_accuracy = accuracy_score(y_true=y_test, y_pred=y_pred_randomforest)
    add_accuracy = accuracy_score(y_true=y_test, y_pred=y_pred_Addboost)
    svm_accuracy = accuracy_score(y_true=y_test, y_pred=y_pred_svm)
    classifiers = ['Decision Tree', 'Random Forest', 'AdaBoost' , 'SVC']
    accuracy = np.array([dectree_accuracy, random_accuracy, add_accuracy, svm_accuracy])
    max_acc = np.argmax(accuracy)
    if(classifiers[max_acc] == 'AdaBoost'):
        y_pred = model_adaboost.predict(X_test)
    elif(classifiers[max_acc] == 'Random Forest'):
        y_pred =  model_randomForest.predict(X_test)
    y_train_pred_final = pd.DataFrame({'admitted_or_not':y_pred, 'admitted_or_not1':y_test})
    y_train_pred_final1 = y_train_pred_final
    y_train_pred_final1['name'] = df.full_name
    y_train_pred_final1['Gre_score'] = df.gre_score
    y_train_pred_final1['toefl'] = df.toefl_score
    y_train_pred_final1['sop'] = df.sop_score
    y_train_pred_final1['lor'] = df.lor_score
    y_train_pred_final1['cgpa'] = df.cgpa
    y_train_pred_final1['research_years'] = df.research_years
    y_train_pred_final1['fin_aid'] = df.plan_fa
    y_train_pred_final1['first_gen'] = df.not_a_first_gen_applicant
    return render_template('template.html',prediction = y_train_pred_final1)

@app.route('/toeflplot')
def toeflplot():
    bytes_obj = do_plot_for_toefl()
    
    return send_file(bytes_obj,
                     attachment_filename='plot_for_toefl.png',
                     mimetype='image/png')

def do_plot_for_toefl():
    df=pd.read_csv("data/grad.csv")
    plt.figure(figsize=(8,8))
    plt.subplot(1,1,1)
    plt.title('TOEFL Total Score')
    sns.boxplot(df.toefl_score)  
    bytes_image = io.BytesIO()
    plt.savefig(bytes_image, format='png')
    bytes_image.seek(0)
    return bytes_image

@app.route('/conmatrix')
def conmatrix():
    df = pd.read_csv('data/grad.csv')
    X = df.drop(['state','admitted_or_not','interest_in_sports','sr_no','full_name','gender','phone_numbers',],axis=1)
    Y= df['admitted_or_not']
    X.shape[0]
    Y.shape[0]
    X.info()
    X_train, X_test, y_train, y_test = train_test_split(X, Y,random_state = 45, test_size = 0.25)
    classify_RandomForest = RandomForestClassifier(criterion = 'entropy')
    model_randomForest = classify_RandomForest.fit(X_train, y_train)
    y_pred_randomforest = model_randomForest.predict(X_test)
    cm = confusion_matrix(y_test,y_pred_randomforest)
    plt.clf()
    cmap = plt.get_cmap('Blues')
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    classNames = ['NO','YES']
    plt.title('')
    plt.ylabel('True Class Values')
    plt.xlabel('Predicted Class Values')
    tick_marks = np.arange(len(classNames))
    plt.xticks(tick_marks, classNames, rotation=45)
    plt.yticks(tick_marks, classNames)
    s = [['',''], ['', '']]
    for i in range(2):
        for j in range(2):
            plt.text(j,i, str(s[i][j])+"  "+str(cm[i][j]))
    bytes_image = io.BytesIO()
    plt.savefig(bytes_image, format='png')
    bytes_image.seek(0)
    return send_file(bytes_image,
                     attachment_filename='plot_for_con.png',
                     mimetype='image/png')


@app.route('/greplot')
def greplot():
    bytes_obj = do_plot_for_gre()
    
    return send_file(bytes_obj,
                     attachment_filename='plot_for_gre.png',
                     mimetype='image/png')

def do_plot_for_gre():
    df=pd.read_csv("data/grad.csv")
    plt.figure(figsize=(8,8))
    plt.subplot(1,1,1)
    plt.title('GRE Plot')
    sns.distplot(df.gre_score)
    bytes_image = io.BytesIO()
    plt.savefig(bytes_image, format='png')
    bytes_image.seek(0)
    return bytes_image

@app.route('/newapplication')
def newapplication():
    return render_template('newapplication.html')


@app.route('/newapplication', methods=['POST'])
def newapplicationPost():
    gre_score = request.form['gre_score']
    toefl_score = request.form['toefl_score']
    high_school_gpa = request.form['gpa']
    research = request.form['research']
    fin_aid = request.form['fin_aid']
    first_gen = request.form['first_gen']
    sop = request.form['sop_score']
    sop_score= calculate_sop_score(sop)
    lor_score = request.form['lor_score']
    
    df = pd.read_csv('data/grad.csv')
    X = df.drop(['state','admitted_or_not','interest_in_sports','sr_no','full_name','gender','phone_numbers',],axis=1)
    Y= df['admitted_or_not']
    X.shape[0]
    Y.shape[0]
    X.info()
    X_train, X_test, y_train, y_test = train_test_split(X, Y,random_state = 45, test_size = 0.25)
    classify_Addboost = AdaBoostClassifier(n_estimators=50,learning_rate=1)
    classify_Addboost.fit(X_train, y_train)
    if fin_aid == 'on':
        fin_aid=1
        
    else:
        fin_aid=1

    if research == 'on':
        research=1
        
    else:
        research=1

    if first_gen == 'on':
        first_gen=1
        
    else:
        first_gen=1

    array = [[gre_score,toefl_score,5,sop_score,lor_score,high_school_gpa,first_gen,research,fin_aid]]
    y_pred_Addboost = classify_Addboost.predict(array)
    return render_template('newapplication.html',predicted_value= y_pred_Addboost,sopScore=sop_score)

def calculate_sop_score(sop):
    text=sop.lower()    
    reg = "a-zA-Z"
    text = re.findall(r'\w+', text) 
    stop_words=set(stopwords.words("english"))
    filtered_sent=[]
    for w in text:
        if w not in stop_words:
            filtered_sent.append(w)
    for words in text:
        if len(words) == 1:
            text.remove(words)
    good_word = []
    good_word = ["extracurricular","won","project","goal","academics","hackathon","potential","leader","skills","ability",
              "strengths","sports","dependable","consistence","persistence","motivation","good","best","right","sensational"
              "character","contribution","accomplishments","initiative","proactive","diligent","hardwork","communication",
              "exceptional","fit","one","science","curious", "presented", "paper","IEEE","IIT","Indian Institute of Technology",
             "revolution","specialized knowledge","challenging myself", "passion","desire","elite","competitive","extra-curricular",
             "hard work","excelled","research paper","delivered lectures", "perseverance","teaching"," experience","interest","commitment",
              "designing","promise","innovation","growth","top","high","program","innovation","responsibility","great","purpose","global",
            "market","abilities","organising","organise","efficiently"]
    good_words= [x.lower() for x in good_word]
    intersecting_words = intersection(text, good_words)
    per = len(intersecting_words)/len(good_words) * 100
    if per == 0:
        return 1
    elif per > 60:
        return(5)
    elif per > 50:
        return(4.5)
    elif per > 40:
        return(4)
    elif per > 35:
        return(3.5)
    elif per > 30:
        return(3)
    elif per > 20:
        return(2.5)
    elif per > 10:
        return(2)
    elif per > 8:
        return(1.5)
    else :
        return(1)
    
def intersection(text, good_words):
    return list(set(text) & set(good_words))

if __name__ == '__main__':
	app.run(debug=True)