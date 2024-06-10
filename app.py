from flask import Flask, render_template, request
import pandas as pd
import string

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB

from PIL import Image
from torchvision import models, transforms
import torch

# nltk.download('punkt')
# nltk.download('stopwords')
sw=nltk.corpus.stopwords.words("english")

#function to clean and transform the user input which is in raw format
def transform_text(text):
    text=text.lower()
    text=word_tokenize(text)
    y=[]
    for i in text:
        if i.isalnum():
            y.append(i)
    text=y[:]
    y.clear()
    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)
    text=y[:]
    y.clear()
    ps=PorterStemmer()
    for i in text:
        y.append(ps.stem(i))
    return " ".join(y)

#Spam Detection Prediction
tfidf1=TfidfVectorizer(stop_words=sw,max_features=20)
def transform1(txt1):
    txt2=tfidf1.fit_transform(txt1)
    return txt2.toarray()

df1=pd.read_csv("Spam Detection.csv")
df1.columns=["Label","Text"]
x=transform1(df1["Text"])
y=df1["Label"]
x_train1,x_test1,y_train1,y_test1=train_test_split(x,y,test_size=0.1,random_state=0)
model1=LogisticRegression()
model1.fit(x_train1,y_train1)


#Sentiment Analysis Prediction 
tfidf2=TfidfVectorizer(stop_words=sw,max_features=20)
def transform2(txt1):
    txt2=tfidf2.fit_transform(txt1)
    return txt2.toarray()

df2=pd.read_csv("Sentiment Analysis.csv")
df2.columns=["Text","Label"]
x=transform2(df2["Text"])
y=df2["Label"]
x_train2,x_test2,y_train2,y_test2=train_test_split(x,y,test_size=0.1,random_state=0)
model2=LogisticRegression()
model2.fit(x_train2,y_train2)

#Stress Detection Prediction
tfidf3=TfidfVectorizer(stop_words=sw,max_features=20)
def transform3(txt1):
    txt2=tfidf3.fit_transform(txt1)
    return txt2.toarray()

df3=pd.read_csv("Stress Detection.csv")
df3=df3.drop(["subreddit","post_id","sentence_range","syntax_fk_grade"],axis=1)
df3.columns=["Text","Sentiment","Stress Level"]
x=transform3(df3["Text"])
y=df3["Stress Level"].to_numpy()
x_train3,x_test3,y_train3,y_test3=train_test_split(x,y,test_size=0.1,random_state=0)
model3=DecisionTreeRegressor(max_leaf_nodes=2000)
model3.fit(x_train3,y_train3)

#Hate & Offensive Content Prediction
tfidf4=TfidfVectorizer(stop_words=sw,max_features=20)
def transform4(txt1):
    txt2=tfidf4.fit_transform(txt1)
    return txt2.toarray()

df4=pd.read_csv("Hate Content Detection.csv")
df4=df4.drop(["Unnamed: 0","count","neither"],axis=1)
df4.columns=["Hate Level","Offensive Level","Class Level","Text"]
x=transform4(df4["Text"])
y=df4["Class Level"]
x_train4,x_test4,y_train4,y_test4=train_test_split(x,y,test_size=0.1,random_state=0)
model4=RandomForestClassifier()
model4.fit(x_train4,y_train4)

#Sarcasm Detection Prediction
tfidf5=TfidfVectorizer(stop_words=sw,max_features=20)
def transform5(txt1):
    txt2=tfidf5.fit_transform(txt1)
    return txt2.toarray()

df5=pd.read_csv("Sarcasm Detection.csv")
df5.columns=["Text","Label"]
x=transform5(df5["Text"])
y=df5["Label"]
x_train5,x_test5,y_train5,y_test5=train_test_split(x,y,test_size=0.1,random_state=0)
model5=LogisticRegression()
model5.fit(x_train5,y_train5) 

data = pd.read_json('Sarcasm_Headlines_Dataset.json', lines=True)
data = data[['headline', 'is_sarcastic']]
vectorizer = CountVectorizer(stop_words='english', max_features=5000)
X = vectorizer.fit_transform(data['headline'])
X_train, X_test, y_train, y_test = train_test_split(X, data['is_sarcastic'], test_size=0.2, random_state=42)
model = MultinomialNB()
model.fit(X_train, y_train)
def checkSarcasm(msg):
    msg_vectorized = vectorizer.transform([msg])
    is_sarcastic = model.predict(msg_vectorized)
    return is_sarcastic


app = Flask(__name__)
@app.route('/')
def index():
    params = {'title':'Home','result':"default"}
    return render_template('index.html',params=params)

@app.route('/text')
def text():
    params = {'title':'Text Analysis','result':"default"}
    return render_template('text.html',params=params)

@app.route('/image')
def image():
    params = {'title':'Image Analysis','result':"default"}
    return render_template('image.html',params=params)

@app.route('/analyzeText',methods=['GET','POST'])
def text_analysis():
    if request.method == 'POST':
        result = "default"
        text = request.form.get('textinput')
        choice = request.form.get('userChoice')
        transformedText=transform_text(text)
        if(choice=="spam"):
            vector_sent=tfidf1.transform([transformedText])
            prediction=model1.predict(vector_sent)[0]
            if prediction=="spam":
                result = "It's a Spam Text!!🤨"
            elif prediction=="ham":
                result = "It's a Ham Text!!😎"
            params = {'title':'Text Analysis','result':result}
        elif(choice=="sentiment"):
            vector_sent=tfidf2.transform([transformedText])
            prediction=model2.predict(vector_sent)[0]
            if prediction==0:
                result = "😣Negetive Text!!😣"
            elif prediction==1:
                result = "😄Positive Text!!😄"
            params = {'title':'Text Analysis','result':result}
        elif(choice=="stress"):
            vector_sent=tfidf3.transform([transformedText])
            prediction=model3.predict(vector_sent)[0]
            if prediction>=0:
                result = "Stressful Text!!"
            elif prediction<0:
                result = "Not A Stressful Text!!"
            params = {'title':'Text Analysis','result':result}
        elif(choice=="offense"):
            vector_sent=tfidf4.transform([transformedText])
            prediction=model4.predict(vector_sent)[0]
            if prediction==0:
                result = "Highly Offensive Text!!😠"
            elif prediction==1:
                result = "Offensive Text!!🫤"
            elif prediction==2:
                result = "Non Offensive Text!!😌"
            params = {'title':'Text Analysis','result':result}
        elif(choice=="sarcasm"):
            # vector_sent=tfidf5.transform([transformedText])
            # prediction=model5.predict(vector_sent)[0]
            # if prediction==1:
            #     result = "It's a Sarcastic Text!!🤪 LOL"
            # elif prediction==0:
            #     result = "It's a Non Sarcastic Text!!🙄"
            isSarcastic = checkSarcasm(text)
            if isSarcastic:
                result = "It's a Sarcastic Text!!🤪 LOL"
            else:
                result = "It's a Non Sarcastic Text!!🙄"
            params = {'title':'Text Analysis','result':result}
    return params['result']


def predict(image):
    
    # create a ResNet model
    resnet = models.resnet101(pretrained = True)

    # transform the input image through resizing, normalization
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean = [0.485, 0.456, 0.406],
            std = [0.229, 0.224, 0.225]
            )])

    # load the image, pre-process it, and make predictions
    img = Image.open(image)
    batch_t = torch.unsqueeze(transform(img), 0)
    resnet.eval()
    out = resnet(batch_t)

    with open('imagenet_classes.txt') as f:
        classes = [line.strip() for line in f.readlines()]

    # return the top 5 predictions ranked by highest probabilities
    prob = torch.nn.functional.softmax(out, dim = 1)[0] * 100
    _, indices = torch.sort(out, descending = True)
    return [(classes[idx], prob[idx].item()) for idx in indices[0][:5]]

@app.route('/analyzeImage',methods=['GET','POST'])
def image_analysis():
    if request.method == 'POST':
        result = "default"
        image = request.files['sample_image']
        result = predict(image)
        params = {'title':'Image Analysis','result':result}
    return params

if __name__ == '__main__':
    app.run(debug=False)
