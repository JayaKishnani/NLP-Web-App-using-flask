#importing libraries
import tensorflow as tf
import keras
from keras.models import load_model
import tensorflow_addons as tfa
from keras_preprocessing.sequence import pad_sequences
from nltk.tokenize import word_tokenize
from flask import Flask, render_template, flash, request, url_for, send_file, Response
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
import os
import io
import nltk
from nltk.tag import StanfordNERTagger
from nltk.tokenize import word_tokenize
# from werkzeug import secure_filename

#Setting environment
os.environ['JAVAHOME'] =  "C:\Program Files\Java\jdk-18.0.2"
nltk.download('punkt')

count = []
# resultant=[]
app = Flask(__name__)
app.secret_key = "blah"

@app.route('/', methods=['GET', 'POST'])
def home():
    return render_template('index.html', data=[{'choose': 'Analyze Sentiment'}, {'choose': 'Predict using custom NER'}, {'choose': 'Predict using stanford NER'}])

@app.route('/predict', methods=['GET','POST'])
def predict():
    if request.method=='POST':
        # print("In predict")
        # print(request)
        # print(request.files)
        uploaded_file = request.files.get('file')
        
        
        if uploaded_file.filename != '':
            uploaded_file.save(uploaded_file.filename)
            # print("File uploaded successfully ", uploaded_file.filename)
            flash("File uploaded successfully")
        else:
            flash("Upload a text file")
            return render_template("index.html")
            
    with open(uploaded_file.filename, 'r') as f:
        text = f.read()
        # print("Text is: ", text)
        Inputdata = request.form.get("inputdata")
        if Inputdata == "Analyze Sentiment":
            results=[]
            lines = text.split('.')
            lines.pop()
            count1, count2, count3 = 0,0,0
            for line in lines:
                sa = SentimentIntensityAnalyzer()
                sentiment_dict = sa.polarity_scores(line)
                if sentiment_dict['compound'] >= 0.05 :
                    string = " Positive "
                    count1+=1
                elif sentiment_dict['compound'] <= - 0.05 :
                    string = " Negative "
                    count2+=1
                else :
                    string = " Neutral "
                    count3+=1
                results.append(string)
            count.append(count1)
            count.append(count2)
            count.append(count3)
            
            global resultant
            resultant=results.copy()
            return render_template('index.html', results = results)

        elif Inputdata == "Predict using stanford NER":
            results=[]
            st = StanfordNERTagger('D:\stanford-ner-4.2.0\stanford-ner-2020-11-17\classifiers\english.all.3class.distsim.crf.ser\english.all.3class.distsim.crf.ser',
                          'D:\stanford-ner-4.2.0\stanford-ner-2020-11-17\stanford-ner.jar',
                          encoding='utf-8')
            lines = text.split('.')
            for line in lines:
                tokenized_line = word_tokenize(line)
                classified_line = st.tag(tokenized_line)
                results.append(classified_line)
            resultant=results.copy()
            return render_template('index.html', results = results)
        
        elif Inputdata == "Predict using custom NER":
            results=[]
            model = load_model('ner_model.h5', custom_objects={"crf":tfa.layers.CRF})
            tags = ['B-gpe', 'I-org', 'I-gpe', 'I-tim', 'O', 'I-art', 'I-nat', 'I-eve', 'I-geo', 'B-art', 'B-per', 'B-org', 'B-eve', 'B-geo', 'B-nat', 'I-per', 'B-tim']
    
            words = list(set(text.split())) 
            word2idx = {w: i + 2 for i, w in enumerate(words)}
            word2idx["UNK"] = 1 # Unknown words
            word2idx["PAD"] = 0 # Padding
            idx2word = {i: w for w, i in word2idx.items()}
            max_len = 75
            x = {}
    
            pad_line = pad_sequences(sequences=[[word2idx.get(w, 0)] for w in word_tokenize(text)], padding="post", value=0, maxlen=max_len)
            for i in range(len(pad_line)):
                p = model.predict(np.array([pad_line[i]]))
                p = np.argmax(p, axis=-1)
                for w, pred in zip(pad_line[i], p[0]):
                    x[w] = tags[pred]
            y = {}
            for num in list(x.keys()):
                y[idx2word[num]] = x[num]
            results.append(y)
            resultant=results.copy()
            return render_template('index.html', results = results)

@app.route('/plot', methods=['GET','POST'])
def plot():
    fig = Figure()
    fig.add_subplot(111).pie(count)
    output = io.BytesIO()
    FigureCanvas(fig).print_png(output)
    return Response(output.getvalue(), mimetype='image/png')

@app.route('/download')
def download_file():
    print(resultant)
    with open('results.txt', 'w') as f:
        for items in resultant:
            f.write('%s\n' %items)
    return send_file('./results.txt', mimetype="text", as_attachment=True, download_name='results.txt')


if __name__=='__main__':
    app.run(debug=True)