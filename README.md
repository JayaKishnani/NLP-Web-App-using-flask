# NLP-Web-App-using-flask

### Aim:
1) To upload a text file from the user by creating UI in **flask** and then select an option to implement predict sentiment(and plot it using piechart) or perform named entity recognition and then save the results.
2) NER model is created using custom **Bi-directional LSTM CRF model** and another method using **stanford NER taggers**.<br>
*(LSTM - Long short term memory and CRF- Conditional Random field)*
3) Sentiment analysis is done using **vader-sentiment analysis vocabulary** to predict neutral, positive and negative sentiments.

### File structure:
1) <code>template/index.html</code> and <code>static/css/main.css</code> contains html and css files respectively for the UI.
2) <code>gmdb_nermodel.ipynb</code> contains the custom trained NER model using Bi-directional LSTM CRF model.
3) <code>ner_model.h5</code> contains saved custom trained NER model using Bi-directional LSTM CRF model.
4) <code> app.py</code> contains the flask backend to integrate our frontend UI with ML models.
5) <code> requirements.txt</code> contains the dependencies to run the program.

### How to run?
1) Clone the repository.<br>
<code> git clone https://github.com/JayaKishnani/NLP-Web-App-using-flask.git </code>
2) Set the virtual environment using anaconda and install <code> requirements.txt</code>.<br>
<code>conda create -n myenv python=3.10</code><br>
<code>pip install -r requirements. txt</code><br>
3) Running <code> app.py</code> using **flask environment**.<br>
<code>python app.py</code>

### Another GUI
The above framework is also implemented using tkinter python library for creating GUI.<br>
https://github.com/JayaKishnani/NLP-app
