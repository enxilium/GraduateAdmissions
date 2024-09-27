import tensorflow as tf
from tensorflow import keras
import pandas as pd
import numpy as np
from flask import Flask, render_template, request, redirect, url_for, session
import os
import pymupdf
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)
app.secret_key = os.getenv('SESSION_SECRET')

universityRatings = {
    "amsterdam": 3.5,
    "anu": 3.3,
    "berkeley": 4.8,
    "ucla": 4.7,
    "ucsd": 4.6,
    "caltech": 4.9,
    "cambridge": 4.9,
    "cape": 2.0,
    "columbia": 4.7,
    "copenhagen": 3.4,
    "cornell": 4.6,
    "duke": 4.5,
    "edinburgh": 3.4,
    "eth": 4.8,
    "harvard": 4.9,
    "hebrew": 3.3,
    "helsinki": 3.2,
    "hongkong": 3.4,
    "imperial": 4.7,
    "kaist": 3.5,
    "kings": 3.4,
    "kyoto": 3.5,
    "leiden": 3.3,
    "manchester": 3.4,
    "mcgill": 3.5,
    "melbourne": 4.6,
    "mit": 4.9,
    "monash": 3.4,
    "munich": 3.5,
    "michigan": 4.6,
    "nus": 4.7,
    "ntu": 4.6,
    "oxford": 4.9,
    "peking": 3.5,
    "princeton": 4.8,
    "queensland": 3.4,
    "seoul": 3.5,
    "stanford": 4.9,
    "sydney": 3.5,
    "tokyo": 4.6,
    "toronto": 4.6,
    "tsinghua": 4.7,
    "turing": 3.3,
    "ucl": 4.7,
    "uchicago": 4.8,
    "upenn": 4.7,
    "warwick": 3.4,
    "yale": 4.8,
    "zurich": 3.5
}

@app.route('/', methods=['GET', 'POST'])
def index():
    new_model = tf.keras.models.load_model('model/model.keras')
    genai.configure(api_key=os.getenv('GEMINI_API_KEY'))
    model = genai.GenerativeModel("gemini-1.5-flash")

    # GRE, TOEFL, University Rating, SOP, LOR, CGPA, Research
    if request.method == 'POST':
        GRE = int(request.form['gre'])
        TOEFL = int(request.form['toefl'])
        university = universityRatings[request.form['university']]
        SOP = request.files['sop']
        LOR = request.files['lor']
        CGPA = float(request.form['cgpa'])
        research = float(request.form['research'])
        
        SOP.save('tmp/sop.pdf')
        LOR.save('tmp/lor.pdf')

        sopFile = pymupdf.open('tmp/sop.pdf')
        sopContent = ""
        for page in sopFile:
            sopContent += page.get_text()
        sopFile.close()

        os.remove('tmp/sop.pdf')

        lorFile = pymupdf.open('tmp/lor.pdf')
        lorContent = ""
        for page in lorFile:
            lorContent += page.get_text()
        
        lorFile.close()
        
        os.remove('tmp/lor.pdf')

        sopRating = model.generate_content('Rate the following statement of purpose for a graduate program on a scale of 1 to 5, and return only a number with NO other text, rounded to one decimal place: "' + sopContent + '"').text
        lorRating = model.generate_content('Rate the following letter of recommendation for a graduate program on a scale of 1 to 5, and return only a number with NO other text, rounded to one decimal place: "' + lorContent + '"').text
        data = np.array([[GRE, TOEFL, university, float(sopRating), float(lorRating), CGPA, research]])
        
        result = round(float(new_model.predict(data)), 2)

        if result < 0:
            result = 0.00
        elif result > 100:
            result = 100.00

        session['result'] = result
        session['GRE'] = GRE
        session['TOEFL'] = TOEFL
        session['university'] = university
        session['sop'] = sopRating
        session['lor'] = lorRating
        session['CGPA'] = CGPA
        session['research'] = "Yes" if research == 1 else "No"
        
        return redirect(url_for('results'))
    else:
        pass

    return render_template('index.html')

@app.route('/results')
def results():
    result = session.get('result', None)
    return render_template('results.html', result = result, GRE = session.get('GRE', None), TOEFL = session.get('TOEFL', None), university = session.get('university', None), sop = session.get('sop', None), lor = session.get('lor', None), CGPA = session.get('CGPA', None), research = session.get('research', None))

if __name__ == '__main__':
    app.run(debug=True)