import os
from flask import Flask, request, render_template, redirect, url_for
from werkzeug.utils import secure_filename
from utils.extract import extract_text_from_file
from utils.nlp import analyze_resume

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'pdf', 'docx'}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return redirect(url_for('index'))
    file = request.files['file']
    if file.filename == '':
        return redirect(url_for('index'))
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(path)
        text = extract_text_from_file(path)
        result = analyze_resume(text)
        return render_template('result.html', result=result)
    return redirect(url_for('index'))

if __name__ == '__main__':
    app.run(host="0.0.0.0",debug=True)
app = app