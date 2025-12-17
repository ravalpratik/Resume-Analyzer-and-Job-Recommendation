AI-Powered Smart Resume Analyzer and Job Recommendation System

Overview
- Flask-based web app to upload resumes (PDF/DOCX), extract text, identify skills/education/experience, predict suitable job roles and recommend missing skills.

Quick start
1. Create a virtual environment and activate it.

Windows (PowerShell):
```
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

2. Train model (creates `models/artifacts`):
```
python models/train.py
```

3. Run app:
```
python app.py
```

Project layout
- `app.py` - Flask app
- `models/train.py` - training script
- `utils/extract.py` - resume text extraction
- `utils/nlp.py` - NLP helpers and matching
- `data/jobs.csv` - sample job descriptions
- `data/skills.txt` - skills list
- `templates/` - HTML pages
- `static/` - CSS

Notes
- This is a starter implementation for a BCA AI course project. You can extend with BERT, richer NER, and a DB-backed admin panel.
