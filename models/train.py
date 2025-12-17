import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import joblib

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
DATA_PATH = os.path.join(BASE_DIR, 'data', 'jobs.csv')
# Save artifacts under models/artifacts to match loader in utils/nlp.py
ARTIFACTS_DIR = os.path.join(BASE_DIR, 'models', 'artifacts')
os.makedirs(ARTIFACTS_DIR, exist_ok=True)


def load_data():
    df = pd.read_csv(DATA_PATH)
    # expect columns: title, description, skills
    df['text'] = df['title'].fillna('') + ' ' + df['description'].fillna('') + ' ' + df['skills'].fillna('')
    return df


def train():
    df = load_data()
    X = df['text'].tolist()
    y = df['title'].tolist()
    vect = TfidfVectorizer(max_features=5000, ngram_range=(1,2))
    Xv = vect.fit_transform(X)
    clf = LogisticRegression(max_iter=500)
    clf.fit(Xv, y)

    # save artifacts
    joblib.dump(vect, os.path.join(ARTIFACTS_DIR, 'tfidf_vectorizer.joblib'))
    joblib.dump(clf, os.path.join(ARTIFACTS_DIR, 'clf.joblib'))

    # save jobs meta for simple skill matching
    jobs_meta = df[['title', 'skills']].rename(columns={'title':'title','skills':'skills'}).to_dict(orient='records')
    joblib.dump(jobs_meta, os.path.join(ARTIFACTS_DIR, 'jobs_meta.joblib'))
    print('Training complete. Artifacts saved to', ARTIFACTS_DIR)

if __name__ == '__main__':
    train()
