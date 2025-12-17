import os
from pdfminer.high_level import extract_text
import docx2txt


def extract_text_from_file(path: str) -> str:
    ext = os.path.splitext(path)[1].lower()
    if ext == '.pdf':
        try:
            return extract_text(path)
        except Exception:
            return ''
    if ext == '.docx':
        try:
            return docx2txt.process(path)
        except Exception:
            return ''
    return ''
