import re

def clean_text(text: str) -> str:
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r'\s{2,}', ' ', text)
    text = re.sub(r'==+.*?==+', '', text)
    return text.strip()

def chunk_by_paragraph(text: str, max_size=400, overlap=50):  # reduced from 700/100
    text = clean_text(text)
    paragraphs = [p.strip() for p in text.split('\n\n') if len(p.strip()) > 60]
    chunks = []
    current = ""

    for para in paragraphs:
        if len(current) + len(para) <= max_size:
            current += " " + para
        else:
            if current:
                chunks.append(current.strip())
            current = para

    if current:
        chunks.append(current.strip())

    overlapped = []
    for i, chunk in enumerate(chunks):
        if i > 0:
            prev_words = chunks[i-1].split()[-10:]   # reduced overlap words
            chunk = " ".join(prev_words) + " " + chunk
        overlapped.append(chunk)

    return overlapped