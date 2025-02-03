def extract_text_from_pdf(pdf_path):
    # Function to extract text from a PDF file using PyMuPDF
    import fitz  # PyMuPDF
    text = ""
    with fitz.open(pdf_path) as pdf_document:
        for page in pdf_document:
            text += page.get_text()
    return text

def normalize_json(title, date_published, content, keywords, summary):
    # Function to normalize data into JSON format
    return {
        "title": title,
        "date_published": date_published,
        "content": content,
        "keywords": keywords,
        "summary": summary
    }

def chunk_text(text, chunk_size):
    # Function to chunk text into fixed-size pieces
    return [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]