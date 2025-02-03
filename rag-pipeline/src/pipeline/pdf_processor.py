import fitz  # PyMuPDF
import os

def extract_text(pdf_folder):
    pdf_texts = {}
    for filename in os.listdir(pdf_folder):
        if filename.endswith('.pdf'):
            pdf_path = os.path.join(pdf_folder, filename)
            document = fitz.open(pdf_path)
            text = ""
            for page in document:
                text += page.get_textpage().extractText()
            pdf_texts[filename] = text
            document.close()
    return pdf_texts

def pick_pdfs(pdf_folder):
    pdf_files = [os.path.join(pdf_folder, f) for f in os.listdir(pdf_folder) if f.endswith('.pdf')]
    return pdf_files

def save_extracted_texts(pdf_texts, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    for filename, text in pdf_texts.items():
        output_file = os.path.join(output_folder, f"{filename}.txt")
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(text)