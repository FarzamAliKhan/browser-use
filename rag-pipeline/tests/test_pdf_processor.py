def test_pdf_processing():
    from src.pipeline.pdf_processor import extract_text_from_pdfs
    import os

    # Assuming the test PDFs are in the data/pdfs directory
    pdf_directory = os.path.join(os.path.dirname(__file__), '../data/pdfs')
    extracted_texts = extract_text_from_pdfs(pdf_directory)

    # Check if the extracted texts are not empty
    assert len(extracted_texts) > 0, "No text was extracted from the PDFs."

    # Further assertions can be added based on expected output format
    for text in extracted_texts:
        assert isinstance(text, str), "Extracted text should be a string."
        assert len(text) > 0, "Extracted text should not be empty."