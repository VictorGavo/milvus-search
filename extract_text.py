import fitz  # PyMuPDF

def extract_pdf_text(pdf_path):
    """
    Extracts text from a given PDF file.

    Parameters:
    - pdf_path (str): The path to the PDF file from which text is to be extracted.

    Returns:
    - str: Extracted text from the PDF.
    """
    text = ""
    with fitz.open(pdf_path) as doc:
        for page in doc:
            text += page.get_text()

    return text
