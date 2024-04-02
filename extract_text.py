import fitz  # PyMuPDF

def segment_pdf(pdf_path):
    """
    Segments a PDF into logical units based on font size, with an additional check to
    exclude segments that are shorter than a minimum length, effectively filtering out
    headers and other short segments that do not constitute meaningful content.

    Parameters:
    - pdf_path (str): Path to the PDF file.

    Returns:
    - List of strings, each containing the text of a segment, excluding those that do not
      meet the minimum length requirement.
    """
    segments = []
    current_segment = []  # Accumulate text for the current segment

    with fitz.open(pdf_path) as doc:
        for page_num, page in enumerate(doc):
            blocks = page.get_text("dict")["blocks"]
            for block in blocks:
                if block['type'] == 0:  # Ensure we're dealing with a text block
                    for line in block["lines"]:
                        for span in line["spans"]:
                            text = span['text'].strip()
                            font_size = span['size']
                            
                            if font_size > 12:
                                # When encountering a larger font size, start a new segment
                                segment_text = " ".join(current_segment).strip()
                                if len(segment_text) >= 50:  # Check segment length before appending
                                    segments.append(segment_text)
                                current_segment = []  # Reset for the new segment
                            
                            # Add non-empty text to the current segment
                            if text:
                                current_segment.append(text)
            
            # End of page: save any accumulated text as the last segment of this page
            segment_text = " ".join(current_segment).strip()
            if segment_text and len(segment_text) >= 50:  # Length check for the last segment
                segments.append(segment_text)
            current_segment = []  # Reset for the next page

    return segments

# if __name__ == "__main__":
#     pdf_path = "data/supremecourt_landmarkcases 01.pdf"  # Adjust the path to your PDF file
#     segments = segment_pdf(pdf_path)

#     # Printing the first few segments for testing purposes
#     for i, segment_text in enumerate(segments[:10], start=1):  # Adjust the slice as needed
#         print(f"Segment {i}:\n{segment_text}\n")
#         print("-----\n")
