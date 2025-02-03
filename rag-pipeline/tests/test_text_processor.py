def test_preprocess_text():
    from src.pipeline.text_processor import preprocess_text

    raw_text = "Sample PDF content for testing."
    expected_output = {
        "title": "Sample PDF Title",
        "date_published": "2023-10-01",
        "content": "Sample PDF content for testing.",
        "keywords": ["sample", "pdf", "testing"],
        "summary": "This is a summary of the sample PDF content."
    }

    output = preprocess_text(raw_text)
    assert output == expected_output, f"Expected {expected_output}, but got {output}"

def test_chunk_text():
    from src.pipeline.text_processor import chunk_text

    json_data = {
        "content": "This is a long content that needs to be chunked into smaller parts for processing."
    }
    chunk_size = 10
    expected_chunks = [
        "This is a ",
        "long conte",
        "nt that ne",
        "eds to be ",
        "chunked int",
        "o smaller p",
        "arts for pr",
        "ocessing."
    ]

    chunks = chunk_text(json_data["content"], chunk_size)
    assert chunks == expected_chunks, f"Expected {expected_chunks}, but got {chunks}"