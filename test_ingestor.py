import pytest
from ingestor import semantic_chunking

def test_semantic_chunking():
    text = "This is the first sentence. This is the second sentence. \n\n This is another paragraph."
    chunks = semantic_chunking(text, min_length=10, max_length=50)
    assert len(chunks) > 0
    assert "This is the first sentence." in chunks[0]

def test_semantic_chunking_empty():
    chunks = semantic_chunking("")
    assert len(chunks) == 0
