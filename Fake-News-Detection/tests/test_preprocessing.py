import pytest
from src.preprocessing import preprocess

def test_preprocess_empty_string():
    assert preprocess("") == ""
    assert preprocess("   ") == ""
    assert preprocess(float('nan')) == ""

def test_preprocess_non_ascii_emoji():
    # Emoji should be stripped, currency normalized or stripped based on behavior
    # 'Weird text here—with emojis 😊 and € currency.'
    text = "Weird text here—with emojis 😊 and € currency."
    clean = preprocess(text)
    assert "emoji" in clean
    assert "currency" in clean
    assert "😊" not in clean

def test_preprocess_contractions():
    # "don't" -> "do not"
    text = "I don't know"
    clean = preprocess(text)
    assert "not" in clean
    assert "know" in clean

def test_preprocess_negations():
    # "not", "no", "nor", "never" should survive stopword removal
    text = "I am not happy, and I will never go there. No way!"
    clean = preprocess(text)
    tokens = clean.split()
    assert "not" in tokens
    assert "never" in tokens
    assert "no" in tokens
