from typing import Tuple, Any

from docx import Document
from googletrans import Translator
from transformers import pipeline


def detect_emotion(text: str, language: str, min_score: float = 0.5) -> tuple[Any, Any]:

    if language != 'en':
        translator = Translator()
        text_en = translator.translate(text, dest='en')
        text = text_en.text

    print(f"Translated text: {text}")

    emotion_detection_model = pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base")
    emotions = emotion_detection_model(text, top_k=None)
    print(f"Detected emotions: {emotions}")

    # anger, disgust, fear, joy, neutral, sadness, surprise
    most_probable_emotion = emotions[0]['label']
    most_probable_score = emotions[0]['score']

    if most_probable_score < min_score:
        return 'neutral', 0.0

    return most_probable_emotion, most_probable_score

def split_sentence(sentence, max_length=182) -> list:
    if len(sentence) <= max_length:
        return [sentence]

    # Find all commas in the sentence
    commas = [pos for pos, char in enumerate(sentence) if char == ',']

    # If there are no commas, just split at max_length
    if not commas:
        return [sentence[:max_length], sentence[max_length:].strip()]

    # Find the comma closest to the middle of the sentence
    middle = len(sentence) // 2
    closest_comma = min(commas, key=lambda x: abs(x - middle))

    # Split the sentence at the closest comma
    left_part = sentence[:closest_comma].strip()
    right_part = sentence[closest_comma + 1:].strip()

    # Recursively split if necessary
    left_split = split_sentence(left_part, max_length)
    right_split = split_sentence(right_part, max_length)

    return left_split + right_split

def extract_text_from_docx(docx_path) -> str:
    doc = Document(docx_path)
    full_text = []
    for paragraph in doc.paragraphs:
        full_text.append(paragraph.text)
    return '\n'.join(full_text)
