from typing import List, Tuple
from word_classes import WORD_CLASSES

VOCAB_WORDS = {word.upper() for word in WORD_CLASSES}

def text_to_sequence(text: str) -> List[Tuple[str, str]]:
    """
    Convert input text into a sequence of (type, token) items.
    type: "WORD" for vocabulary words, "LETTER" for fingerspelling A-Z.
    """
    seq: List[Tuple[str, str]] = []
    if not text:
        return seq
    for raw in text.strip().split():
        word = ''.join(ch for ch in raw.upper() if ch.isalpha())
        if not word:
            continue
        if word in VOCAB_WORDS:
            seq.append(("WORD", word))
        else:
            for ch in word:
                if 'A' <= ch <= 'Z':
                    seq.append(("LETTER", ch))
    return seq
