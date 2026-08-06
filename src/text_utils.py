import re
import unicodedata


TOKEN_PATTERN = re.compile(
    r"[a-z0-9]+(?:'[a-z0-9]+)?|[^\w\s]",
    re.IGNORECASE
)


def normalize_text(text):
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("’", "'")
    text = " ".join(text.split())

    return text.lower()


def tokenize(text):
    normalized_text = normalize_text(text)
    return TOKEN_PATTERN.findall(normalized_text)