import re

def decompose_into_claims(text):
    """
    Split text into atomic claims using sentence boundaries.
    """

    # Split on period, question mark, exclamation
    sentences = re.split(r'(?<=[.!?]) +', text)

    # Clean empty or very short sentences
    claims = [s.strip() for s in sentences if len(s.strip()) > 20]

    return claims