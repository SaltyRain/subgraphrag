import re

def format_title_to_wikimapper(raw_title: str) -> str:
    """
    Normalize a Wikipedia title into wikimapper friendly format:
    - Capitalize each word
    - Preserve hyphens (e.g., 'al-Qaeda' -> 'Al-Qaeda')
    - Replace whitespace with underscores
    - Strip leading/trailing spaces
    """
    def capitalize_word(w: str) -> str:
        if "-" in w:
            return "-".join(part.capitalize() for part in w.split("-"))
        return w.capitalize()

    # Remove leading/trailing spaces and split by whitespace
    words = re.split(r"(\s+)", raw_title.strip())
    title_case = "".join([capitalize_word(word) if word.strip() else word for word in words])

    # Replace whitespace with underscores
    normalized = re.sub(r"\s+", "_", title_case)
    return normalized