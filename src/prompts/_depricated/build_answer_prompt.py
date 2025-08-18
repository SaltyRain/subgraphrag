def build_answer_prompt(context: str, question: str) -> str:
    return (
        "You are a factual question-answering assistant.\n"
        "Respond ONLY with the answer strictly supported by the provided context.\n"
        "Do NOT make guesses, add explanations, or include unrelated information.\n"
        "If the context does not contain the answer, respond exactly with: Not enough information\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {question}\n"
        "Answer (one short phrase):"
    )
