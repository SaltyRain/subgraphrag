def build_naive_prompt(triplets: list[dict], question: str, explain: bool = False) -> str:
    triplet_lines = [
        f"- {t['subject']['label']} – {t['predicate']['label']} – {t['object']['label']}"
        for t in triplets
    ]

    base_prompt = (
            "You are a question-answering assistant. "
            "Use the provided facts to answer the question.\n\n"
            "Facts:\n" +
            "\n".join(triplet_lines) +
            f"\n\nQuestion: {question}\n\n"
    )

    if explain:
        return base_prompt + "Answer the question and explain which facts support your answer.\n\nAnswer:"
    else:
        return base_prompt + "Answer:"
