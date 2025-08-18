def build_summary_prompt(triplets: list[dict]) -> str:
    """
    Build a prompt that instructs an LLM to generate a coherent paragraph from a list of triplets.

    Args:
        triplets (list[dict]): A list of dictionaries representing triplets with human-readable labels.

    Returns:
        str: A formatted prompt for the LLM.
    """
    fact_lines = [
        f"- {t['subject']['label']} – {t['predicate']['label']} – {t['object']['label']}"
        for t in triplets
    ]

    prompt = (
            "You are a summarization assistant. You will receive a list of factual triplets about a topic. "
            "Each triplet represents a subject–predicate–object relationship.\n\n"
            "Your task is to turn these triplets into a coherent paragraph of factual information that could be easily understood by a human reader.\n\n"
            "Facts:\n"
            + "\n".join(fact_lines) +
            "\n\nWrite a paragraph describing these facts:"
    )

    return prompt


def build_generation_prompt_with_summary(question: str, summary: str) -> str:
    """
    Build a prompt for an LLM to answer a question based on a provided factual summary.

    Args:
        question (str): The user question.
        summary (str): A paragraph of factual information derived from a knowledge graph.

    Returns:
        str: A formatted prompt string for the LLM.
    """
    return (
        "You are an expert assistant designed to answer questions using a structured factual summary "
        "generated from a knowledge graph. The summary has been constructed from subject–predicate–object triplets "
        "retrieved from Wikidata. Use only the information in the summary to answer the question.\n\n"
        f"Context:\n{summary}\n\n"
        f"Question: {question}\n\n"
        "Your task:\n"
        "- Answer the question as accurately as possible using the context.\n"
        "- If the answer requires multiple facts, connect them logically.\n"
        "- If the context does not contain enough information, say \"I don't know based on the provided facts.\"\n"
        "- At the end, briefly explain which parts of the summary support your answer.\n\n"
        "Answer:"
    )
