def build_context_prompt(triplets: list[str], question: str) -> str:
    """
    Build a prompt for an LLM to generate a coherent paragraph based on all ranked triplets.

    Args:
        triplets (list[str]): A list of triplets in plain-text format (e.g. "subject — predicate — object").
        question (str): The original question the context is related to.

    Returns:
        str: A formatted prompt suitable for input to an LLM.
    """
    triplet_lines = [f"- {triplet.strip()}" for triplet in triplets]

    prompt = (
        f"You are a knowledge assistant. You will be given a set of factual statements related to the following question:\n"
        f"\"{question}\"\n\n"
        f"Your task is to write a short, coherent paragraph summarizing these facts so that they are easy to understand for a human reader.\n\n"
        f"Facts:\n" + "\n".join(triplet_lines) +
        "\n\nWrite a paragraph that summarizes the facts clearly:"
    )

    return prompt
