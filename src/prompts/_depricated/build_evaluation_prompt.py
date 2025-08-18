def build_evaluation_prompt(answer: str, gold_answers: list[str], question: str) -> str:
    """
    Build a prompt for evaluating an answer based on gold answers.
    """
    return (
        f"You are an evaluation assistant. You are given a question, a gold answer, and a predicted answer.\n\n"
        f"Question: {question}\n"
        f"Gold answers: {', '.join(gold_answers)}\n"
        f"Predicted answer: {answer.strip()}\n\n"
        f"Is the predicted answer correct? Answer with 'Yes' or 'No' and optionally a short justification."
    )
