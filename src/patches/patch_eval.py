import os
import pandas as pd

from typing import Hashable
from dotenv import load_dotenv
from ragas import evaluate, SingleTurnSample, EvaluationDataset
from ragas.metrics import answer_correctness

from langchain_ollama import OllamaLLM, OllamaEmbeddings


# === PATHS ===
INPUT_CSV  = "results/exp1_e2e/eval/r-high/evaluation.csv"
OUTPUT_CSV = "results/exp1_e2e/eval/r-high/evaluation.patched.csv"
EXPERIMENT_NAME = "ragas_recompute_missing"

load_dotenv()

def build_ragas_llm_and_emb():
    """Create LLM/Embeddings for RAGAS with strict JSON output to avoid parser errors."""
    llm = OllamaLLM(
        model=os.getenv("LLM_MODEL_NAME") or "llama3.1",
        base_url=os.getenv("LLM_BINDING_HOST") or "http://172.21.112.1:11434",
        temperature=0.0,
        format="json",
    )
    emb = OllamaEmbeddings(
        model=os.getenv("LLM_EMBEDDING_MODEL_NAME") or "nomic-embed-text",
        base_url=os.getenv("LLM_BINDING_HOST") or "http://172.21.112.1:11434",
    )
    return llm, emb

def rows_to_samples(df_subset: pd.DataFrame) -> [list[SingleTurnSample], list[Hashable]]:
    """
    Convert rows with columns: user_input, response, reference
    into RAGAS SingleTurnSample list.
    """
    samples = []
    kept_indices = []
    for idx, row in df_subset.iterrows():
        user_input = str(row.get("user_input", "") or "").strip()
        response   = str(row.get("response", "") or "").strip()
        reference  = str(row.get("reference", "") or "").strip()

        if not user_input or not response or not reference:
            continue

        samples.append(
            SingleTurnSample(
                user_input=user_input,
                response=response,
                reference=reference,
            )
        )
        kept_indices.append(idx)
    return samples, kept_indices

def main():
    # 1) Load CSV
    df = pd.read_csv(INPUT_CSV)
    if "answer_correctness" not in df.columns:
        raise ValueError("Column 'answer_correctness' not found in input CSV.")

    # 2) Find rows with missing/null answer_correctness
    mask_missing = df["answer_correctness"].isna()
    missing_df = df[mask_missing].copy()

    print(f"Total rows: {len(df)}")
    print(f"Missing answer_correctness rows: {len(missing_df)}")

    if missing_df.empty:
        print("Nothing to patch. Saving a copy with the same content...")
        df.to_csv(OUTPUT_CSV, index=False)
        print(f"Saved: {OUTPUT_CSV}")
        print(f"Global mean correctness: {df['answer_correctness'].mean(skipna=True):.6f}")
        return

    # 3) Build samples only for rows we can evaluate (must have reference)
    samples, kept_indices = rows_to_samples(missing_df)
    print(f"Rows eligible for recompute (with reference present): {len(samples)}")

    if not samples:
        print("No eligible rows to recompute (missing reference). Saving original with no changes...")
        df.to_csv(OUTPUT_CSV, index=False)
        print(f"Saved: {OUTPUT_CSV}")
        print(f"Global mean correctness: {df['answer_correctness'].mean(skipna=True):.6f}")
        return

    # 4) RAGAS evaluation for the subset
    llm, emb = build_ragas_llm_and_emb()
    dataset = EvaluationDataset(samples=samples)
    results = evaluate(
        dataset=dataset,
        metrics=[answer_correctness],
        llm=llm,
        embeddings=emb,
        experiment_name=EXPERIMENT_NAME,
        show_progress=True,
        batch_size=8,
    )

    subset_df = results.to_pandas()
    if "answer_correctness" not in subset_df.columns:
        raise RuntimeError("RAGAS result did not contain 'answer_correctness' column.")

    # 5) Map back: the order of samples corresponds to kept_indices
    # We substitute the calculated values into the original df
    for i, idx in enumerate(kept_indices):
        df.at[idx, "answer_correctness"] = float(subset_df.iloc[i]["answer_correctness"])

    # add safe check that the folder exists
    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)

    # 6) Save patched CSV
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"Patched CSV saved: {OUTPUT_CSV}")

    # 7) Print global mean correctness
    mean_corr = df["answer_correctness"].mean(skipna=True)
    print(f"Global mean answer_correctness: {mean_corr:.6f}")

if __name__ == "__main__":
    main()
