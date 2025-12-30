# Subgraph Retrieval for GraphRAG

**Master’s Thesis — Scalable Subgraph-Based Retrieval for Graph-Enhanced RAG Systems**

This repository contains the full experimental codebase, scripts, and reproducibility setup for my Master’s thesis on **efficient subgraph retrieval for Graph-based Retrieval-Augmented Generation (GraphRAG)**, evaluated on the **WebQSP-WD** benchmark.

The project focuses on bridging large knowledge graphs (Wikidata) and Large Language Models via query-focused subgraph extraction and structured context fusion.

---

## 🧠 Abstract

Large Language Models (LLMs) are limited in their ability to perform multi-hop reasoning over large-scale knowledge graphs due to context length constraints and inefficient retrieval strategies. While Retrieval-Augmented Generation (RAG) mitigates this by injecting external knowledge, naive document-level retrieval remains insufficient for structured, relational queries.

This thesis investigates **subgraph retrieval as an intermediate retrieval layer** between knowledge graphs and LLMs. The proposed approach extracts compact, query-focused subgraphs that preserve relational structure while remaining suitable for LLM consumption.

The work introduces a modular **SubgraphRAG pipeline** combining learned retrieval (SRTK), symbolic expansion, and LLM-based context fusion, and evaluates it against strong baselines such as **Naive RAG** and **LightRAG** on the WebQSP-WD dataset.

---

## 🚀 Contributions

* Design and implementation of a **modular subgraph retrieval pipeline** for GraphRAG
* Integration of **SRTK-based learned retrievers** with symbolic neighborhood expansion
* End-to-end **SubgraphRAG system**, including:

  * subgraph construction from retrieved triplets
  * query-focused context fusion
  * answer generation using LLMs
* Fully **reproducible experimental setup** with shell-based pipelines
* Empirical comparison against **Naive RAG** and **LightRAG** on WebQSP-WD

---

## 🗂 Project Structure

```text
.
├── reproduce/              # Reproducible experiment scripts
│   ├── data__*.sh           # Dataset preparation
│   ├── srtk__*.sh           # SRTK preprocessing, training, retrieval
│   ├── sg__*.sh             # SubgraphRAG pipeline
│   └── baselines__*.sh      # Naive RAG & LightRAG baselines
├── thesis/                  # Thesis document
│   └── thesis.pdf
├── src/                     # Core Python implementation
├── configs/                 # Experiment & model configurations
├── .env.example             # Environment variable template
└── README.md
```

---

## 🔁 Reproducibility

All experiments are fully reproducible via shell scripts.

Before running any script, make sure to:

* create a `.env` file based on `.env.example`
* verify dataset and model paths inside the scripts if required

All commands are executed from the project root directory.

---

## 📦 Data Preparation

### Prepare Training Dataset

```bash
./reproduce/data__01_prepare_train_dataset.sh
```

### Prepare Test Dataset

```bash
./reproduce/data__02_prepare_test_dataset.sh
```

### Fetch Wikipedia Articles

```bash
./reproduce/data__03_fetch_wikipedia_articles.sh
```

---

## 🧩 SRTK Pipeline

### Preprocessing

```bash
./reproduce/srtk__01_preprocess_data.sh
```

### Training the Retriever

```bash
./reproduce/srtk__02_train_retriever.sh
```

### Subgraph Retrieval

```bash
./reproduce/srtk__03_retrieve_subgraphs.sh
```

---

## 🔗 SubgraphRAG Pipeline

### Build Labels Map

```bash
./reproduce/sg__00_update_labels_map.sh
```

### Construct Subgraphs from Triplets

```bash
./reproduce/sg__01_construct_subgraphs.sh
```

### Generate Query-Focused Summaries

```bash
./reproduce/sg__02_fuse_contexts.sh
```

### Generate Answers

```bash
./reproduce/sg__03_generate_precise_answers.sh
```

### Evaluation

```bash
./reproduce/sg__04_eval_subgraphrag.sh
```

---

## 📊 Baselines

### Naive RAG

```bash
./reproduce/baselines__naiverag_experiment.sh
```

### LightRAG

```bash
./reproduce/baselines__01_build_lightrag_index.sh
./reproduce/baselines__02_lightrag_experiment.sh
```

---

## 📚 Dataset

Experiments are conducted on **WebQSP-WD**, a Wikidata-aligned version of the WebQSP question-answering benchmark.

🔗 Dataset download:
[https://public.ukp.informatik.tu-darmstadt.de/coling2018-graph-neural-networks-question-answering/WebQSP_WD_v1.zip](https://public.ukp.informatik.tu-darmstadt.de/coling2018-graph-neural-networks-question-answering/WebQSP_WD_v1.zip)

---

## 📄 Thesis

📘 **Master’s Thesis (PDF)**
[`thesis/thesis.pdf`](thesis/thesis.pdf)

**Title:** Subgraph Retrieval for Graph-Based Retrieval-Augmented Generation
**Author:** Timur Garipov
**Degree:** Master’s Thesis
**Field:** Computer Science / Artificial Intelligence

---

## 📝 Citation

If you use this work, please cite:

```bibtex
@mastersthesis{garipov2025subgraphrag,
  title     = {Scalable Subgraph-Based Retrieval for Graph-Enhanced RAG Systems},
  author    = {Garipov, Timur},
  year      = {2025},
  school    = {University of Passau}
}
```

---

## ⚠️ Disclaimer

This repository is intended for **research and educational purposes**.
Some components are experimental and may require adaptation for production environments.

---

## 📜 License

The code in this repository is released under the **MIT License**, unless stated other
