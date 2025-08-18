# Master Thesis

WebQSP-WD available at: https://public.ukp.informatik.tu-darmstadt.de/coling2018-graph-neural-networks-question-answering/WebQSP_WD_v1.zip



## Scripts
Please before run make sure to set correct parameters inside the `.env` file and each script. You can run them from the project root directory.

## Data Preparation
### 🛠 Prepare Training Dataset


```bash
./reproduce/data__01_prepare_train_dataset.sh
```

### 🛠 Prepare Test Dataset
```bash
./reproduce/data__02_prepare_test_dataset.sh
```

### 🛠 Fetch Wiki articles
```bash
./reproduce/data__03_fetch_wikipedia_articles.sh
```

## srtk tools - preprocess, train, retrieve + evaluate

### 🛠 Preprocess
```bash
./reproduce/srtk__01_preprocess_data.sh
```
### 🛠 Train
```bash
./reproduce/srtk__02_train_retriever.sh
```

### 🛠 Retrieve
```bash
./reproduce/srtk__03_retrieve_subgraphs.sh
```

## SubgraphRAG
### 🛠 Build labels map
```bash
./reproduce/sg__00_update_labels_map.sh
```
### 🛠 Construct subgraphs from triplets
```bash
./reproduce/sg__01_construct_subgraphs.sh
```
### 🛠 Generate query-focused summaries
```bash
./reproduce/sg__02_fuse_contexts.sh
```
### 🛠 Generate answers
```bash
./reproduce/sg__03_generate_precise_answers.sh
```
### 🛠 Evaluate
```bash
./reproduce/sg__04_eval_subgraphrag.sh
```

## Baselines

### 🛠 Run naive RAG experiment
```bash
./reproduce/baselines__naiverag_experiment.sh
```
### 🛠 Build Lightrag index
```bash
./reproduce/baselines__01_build_lightrag_index.sh
```
### 🛠 Evaluate
```bash
./reproduce/baselines__02_lightrag_experiment.sh
```