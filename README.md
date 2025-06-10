# DataSwift

**DataSwift** is a tool that speeds up query workloads while being safe and easily deployable.



## 🚀 Overview

Traditional learned query optimizers often suffer from unpredictable slowdowns on unseen or rare query patterns. DataSwift addresses this by integrating:  
1. **LLM‐derived SQL embeddings** and **GNN‐based plan encodings** for rich query representations  
2. **Inductive Matrix Completion** with uncertainty estimation to predict hint performance  
3. **Embedding‐Indexed Memory (EIM)** to recall proven hints for similar past queries  
4. **Thompson‐sampling bandit** for balanced exploration/exploitation and safe fallback to the default optimizer



## 🎯 Key Features

- **Zero Catastrophic Regressions**: Only 0.7 % of queries ever slow down, and none exceed catastrophic thresholds.  
- **Tail‐Latency Speedups**: Achieves a 1.4× speedup on the slowest 5 % of queries and a 1.1× end‐to‐end workload improvement :contentReference[oaicite:1]{index=1}.
- **Safe Hint Recommendation**: Combines IMC’s calibrated predictions with a memory cache and bandit selector to ensure stability.



## 🏗️ Architecture

Full details are located in  [Extended Abstract](./Inductive_Matrix_Completion_with_Embedding_Memory_and_Bandit_Exploration_for_Safe_Hint_Recommendation__Extended_Abstract.pdf).

1. **Query Embedding**  
   - SQL text → SentenceTransformer → 120‐dim vector  
   - Plan DAG → GNN → 512‐dim structural vector  
2. **IMC Predictor**  
   - Concatenate embeddings → low‐rank IMC → mean latency (μ) + uncertainty (σ)  
3. **Embedding‐Indexed Memory (EIM)**  
   - Faiss L2 index of past (embedding, best‐hint) → retrieve safe hints  
4. **Bandit Selector**  
   - Arms: IMC suggestion, any EIM hints, default plan  
   - Thompson sampling to choose hint → execute → update bandit & EIM

