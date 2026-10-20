# Optimization of GEMM Kernel Execution and Tiling Strategies

This document outlines the scope of the research project on General Matrix Multiplication (GEMM) performance optimization.

## 1. Cost Model
### 1.1 Background
Matrix multiplication (`mat_mul`) performance on hardware accelerators heavily depends on execution strategies (e.g., same-core K-split, multi-core K-split) and memory hierarchy tiling. The goal of this cost model is to accurately estimate execution latency for a given `mat_mul` implementation and tiling configuration.

### 1.2 Methodology
The cost model combines analytical hardware modeling with machine learning to capture non-linear system noise:

#### 1.2.1 Analytical Function
An analytical execution formulation modeling the execution pipeline, explicit memory transfers (DMA/L1/L2/UB), vector operations, and matrix execution units (Cube/Tensor Cores), accounting for double-buffering pipeline overlap:
$$T_{\text{analytical}} = \max(T_{\text{DMA}}, T_{\text{Compute}}) + T_{\text{overhead}}$$

#### 1.2.2 Residual Error Estimator
A Gradient Boosted Decision Tree (GBDT) model trained on empirical kernel profiles to predict non-deterministic hardware residual errors ($T_{\text{actual}} - T_{\text{analytical}}$). Features include problem dimensions ($M, N, K$), tile sizes ($m_0, n_0, k_0$), memory strides, and alignment factors.

## 2. Search & Optimization Algorithm
Using the hybrid cost model as an evaluation function, we implement a metaheuristic search algorithm (e.g., Genetic Algorithm combined with local search) to navigate the non-convex tiling search space and identify optimal split-K factors and tile sizes under hardware memory constraints.

## 3. Experiments & Evaluation
We validate the proposed optimization framework across three key dimensions:

### 3.1 Cost Model Accuracy
Compare prediction errors of the proposed hybrid cost model against pure analytical models and pure data-driven baseline models.

### 3.2 Tiling Policy Effectiveness
Evaluate kernel performance (runtime and TFLOPS) using the cost-model-guided tiling strategy against vendor default tiling heuristics.

### 3.3 Strategy & Implementation Selection
Compare dynamic kernel strategy selection (e.g., same-core vs. multi-core K-split) guided by the search algorithm against standard static library dispatches.

## 4. Extension to Fused Operators
Extend the cost-modeling and tiling optimization framework to complex fused operators (e.g., FlashAttention), accounting for online softmax rescaling, intermediate state retention, and SRAM/UB capacity limitations.