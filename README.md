# Optimization of GEMM

This document set up the scope of the reseach project. The content of this project includes,

- Develop a cost model to estimate the run time of a GEMM operator.
  - An analytic form of the cost function taking into account of data-copy, cube operation, fixpipe, etc.
  - A residual error estimator which is trained by a gradient boost tree model.
- A local search algorithm to find the best tiling strategy based on the cost model.
- Experiments to validate the tiling strategy and algorithm selection.

## 1. Cost Model
### 1.1 Backgrounds
There are plenty of ways to implement a mat_mul operator, e.g.
- Same-core k-split
- Multi-core k-split

The goal of the development of the cost model is to estimate the run time of a specific mat_mul implement under a given tiling strategy. 

### 1.2 Methodology
A cost model includes the following two parts.
#### 1.2.1 Analytical function
Use an anlytical function form to represent the flow of mat_mul operation.
#### 1.2.2 Residual Error Estimator
Use a gradient boost tree model to predict the error. 

## 2. Local Search

With the cost funtion, we can develop a local search heuristic like Genetic Algorithm to find the best tiling strategy.


## 3. Experiments

### 3.1 Validate the cost model
Compare the cost model with a pure data-driven method.

### 3.2 Tiling policy
Compare the effectiveness of the heuristic selected tiling policy versus the default selection.

### 3.3 Implementation Selection
Compare the default selection of mat_mul implementation vs the heuristic selected implementation.

## 4. Extension

Validate the effectiveness in a complex operator, e.g. flash attention by adopting the enhanced mat_mul operator. 