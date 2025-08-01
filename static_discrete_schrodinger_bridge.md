# Static Discrete Schrödinger Bridge Problem

## Overview

This document outlines the structure and implementation of the static discrete Schrödinger bridge problem. The goal is to find the optimal transport plan between two discrete probability distributions that minimizes the relative entropy with respect to a given reference measure.

## Problem Formulation

### Mathematical Background

The static Schrödinger bridge problem can be formulated as follows:

Given:
- Source distribution $\mu$ on finite set $\mathcal{X}$
- Target distribution $\nu$ on finite set $\mathcal{Y}$  
- Reference measure $R$ on $\mathcal{X} \times \mathcal{Y}$

Find the optimal coupling $\pi^*$ that minimizes:
$$\min_{\pi} D_{KL}(\pi \| R)$$

Subject to marginal constraints:
- $\sum_{y} \pi(x,y) = \mu(x)$ for all $x \in \mathcal{X}$
- $\sum_{x} \pi(x,y) = \nu(y)$ for all $y \in \mathcal{Y}$

### Solution Method

The problem is solved using the **Sinkhorn-Knopp algorithm** (also known as the iterative proportional fitting procedure), which is an efficient method for computing optimal transport plans.

## Notebook Structure

### 1. Problem Setup and Data Generation

**File Section:** `1. Problem Setup and Data Generation`

**Key Functions:**
- `generate_discrete_distributions(n_x, n_y)`: Creates random source and target distributions
- `generate_reference_measure(n_x, n_y, method)`: Creates reference measure with different methods:
  - `'uniform'`: Constant reference measure
  - `'gaussian'`: Gaussian-based reference measure
  - `'random'`: Random exponential reference measure

**Learning Objectives:**
- Understand how to generate discrete probability distributions
- Learn different ways to construct reference measures
- Set up the problem parameters and data

### 2. Sinkhorn-Knopp Algorithm Implementation

**File Section:** `2. Sinkhorn-Knopp Algorithm Implementation`

**Key Function:**
- `sinkhorn_knopp_algorithm(mu, nu, R, max_iter, tol, verbose)`: Implements the core algorithm

**Algorithm Steps:**
1. Initialize dual variables $u$ and $v$
2. Iteratively update:
   - $u_i = \log(\mu_i) - \log(\sum_j R_{ij} e^{v_j})$
   - $v_j = \log(\nu_j) - \log(\sum_i R_{ij} e^{u_i})$
3. Compute optimal coupling: $\pi_{ij} = R_{ij} e^{u_i + v_j}$
4. Check convergence using marginal constraint violations

**Learning Objectives:**
- Understand the dual formulation of the Schrödinger bridge problem
- Learn the iterative nature of the Sinkhorn-Knopp algorithm
- Implement convergence checking and error monitoring

### 3. Solve the Schrödinger Bridge Problem

**File Section:** `3. Solve the Schrödinger Bridge Problem`

**Content:**
- Execute the algorithm with the generated data
- Display convergence information
- Verify basic properties of the solution

**Learning Objectives:**
- Run the complete algorithm end-to-end
- Interpret convergence results
- Understand the output structure

### 4. Visualization and Analysis

**File Section:** `4. Visualization and Analysis`

**Key Function:**
- `plot_schrodinger_bridge_results(mu, nu, R, pi_optimal, info)`: Comprehensive visualization

**Visualizations:**
1. **Source and Target Distributions**: Bar plot comparing original distributions
2. **Reference Measure**: Heatmap of the reference measure $R$
3. **Optimal Coupling**: Heatmap of the optimal transport plan $\pi^*$
4. **Convergence History**: Log-scale plot of error over iterations
5. **Marginal Constraint Verification**: Comparison of original vs. reconstructed marginals
6. **KL Divergence Analysis**: Contribution of each cell to the total KL divergence

**Learning Objectives:**
- Visualize high-dimensional transport plans
- Understand the relationship between reference measure and optimal coupling
- Monitor algorithm convergence
- Verify constraint satisfaction

### 5. Analysis and Validation

**File Section:** `5. Analysis and Validation`

**Key Function:**
- `validate_solution(pi, mu, nu, R)`: Comprehensive solution validation

**Validation Checks:**
- **Marginal Constraints**: Verify that $\sum_j \pi_{ij} = \mu_i$ and $\sum_i \pi_{ij} = \nu_j$
- **Positivity**: Ensure all elements of $\pi$ are non-negative
- **Normalization**: Check that $\sum_{ij} \pi_{ij} = 1$
- **KL Divergence**: Compute the total KL divergence from reference measure

**Learning Objectives:**
- Implement rigorous validation procedures
- Understand numerical precision issues
- Learn to verify mathematical constraints

### 6. Comparison with Other Methods

**File Section:** `6. Comparison with Other Methods`

**Comparison Methods:**
1. **Independent Coupling**: $\pi_{ij} = \mu_i \nu_j$ (product of marginals)
2. **Uniform Coupling**: $\pi_{ij} = \frac{1}{|\mathcal{X}||\mathcal{Y}|}$
3. **Reference Measure**: Normalized reference measure
4. **Sinkhorn-Knopp**: Our optimal solution

**Learning Objectives:**
- Compare different transport strategies
- Understand why the Sinkhorn-Knopp solution is optimal
- Learn to benchmark algorithms

### 7. Conclusion and Summary

**File Section:** `7. Conclusion and Summary`

**Content:**
- Summary of results and performance
- Key insights about the algorithm
- Discussion of convergence properties
- Final validation of optimality

## Mathematical Details

### Dual Formulation

The primal problem:
$$\min_{\pi} \sum_{ij} \pi_{ij} \log\left(\frac{\pi_{ij}}{R_{ij}}\right)$$
subject to $\sum_j \pi_{ij} = \mu_i$ and $\sum_i \pi_{ij} = \nu_j$

Has the dual formulation:
$$\max_{u,v} \sum_i u_i \mu_i + \sum_j v_j \nu_j - \sum_{ij} R_{ij} e^{u_i + v_j}$$

The optimal coupling is then:
$$\pi_{ij}^* = R_{ij} e^{u_i^* + v_j^*}$$

### Convergence Properties

- The Sinkhorn-Knopp algorithm converges linearly
- The error decreases exponentially in the number of iterations
- The algorithm is guaranteed to converge to the unique optimal solution

## Implementation Notes

### Numerical Considerations

1. **Log-space computations**: All computations are done in log-space to avoid numerical underflow
2. **Convergence criteria**: Use maximum marginal constraint violation as convergence criterion
3. **Regularization**: The algorithm naturally handles the regularization through the reference measure

### Computational Complexity

- **Time Complexity**: $O(n_x n_y \cdot \text{iterations})$
- **Space Complexity**: $O(n_x n_y)$
- **Typical iterations**: 50-200 for well-conditioned problems

## Extensions and Applications

### Possible Extensions

1. **Entropic Regularization**: Add regularization parameter $\epsilon$ to control the trade-off between optimality and computational efficiency
2. **Multi-marginal Problems**: Extend to problems with more than two marginals
3. **Continuous Distributions**: Discretize continuous distributions for numerical solution
4. **Dynamic Schrödinger Bridge**: Extend to time-dependent problems

### Applications

1. **Optimal Transport**: Computing optimal transport plans between distributions
2. **Image Processing**: Image interpolation and morphing
3. **Machine Learning**: Generative modeling, domain adaptation
4. **Economics**: Matching problems, resource allocation
5. **Physics**: Quantum mechanics, statistical mechanics

## References

1. Sinkhorn, R., & Knopp, P. (1967). Concerning nonnegative matrices and doubly stochastic matrices.
2. Cuturi, M. (2013). Sinkhorn distances: Lightspeed computation of optimal transport.
3. Peyré, G., & Cuturi, M. (2019). Computational optimal transport.
4. Léonard, C. (2014). A survey of the Schrödinger problem and its connections with optimal transport.

## Code Structure Summary

```
static_discrete_schrodinger_bridge.ipynb
├── 1. Problem Setup and Data Generation
│   ├── generate_discrete_distributions()
│   └── generate_reference_measure()
├── 2. Sinkhorn-Knopp Algorithm Implementation
│   └── sinkhorn_knopp_algorithm()
├── 3. Solve the Schrödinger Bridge Problem
├── 4. Visualization and Analysis
│   └── plot_schrodinger_bridge_results()
├── 5. Analysis and Validation
│   └── validate_solution()
├── 6. Comparison with Other Methods
│   └── compare_with_other_methods()
└── 7. Conclusion and Summary
```

This structure provides a comprehensive introduction to the static discrete Schrödinger bridge problem, with both theoretical understanding and practical implementation. 