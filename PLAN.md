# Repository Planning: topological-flow-matching

This document will be used to plan out the structure, features, and development roadmap for the topological-flow-matching project.

---

## Sections
- Project Overview
- Features
- Architecture
- Milestones
- Notes 


## Project Overview
This project implements Topological Flow Matching, which solves the Schrodinger bridge problem (SBP) and its optimal transport (OT) zero-noise limit by learning the control that guides the optimal evolution under a given "topological" reference (prior) process. 

The project has five key components: 
1. Explicit expression for topological Schrodinger bridges with Dirac delta boundary marginals. This is given explicitly in [1] for the multivariate case. However, in practice we only need the 1-dimensional case, since the SDE can be diagonalised. 
2. Computation of the solution to the static SBP, which, under some conditions, is equivalent to finding the transference plan solving entropic OT. In the zero-noise case it should be possible to do this exactly, whereas in the noisy case we find an approximate solution. 
3. Computation of 1- and 2-Wasserstein distance between two empirical distributions, which is how the models are evaluated in [1]. 
4. Implementation of the neural network that will predict the control. This will be taken as closely as possible from the codebase for [1] available at https://github.com/cookbook-ms/topological_SB_matching/tree/main. 
5. Training of parametrised control models via conditional flow matching loss. 



## References
[1] Maosheng Yang, Topological Schrodinger Bridge Matching, 2025, ICLR 2025 Spotlight

