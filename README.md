# Connected by Construction: Learning Tractable Near-Tour Marginals for Traveling Salesman Problems

## Abstract

Learning-based methods for the traveling salesman problem (TSP) are often evaluated by the tours produced after decoding or search. However, the learned object itself frequently lives in a surrogate space, such as heatmaps, assignments, construction policies, or search-guidance scores. This can obscure a fundamental question: **what Hamiltonian structure has actually been learned before decoding?**

In this study, we address this question by learning TSP through a structurally meaningful latent object, rather than leaving most of the Hamiltonian structure to the final decoding stage. Based on a **connected-by-construction rooted 1-tree Gibbs family**, we propose an end-to-end unsupervised learning pipeline called **C2TSP**.

C2TSP learns residual edge perturbations from unbiased TSP costs through implicit differentiation. For structural correction, a smoothed Held--Karp layer restores expected degree balance, while certificate-guided sharpening further pushes the connected distribution toward more tour-like structures.

Experiments show that C2TSP yields strong decoding performance while preserving interpretable structural information. Ablations further verify that edge perturbation and certificate-guided sharpening jointly improve both tour cost and tour-like structure.
