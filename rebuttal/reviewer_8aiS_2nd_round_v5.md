We thank the reviewer for the careful follow-up. In response, we added zero-shot experiments on two metric non-Euclidean datasets using the same TSP-100 checkpoints. The non-Euclideanity analysis shows that new datasets deviate substantially more from Euclidean geometry than the non-Euclidean TSPLIB instances considered in our previous response. The experimental results show that **C2TSP remains effective** and outperforms every learned baseline by a wide margin on both datasets. This advantage is consistent with C2TSP's edge-based representation and algorithmic components. We will also add these results to the appendix.

**Non-Euclidean benchmarks:**

- **Dataset A (graph shortest-path metric) [1], [2].** We sample nodes uniformly from [0,1]², retain a sparse Erdős–Rényi subset of edges, and assign each retained edge its Euclidean length. The complete target cost matrix is then defined by the weighted shortest-path distances between all pairs of nodes in the sparse graph. Detours around missing edges make the resulting metric non-Euclidean.
- **Dataset B (Hamiltonian-cycle 1–2 metric) [3].** We take the Flinders Hamiltonian-cycle graphs and set cost 1 on the edges of each source graph and cost 2 elsewhere. These instances have no coordinates at all. Every source graph is Hamiltonian, so the exact optimum is n.

**Quantifying non-Euclideanity.** We use a spectral non-Euclideanity score based on Gower's criterion [4]: the score is zero for Euclidean distances, and larger values indicate stronger deviations from Euclidean geometry. TSPLIB's `geo`, `att`, and rounded `euc_2d` instances score in 0.001–0.009, confirming the reviewer's point that they are nearly Euclidean. Dataset A scores 0.27–0.40 and Dataset B scores 0.10–0.28. Substantially higher scores of both datasets support their use as non-Euclidean benchmarks.

**Setup.** All methods use the same decoder setting (L2×100). The four baselines take coordinates as input: on Dataset A they receive the original node coordinates, and on Dataset B they receive a metric MDS embedding of the target matrix [5] (because these baselines support only coordinate-based input). C2TSP takes the cost matrix directly.

**Dataset A: graph shortest-path metric.** Cells report the mean optimality gap (%) ± standard deviation across instances, and the per-instance wall time in seconds (in parentheses).

|Method|n=100|n=200|n=500|n=1000|
|---|---:|---:|---:|---:|
|**C2TSP**|1.41±0.63 (0.0)|1.52±0.47 (0.1)|1.93±0.42 (0.3)|2.05±0.22 (2.0)|
|DIFUSCO|8.03±2.21 (0.3)|20.56±2.34 (0.7)|108.84±4.95 (5.7)|185.14±5.12 (15.9)|
|Fast-T2T|7.67±2.04 (0.0)|20.51±2.55 (0.0)|108.06±4.91 (0.2)|184.33±5.60 (0.5)|
|DIMES|7.14±2.29 (0.0)|20.39±2.21 (0.0)|107.67±5.72 (0.0)|184.31±5.90 (0.2)|
|UTSP|7.31±1.68 (0.0)|20.39±2.80 (0.0)|107.97±4.63 (0.0)|184.19±5.29 (0.1)|

**Dataset B: Hamiltonian-cycle 1–2 metric.** The exact optimum is n for every instance.

|Method|3-regular<br>(24 inst.)|sparse<br>(9 inst.)|dense<br>(4 inst.)|
|---|---:|---:|---:|
|**C2TSP**|3.56±3.24 (6.3)|4.20±0.79 (3.9)|0.60±0.40 (0.6)|
|DIFUSCO|63.80±32.78 (17.1)|66.53±25.92 (8.6)|18.71±6.86 (2.1)|
|Fast-T2T|62.05±32.67 (0.5)|62.43±25.36 (0.3)|18.59±8.36 (0.1)|
|DIMES|59.46±32.56 (0.2)|59.36±24.57 (0.1)|18.82±7.24 (0.0)|
|UTSP|61.16±32.71 (0.1)|61.52±24.83 (0.1)|15.84±8.95 (0.0)|

**Conclusion.** C2TSP remains effective on non-Euclidean instances: its gap stays at the few-percent level (1.4–2.1% on Dataset A; 0.6–4.2% on Dataset B), and outperforms every baseline. This advantage is consistent with the model design: a non-Euclidean target cost matrix cannot generally be recovered exactly from coordinates, whereas C2TSP's rooted 1-tree family and HK equilibration operate directly on the cost matrix.

**Limitation to graphs with n≤1000.** Regarding the reviewer's additional question about the limitation to graphs with at most n=1000 nodes, there are two reasons. First, evaluation up to n=1000 follows the standard practice used in related studies, including Fast-T2T and UTSP. Second, although methods such as DIFUSCO and DIMES have reported results on larger instances, some baseline implementations like DIFUSCO encounter out-of-memory errors in our evaluation environment, as also reflected in Table 2 of the manuscript. Extending all baseline implementations to larger instances through memory optimization is beyond the scope of this study.

**Reference**

[1] Bringmann, Karl, et al. ''Random Shortest Paths: Non-Euclidean Instances for Metric Optimization Problems.''

[2] Klootwijk, Stefan, et al. ''Probabilistic Analysis of Optimization Problems on Generalized Random Shortest Path Metrics.''

[3] Baniasadi, Pouya, et al. ''A New Benchmark Set for Traveling Salesman Problem and Hamiltonian Cycle Problem.''

[4] Gower, John C. ''Some Distance Properties of Latent Root and Vector Methods Used in Multivariate Analysis.''

[5] De Leeuw, Jan. ''Applications of Convex Analysis to Multidimensional Scaling.''
