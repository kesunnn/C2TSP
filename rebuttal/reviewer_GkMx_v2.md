**(W1) Can this method be applied to problems beyond ordinary TSP?**

Yes. Its relevance extends in two ways: TSP itself appears in many practical routing and sequencing problems, and the broader structure-aware pipeline may transfer to other combinatorial problems.

First, TSP is a basic model or core subproblem in many routing and sequencing applications such as delivery routing, and moreover many NP-hard problems can be reduced to TSP such as 3-SAT and machine scheduling. Our method is therefore relevant to a broad range of real-world problems. Second, we believe that our framework provides a **methodological template** for structure-aware learning: identify a tractable combinatorial structure that enforces a key property of feasible solutions, use it as the latent support, and design problem-specific mechanisms to address the remaining gaps. The current rooted 1-tree construction is specific to TSP. However, the same structure-aware pipeline may be useful for other combinatorial problems when an appropriate tractable structure can be identified.

**(W2) State the purpose of LKH-3 experiments.**

We thank the reviewer for this suggestion: the experiment is designed to test whether the learned object maintains near-tour structure, and we take this opportunity to make the purpose of Table 2 explicit. We will state this purpose at the beginning of the Experiments section.

**(M1) Is the rooted 1-tree Gibbs family computable in polynomial time?**

Yes. We agree that ''computable exactly'' is imprecise and will revise it to ''computable exactly in polynomial time'' in this revision.

In detail, for the fixed-root rooted 1-tree Gibbs family, the partition function factorizes into a weighted spanning-tree partition function and a root-edge-pair normalizer. It can be evaluated in $O(n^3)$ time on a dense graph. The edge marginals, expected degrees, and degree variances and covariances required in this work are also computable exactly in polynomial time from the same factorization.

**(M2) Define $d_i(U)$.**

$d_i(U)$ denotes the degree of node $i$ in the rooted 1-tree $U$. In this revision, we will define this notation in the paragraph before Eq. (2).

**(M3) Add missing parentheses in equation references.**

We will add the missing parentheses in these four places, and we will check every equation reference in the manuscript.

**(M4) Cite LKH-3 at first appearance.**

We will cite LKH-3 at its first appearance.

**(Q1) Can the proposed method or its ideas apply to other problems or TSP variants?**

Yes. Our method can be applied to problems that use TSP as a basic model or core subproblem. More broadly, other combinatorial problems may follow our methodological template when an appropriate tractable structural family can be identified. Please refer to our response to W1 for details.
