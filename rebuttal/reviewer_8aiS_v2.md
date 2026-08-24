**(W1) Is C2TSP itself free from decoder dependence?**

No.
All learning-based TSP methods evaluated in our experiments rely on repair-based decoding to convert their learned representations into feasible tours. **Our critique of prior works** is not about whether decoders are used or not, but that the learned representation and the decoder's contribution are difficult to separate in their works[1]. **In contrast, C2TSP enables this separation:** the representation can be judged with no decoder at all, and compared fairly at any fixed decoder.

**Evidence with no decoder.** Table 3 and Figure 1 (middle) of the paper evaluate the learned representation with no search at all: directly on the learned marginals, C2TSP attains 0.875 optimal-edge coverage and 0.90 top-2 concentration.

**Evidence at fixed decoders (Table 1).** Under a fixed decoder, C2TSP leads in 20 of 25 cells, including L1 (repair only, no search), and its performance margin is largest where the decoder is weakest.

**Evidence at the strongest decoder (Table 2).** LKH-3 can be viewed as the strongest decoder. It poses the reviewer's concern in its most extreme form: *once the decoder is this strong, does the upstream learning part still matter at all?*, and Table 2 is our answer: Yes, in both directions. When the learned signal is used only shallowly (H1, an initial tour), learning indeed barely matters. Every method is within noise of vanilla LKH-3, since local search quickly escapes any starting tour. However, when the learned signal drives LKH-3's candidate machinery (H2 → H4), the representation becomes the deciding factor: heatmap-based representations degrade LKH-3 by one to three orders of magnitude at n≥500, whereas C2TSP improves on vanilla LKH-3 for n≤ 500 at comparable wall time and stays within ∼1‰ of it at every size. A strong decoder therefore does not make learning irrelevant, and it makes the quality of the learned representation decide whether the learned signal helps the solver or degrades its performance.

We will sharpen the wording in the Introduction section to clarify that our claim concerns attribution at fixed decoders and at the marginal level, not independence from decoding.

**(W2) The presentation is hard to follow.**

We thank the reviewer for these concrete suggestions and will revise the paper accordingly. We answer each point directly here and describe the corresponding revision.

**(1) Undefined terms in the experiments section.** Greedy degree-2 repair is the standard search-free decoder for edge-based outputs and LKH-3 is the state-of-the-art heuristic TSP solver.

**Revision.**
In the revision, both are defined at first use in the Experiments section. We will check the section so that every named component will be explained at their first appearances.

**(2) The overall algorithm.** C2TSP is a four-stage pipeline: (i) a GNN reads the instance and predicts a residual edge field; (ii) the projected residual tilts the true edge costs D, and the smoothed Held–Karp equilibration solves for node duals under which the rooted 1-tree Gibbs distribution has expected degree two at every node; (iii) K certificate-guided sharpening and re-equilibration steps reduce the certificate upper bound on the residual non-tour mass; (iv) the resulting exact edge marginals define the unsupervised training loss ⟨ D,μ⟩, differentiated through the equilibrium via the implicit function theorem, and are decoded into tours at test time.

**Revision.**
In the revision we will add a pipeline figure at the start of the Methodology section depicting these four stages.

**(3) TSP definition**. In the revision we will state the problem explicitly at the start of the Methodology section: given a complete graph with edge costs D, find a Hamiltonian cycle minimizing $⟨ D,x_H⟩$. We will also elaborate briefly on the significance of solving TSP as many NP-hard problems can be reduced to TSP.

**(W3) Test beyond small, uniform, Euclidean instances.**

We thank the reviewer for this comment and have added a TSPLIB benchmark[2] covering both (1) structured node distributions and (2) non-Euclidean metrics. The first helps demonstrate that our method can be generalized to challenging TSP instances that was not covered in the training data, and the second helps demonstrate that our method can generalize to cases even if the strongest assumption is relaxed (the non-euclidean cases). Taken together, these results provide stronger evidence that explicitly incorporating solution structure and algorithmic processes into the learning pipeline is critical to improving performance. Finally, regarding the concern about instance size, we would like to clarify that TSP-1000 already represents a very large problem scale for this line of research. It is the largest test size considered in comparable learning-based TSP studies, including Fast-T2T[4] and UTSP[5].

**Setup of our new experiments.** All methods use the same TSP-100 trained checkpoints as in the main text, applied zero-shot with no fine-tuning. Aligned with Table 1 of the paper, we evaluate on every TSPLIB instance with fewer than 1000 nodes: 50 structured instances and 27 non-Euclidean instances. Gaps are computed against the published TSPLIB optima.

**Zero-shot transfer to TSPLIB.** Cells report the mean optimality gap (%) ± standard deviation across the instances, and the per-instance wall time in seconds (in parentheses).

|Decoder|Structured (50 inst.), L1|Structured (50 inst.), L2 ×100|Non-Euclidean (27 inst.), L1|Non-Euclidean (27 inst.), L2 ×100|
|---|---:|---:|---:|---:|
|**C2TSP**|**10.61**±7.27 (3.55)|**2.46**±1.93 (3.55)|**7.36**±3.96 (2.90)|**1.81**±1.94 (2.90)|
|DIFUSCO|47.50±29.86 (2.35)|11.72±15.34 (2.35)|31.98±26.96 (1.84)|7.22±11.42 (1.84)|
|Fast-T2T|27.72±17.16 (0.06)|7.06±8.48 (0.07)|25.67±17.39 (0.05)|5.13±6.97 (0.06)|
|DIMES|16.74±5.80 (0.02)|3.42±1.76 (0.03)|19.13±10.29 (0.02)|3.95±3.66 (0.02)|
|UTSP|36.06±15.00 (0.02)|9.18±6.18 (0.02)|29.09±21.96 (0.01)|7.26±7.77 (0.01)|

**Results.**
The ranking from the main text transfers to TSPLIB: C2TSP attains the best mean gap in all cells. Full per-instance results will be included in the appendix of the revision.

**Explanation.**
Heatmap-based baselines learn edge scores directly from the training distribution. In C2TSP, the network only learns a residual edge field, while connectivity and expected degree balance are supplied by the rooted 1-tree family and HK equilibration. Because these guarantees are independent of the node distribution and distance metric, they reduce the amount of global structure the network must learn and may improve zero-shot transfer.

**(Q1) Why LKH-3 rather than LKH or LKH-2?**

Because the choice does not change the solver’s behavior on symmetric TSP, and LKH-3 is the standard backend in the recent literature.

First, for pure symmetric TSP, LKH-3 runs the same core Lin–Kernighan–Helsgaun machinery as LKH-2. The choice therefore does not alter the solver's behavior. We simply use the latest version of the LKH family. Second, LKH-3 is the reference heuristic used throughout the recent learning-for-TSP literature (e.g., DIFUSCO[3] and Fast-T2T[4]), and all methods in Table 2 are integrated with the same LKH-3 backend, ensuring a fair comparison.

**(Q2) Clarify the issues listed in the Weaknesses section.**

We have addressed each concern listed in the Weaknesses section. Please see our point-by-point responses above for details.

**(Formatting Concern) Abstracts of the metainfo and the PDF do not match.**

We apologize for the confusion. The metadata abstract was submitted before the PDF abstract was finalized. The PDF abstract was subsequently updated before the submission deadline to reflect the finalized experiments, while the proposed method remained unchanged.

**Reference**

[1] Xia, Yifan, et al. ''Position: Rethinking Post-Hoc Search-Based Neural Approaches for Solving Large-Scale Traveling Salesman Problems.'' Proceedings of the 41st International Conference on Machine Learning, JMLR.org, 2024, article 2224.

[2] Reinelt, Gerhard. ''TSPLIB—A traveling salesman problem library.'' ORSA journal on computing 3.4 (1991): 376-384.

[3] Sun, Zhiqing, and Yiming Yang. ''Difusco: Graph-based diffusion solvers for combinatorial optimization.'' Advances in neural information processing systems 36 (2023): 3706-3731.

[4] Li, Yang, et al. ''Fast t2t: Optimization consistency speeds up diffusion-based training-to-testing solving for combinatorial optimization.'' Advances in Neural Information Processing Systems 37 (2024): 30179-30206.

[5] Min, Yimeng, Yiwei Bai, and Carla P. Gomes. "Unsupervised learning for solving the travelling salesman problem." Advances in neural information processing systems 36 (2023): 47264-47278.
