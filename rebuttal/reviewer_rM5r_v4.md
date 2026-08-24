**(W1) Are the better LKH integration results natural?**

No. The structural alignment alone is not sufficient. The advantage comes from the quality of the learned signal, which captures solution structure within the connected rooted 1-tree family.
Two observations show that the advantage does not follow automatically from this alignment.

First, vanilla LKH-3 (H0) already uses α-nearness candidates computed by the classical Held–Karp subgradient procedure. If producing “candidates of the type LKH expects” were sufficient, C2TSP integration could at best match H0. Instead, replacing LKH's internally computed α-nearness with C2TSP's learned candidates and duals (H2→H4) improves on vanilla LKH-3 for n≤500 at comparable wall time, and stays within ∼1‰ of it at every size. The experiment therefore tests signal quality: C2TSP recovers 1-tree structure accurate enough to outperform the solver's own hand-tuned construction of the same object.

Second, and most directly, the additional baselines in our response to Q1 hold the H2 budget fixed: non-learned α-nearness gives gaps (‰) of 5.93-6.23, versus 0.11-0.20 for C2TSP. The learned edge ranking therefore adds information beyond structural compatibility.

**(W2) Test on TSPLIB, structured distribution.**

We thank the reviewer for this point and have added a TSPLIB benchmark[1] covering both structured node distributions and non-Euclidean metrics. The results show that the performance advantage extends beyond uniformly sampled Euclidean instances. Full per-instance results will be included in the appendix of the revision.

**Setup.** All methods use the same TSP-100 trained checkpoints as in the main text, applied zero-shot with no fine-tuning. Aligned with Table 1 of the paper, we evaluate on every TSPLIB instance with fewer than 1000 nodes: 50 structured instances and 27 non-Euclidean instances.

**Zero-shot transfer to TSPLIB.** Cells report the mean optimality gap (%) ± standard deviation across the instances, and the per-instance wall time in seconds (in parentheses).

|Method|Structured (50 inst.), L1 ↓|Structured (50 inst.), L2 ×100 ↓|Non-Euclidean (27 inst.), L1 ↓|Non-Euclidean (27 inst.), L2 ×100 ↓|
|---|---:|---:|---:|---:|
|**C2TSP**|**10.61**±7.27 (3.55)|**2.46**±1.93 (3.55)|**7.36**±3.96 (2.90)|**1.81**±1.94 (2.90)|
|DIFUSCO|47.50±29.86 (2.35)|11.72±15.34 (2.35)|31.98±26.96 (1.84)|7.22±11.42 (1.84)|
|Fast-T2T|27.72±17.16 (0.06)|7.06±8.48 (0.07)|25.67±17.39 (0.05)|5.13±6.97 (0.06)|
|DIMES|16.74±5.80 (0.02)|3.42±1.76 (0.03)|19.13±10.29 (0.02)|3.95±3.66 (0.02)|
|UTSP|36.06±15.00 (0.02)|9.18±6.18 (0.02)|29.09±21.96 (0.01)|7.26±7.77 (0.01)|

**Results.**
The ranking from the main text transfers to TSPLIB: C2TSP attains the best mean gap in all cells.

**Explanation.**
Heatmap-based baselines learn edge scores directly from the training distribution. In C2TSP, the network only learns a residual edge field, while connectivity and expected degree balance are supplied by the rooted 1-tree family and HK equilibration. Because these guarantees are independent of the node distribution and distance metric, they reduce the amount of global structure the network must learn and improve zero-shot transfer.

**(W3) Report the training wall time.**

We thank the reviewer for this suggestion, and we will add a full training-cost accounting to the appendix of the revision. All models were trained for 100 epochs on the same machine on the same TSP-100 dataset:

**Model size and training time comparison.**

|Method|Params|Train inst.|Wall time|
|---|---:|---:|---|
|C2TSP|123,750|2,048|1 h 16 m 40 s|
|DIFUSCO|5,333,762|2,048|52 m 05 s|
|Fast-T2T|5,333,762|2,048|1 h 16 m 12 s|
|DIMES|64,321|2,048|15 m 25 s|
|UTSP|36,072|2,048|1 m 51 s|

We summarize here three major findings. (1) The reviewer's observation is correct, and the table quantifies it: each C2TSP forward pass runs one HK equilibration plus K sharpening re-equilibrations , and this dominates its training cost. (2) The absolute cost nonetheless stays within the same range as the baselines: essentially identical to Fast-T2T and ∼1.5× DIFUSCO on identical data. (3) The comparison slightly favors DIFUSCO and Fast-T2T: DIFUSCO and Fast-T2T are supervised and additionally require pre-generation of optimal tours as labels for training, and this cost is not included in the wall time. It is small at $n=100$ but grows with instance size (exponentially), whereas C2TSP trains entirely label-free.

**(W4) Is performance sensitive to the choice of the fixed root?**

We performed additional experiments and verified that the performance is not affected by the fixed root node. Specifically, we evaluated the trained model on the TSP-50 testset 50 times, once with each node selected as the root: the gaps range from 1.528% to 1.717% with a standard deviation of 0.046%, so performance is stable across all root choices.

All numbers in the table below are optimality gaps in %.

|Roots evaluated|Mean|Std.|Min|Max|Median|
|---|---:|---:|---:|---:|---:|
|Root 0--49|1.626|0.046|1.528|1.717|1.634|

**(Q1) Does the LKH robustness come from learning or from structural compatibility?**

From learning. We ran the requested controls. Structurally compatible but non-learned candidates are far worse than C2TSP.

**Cells report gaps (‰) against Concorde optimal value, 100 instances per size.**

|Candidate source|TSP100|TSP200|TSP500|
|---|---:|---:|---:|
|Random 5 of 20-NN|666.44|568.66|345.84|
|Distance 5-NN|7.30|7.93|6.86|
|Non-learned 1-tree (α-nearness)|6.04|6.23|5.93|
|LKH-3 vanilla (α-nearness + HK ascent)|0.38|0.22|0.36|
|Distance 5-NN + C2TSP initial tour|2.49|2.67|4.35|
|**C2TSP (H2)**|**0.11**|**0.13**|**0.20**|

We summarize here three major findings. (1) Structural compatibility is not the explanation. Random 5-of-20-NN candidates are equally compatible with LKH's machinery, yet yield gaps of 346–666. Relative to the distance-only 5-NN ranking under identical H2 settings, replacing the ranking score by C2TSP's learned marginal is worth 34--66×. (2) Non-learned 1-tree candidates yield gaps of 5.9–6.2. Adding Held–Karp subgradient ascent to it recovers vanilla LKH-3, so the classical ascent is worth 16--28×. (3) C2TSP's initial tour on top of distance-only candidates reaches only 2.49–4.35: the learned information is primarily carried by the candidate ranking. The full table along with the discussion will be added to the appendix.

**(Q2) Is performance stable across roots, and does averaging over roots help?**

Performance is stable across roots, and averaging does not help.
The 50-root evaluation in our response to W4 gives a gap range of 1.528%–1.717% with a standard deviation of 0.046% on TSP-50. Averaging model outputs over 10 roots leaves the L1–L3 decoders essentially unchanged and clearly hurts L4 decoder.

**Setup.** We conducted an evaluation using one trained TSP50 model on 1,000 test instances. The single-root baseline uses root 0. For the 10-root ensemble, we average the model outputs before decoding, while retaining root 0 as the decoder root. The table reports mean optimality gaps in %.

|Root setting|L1|L2×1|L2×10|L2×100|L3|L4|
|---|---:|---:|---:|---:|---:|---:|
|Single root 0|6.100|4.229|1.328|1.316|2.966|1.920|
|Average of roots 0--9|6.176|4.274|1.324|1.302|2.988|**3.973**|

**Results.** Root averaging has little effect on L1--L3 under the same decoding budget. Only L4 deteriorates from 1.920% to 3.973%. Since L4 selects a root-0 rooted 1-tree using perturbed edge costs, averaging costs across roots disrupts the root-dependent score structure and makes the 1-tree selection less coherent. We therefore retain single-root inference, and will add this result to the appendix.

**(Q3) Are forward-map constraints better than loss penalties in terms of final performance?**

Yes. In the requested ablation, penalty-only training gives a 20.81% gap, forward HK alone gives 2.21%, and adding the penalty on top of forward HK brings no further gain. We will also include this additional results to the appendix of the camera-ready version of our paper.

**Setup.**
We disable stage 2 (edge sharpening) to isolate the effect of forward equilibration. For penalty-based variants, the training loss is

$L=cost+5‖r‖₂²$

where r is the vector of degree-2 residuals for each graph. The table below reports $‖r‖₂$ averaged across test graphs. We compare (a) forward HK only, (b) penalty only, and (c) forward HK with the penalty. Results use the TSP50 test set, and all gaps are reported in %.

|Method|Gap (%)|Mean residual $‖r‖₂$|
|---|---:|---:|
|(a) Forward HK only|2.206|0.0341|
|(b) Penalty only|20.811|0.6959|
|(c) Forward HK + penalty|2.212|0.0337|

**Result.** Forward HK provides a clear measurable advantage over penalty-only training: removing the forward equilibration increases the gap from approximately 2.21% to 20.81% and substantially worsens the residual. In contrast, (a) and (c) achieve nearly identical tour quality and residuals, indicating that the additional penalty provides no measurable benefit once HK equilibration is imposed in the forward map.

**(Q4) Add multi-seed results for the core tables.**

We appreciate this suggestion and we repeated the full evaluation of Tables 1 and 2 of the paper with 10 random seeds on the same test instances. No ranking in Table 1 or Table 2 flips under any seed, so all conclusions are unchanged. The full multi-seed tables will be updated in the revision.

There are two additional remarks worth further elaboration. (1) Many cells are deterministic by design: C2TSP, DIMES, and UTSP inference is deterministic, and the L1/L2 decoders contain no randomness. (2) Where variance exists, it is negligible relative to the reported effects: the largest seed std in the entire table is ± 0.28% (UTSP L3, TSP500), and every C2TSP cell has a std of at most ± 0.07%.

**Reference**

[1] Reinelt, Gerhard. ``TSPLIB—A traveling salesman problem library.'' ORSA journal on computing 3.4 (1991): 376-384.
