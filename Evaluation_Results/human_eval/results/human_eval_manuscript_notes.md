# Manuscript Notes for FRAME Human Validation

## Suggested Results Text

Three domain experts independently evaluated each FRAME benchmark datapoint as a whole, comparing both `input.txt` and `output.txt` against the source paper PDF. The ratings therefore represent overall input/output extraction quality for each datapoint, rather than separate scores for input keys and output keys. Across all expert ratings, 90.0% of scores were 4 or 5 on the five-point Likert scale, indicating that the extracted fields were generally judged suitable for benchmark evaluation and VLM fine-tuning. Inter-expert reliability was assessed using Gwet's AC2 with quadratic weights (Gwet, 2014), which is robust to the high-score concentration that makes traditional Kappa metrics unreliable in this setting. The mean pairwise AC2 was 0.91, and 80.0% of comparable rating units differed by no more than one point (adjacent agreement). The full machine-readable summary retains QWK and Krippendorff alpha as supplementary diagnostics, but AC2 is emphasized because kappa-style chance models are unstable for these highly skewed ordinal ratings.

## Figure 1 Caption

Distribution of domain-expert Likert ratings for FRAME benchmark datapoint quality. Each rating evaluates the input and output extraction together for a paper-level benchmark datapoint, grouped by Correctness, Completeness, Coherence, and Faithfulness. The annotation at the end of each row reports the percentage of ratings greater than or equal to 4, while open markers indicate the corresponding high-score share for each individual expert.

## Figure 2 Caption

Inter-expert agreement for FRAME human validation. The AC2 heatmap reports pairwise Gwet's AC2 with quadratic weights for each expert pair, with an additional mean AC2 column for each criterion. The agreement panel reports exact agreement and adjacent agreement. Adjacent agreement counts rating units for which expert scores differ by no more than one point on the 1-5 scale.

## Figure 3 Caption

Document-level quality landscape for the FRAME benchmark subset evaluated by domain experts. Rows correspond to evaluated paper titles, columns correspond to overall input/output extraction criteria, and each cell reports the mean expert rating for the paper-level datapoint. Cells outlined in black indicate score ranges of at least two points across experts, highlighting papers or criteria that may need manual review.

## Combined Figure Caption

Human validation of FRAME benchmark datapoints. (A) Distribution of expert Likert ratings by criterion, with end-of-row annotations showing the percentage of ratings greater than or equal to 4 and open markers showing individual expert high-score shares. (B) Pairwise and mean Gwet's AC2 by criterion. (C) Exact and adjacent inter-expert agreement by criterion.

## Table Caption

Human validation summary for FRAME benchmark datapoints. Each row summarizes expert ratings for a qualitative criterion. Ratings assess agreement between the combined input/output benchmark extraction and the original scientific paper, not the performance of a downstream VLM on the benchmark. Gwet's AC2 is reported as the primary chance-corrected agreement metric; QWK and Krippendorff alpha are retained in the CSV summary for reference.
