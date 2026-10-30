# FRAME Human-Evaluation Analysis

This module generates the three human-validation figures and one summary table for the FRAME benchmark extraction study.

The analysis is for benchmark quality: domain experts compare both `input.txt` and `output.txt` against the original paper PDF and rate the full paper-level benchmark datapoint. The current FRAME human-evaluation scores are therefore overall input/output extraction ratings, not separate input-only and output-only ratings.

## Expected Rating Data

The script accepts either long-form or wide-form files.

### Wide Form

Use this for evaluator workbooks similar to `Evaluation.xlsx`:

```csv
expert_id,doi_norm,title,artifact_type,Correctness,Completeness,Coherence,Faithfulness
expert_1,10_1007_example_2026_001,Example title,combined,5,4,5,5
```

Use `artifact_type=combined` when one score evaluates the complete input+output benchmark datapoint. If a future study collects separate ratings for `input.txt` and `output.txt`, use `input` and `output`, or separate workbook sheets named `input` and `output`.

### Long Form

```csv
expert_id,doi_norm,title,artifact_type,criterion,score
expert_1,10_1007_example_2026_001,Example title,combined,Correctness,5
expert_1,10_1007_example_2026_001,Example title,combined,Completeness,4
```

## Generate A Template

From the workspace root:

```bash
/data/mguru/04_Finetuning/finetune/bin/python \
  fine_tune_llm_post_processing/evaluation/human_eval/analyze_human_eval.py \
  --write-template fine_tune_llm_post_processing/evaluation/human_eval/human_eval_template.csv
```

## Run The Analysis

CSV files work with the current environment:

```bash
/data/mguru/04_Finetuning/finetune/bin/python \
  fine_tune_llm_post_processing/evaluation/human_eval/analyze_human_eval.py \
  --ratings expert_1.csv expert_2.csv expert_3.csv \
  --output-dir fine_tune_llm_post_processing/evaluation/human_eval/results
```

Excel files require `openpyxl`:

```bash
/data/mguru/04_Finetuning/finetune/bin/pip install openpyxl
```

Then run the same command with `.xlsx` paths.

## Outputs

The script writes:

- `normalized_human_ratings.csv`: tidy rating table used for all analyses.
- `human_eval_validation_report.md`: DOI matching, duplicate rows, and missing rating units.
- `human_eval_summary_table.csv`: numeric table for the manuscript.
- `human_eval_summary_table.md`: manuscript-friendly Markdown version of the table.
- `human_eval_manuscript_notes.md`: draft captions and results language.
- `figure_1_likert_distribution.png/.pdf`: Likert distribution by criterion for the overall input/output extraction target.
- `figure_2_inter_expert_reliability.png/.pdf`: pairwise QWK, three-rater reliability, exact agreement, and adjacent agreement.
- `figure_3_document_quality_landscape.png/.pdf`: DOI-level quality heatmap with disagreement outlines.

## Notes

- Ratings should be integers from 1 to 5.
- Normalized DOI keys are matched against `final_input.json` and `final_output.json`.
- If the file has no `expert_id` column, the expert ID is inferred from the filename.
- If the file has no `artifact_type` column, the script treats ratings as `combined` unless it can infer a separate `input` or `output` target from sheet names, filenames, or prefixed columns such as `input_correctness`.
- Adjacent agreement means the maximum expert-score difference for a DOI/evaluation-target/criterion unit is no more than one point.