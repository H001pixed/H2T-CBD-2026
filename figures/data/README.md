# Local plotting input schemas

This directory contains documentation only. Supply your numerical plotting inputs locally, or pass a separate directory through `--data-dir`. No experimental values or example data rows are distributed.

## Figure 2

`within_corpus_seed_pairs.csv` has these columns:

```text
dataset,protocol,seed,baseline,comparison,qwk_baseline,qwk_comparison,complete
```

Use `feat` or `asap2` for `dataset`, `P-in` for `protocol`, and `True` for complete pairs. Model keys are `blackbox`, `no_anchor`, `h2tcbd`, and `featconcat`. Supply all 10 seed-level QWK values per corpus and model, represented through paired rows. Repeated values for the same corpus, model, and seed must agree.

## Figures 3 and 4

`ars_retention.csv` has these columns:

```text
group_index,evaluation_index,retained,total
```

Provide one row per group and evaluation, for a total of 30 rows. Group indices 0-5 follow the six feature groups in the root README. Evaluation indices 0-4 are ASAP-1 leave-one-prompt-out, ASAP-2.0 leave-one-prompt-out, ASAP-2.0 official split, PERSUADE non-overlapping leave-one-prompt-out, and CELA single-prompt, respectively. `retained` is a count; `total` is the number of runs in that evaluation.

`paired_scoring.csv` has these columns:

```text
evaluation_index,mean_difference,ci_lower,ci_upper
```

Supply one row for each evaluation index. Differences are selective minus full anchoring, with the corresponding 95% interval bounds.

## Figure 5

`case_contributions.csv` has these columns:

```text
role,human_score,model_score,word_count,intercept,pred_unit,length_fluency,lexical_complexity,syntactic_complexity,cohesion_readability
```

Use `high`, `mid`, and `low` for the three roles. The four feature-group columns contain signed coordinate contributions before the sigmoid and score mapping. Supply the intercept and continuous prediction from the same model. Essay text and essay identifiers are not required.
