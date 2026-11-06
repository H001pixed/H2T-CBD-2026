# Data preparation

Obtain the original sources from their providers:

| Corpus | Source | Local input |
|---|---|---|
| ASAP-1 | [Kaggle competition](https://www.kaggle.com/c/asap-aes) | `workspace/data/sources/asap1_official/training_set_rel3.tsv` |
| ASAP-2.0 | [Author repository](https://github.com/scrosseye/ASAP_2.0) | `workspace/data/sources/asap2_official/ASAP_2_Final_github_train.zip` and `ASAP_2_Final_github_test.zip` |
| PERSUADE 2.0 | [Author repository](https://github.com/scrosseye/persuade_corpus_2.0) | `workspace/data/sources/persuade2_official/persuade_corpus_2.0_train.csv` |
| CELA | [Author repository](https://github.com/gzutxy/CELA) | `workspace/data/sources/cela_official/CELA.csv` |

Run the following commands from the repository root after placing the source files in these locations:

```bash
python Code/01_Features/Prepare_Feat.py --root workspace --datasets feat
python Code/01_Features/Prepare_Feat.py --root workspace --datasets asap2 --include-official-test
python Code/01_Features/Fix_Feat.py --root workspace --datasets feat asap2
python Code/01_Features/Prepare_PERSUADE.py --source workspace/data/sources/persuade2_official/persuade_corpus_2.0_train.csv --existing-root workspace/data/01_raw_datasets --output workspace/data/01_raw_datasets/persuade_nonoverlap
python Code/01_Features/Prepare_CELA.py --source workspace/data/sources/cela_official/CELA.csv --existing-root workspace/data/01_raw_datasets --output workspace/data/01_raw_datasets/cela_single_prompt
python Code/14_ARS_Supplement/Extract_Supplement_Features.py --root workspace
```

ASAP-1 uses the fixed prompt-specific score conversion in `Prepare_Feat.py`. ASAP-2.0 and PERSUADE retain scores from 1 to 6. CELA retains scores from 1 to 5 with half-point steps. PERSUADE preparation excludes the seven prompts shared with ASAP-2.0, checks text overlap against ASAP-1 and ASAP-2.0, and removes within-source duplicate text.

Prepared tables contain `id`, `text`, `score`, `essay_set`, `partition`, `text_hash`, and the 19 computed feature columns. Each table is saved as `workspace/data/01_raw_datasets/<dataset>/data.csv`, with accompanying metadata. The internal identifiers are `feat`, `asap2`, `persuade_nonoverlap`, and `cela_single_prompt`.

The feature extractor uses NLTK tokenization and tagging resources, TextBlob, textstat, and pyspellchecker. Install the Python dependencies from `requirements.txt`; install the NLTK resources requested by the feature extractor before feature computation.
