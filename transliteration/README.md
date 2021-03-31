## Transliteration

For transliteration, we experiment on the 
[NEWS2015 shared task data](https://www.aclweb.org/anthology/W15-3901.pdf) shared task dataset. 
We received the train and development data for 11 script pairs from Wu et al. (2018), (2019) and (2020). 
The labels for the test set are not publicly available.

We randomly sampled 1,000 names from the official training sets as our development sets for script 
pairs with more than 20,000 training examples and 100 for script pairs with less training 
examples. The official development sets are used for testing. This repository contains the 
ids of our own train / dev / test split. Our split
can be recreated with the original train and development XML files (see preprocessing below).

We learn morphological inflection on 
character-level and substitute all space characters with the diameter character 
(which does not occur in the training data otherwise).

The input to our models looks like this:
`B A D N E R`

The output of our models looks like this: 
`バ ー ド ナ ー`

The preprocessing script needs to process XML files and requires `lxml`:
```
pip install lxml
```

To create the format for training and evaluating the models, place the original XML files 
in the respective script pair subfolders the `data` directory and call this script:
```
bash scripts/preprocess.sh
```

The evaluation script uses `nltk` to compute the minimum edit distance:
```
pip install nltk
```

To train and evaluate the models presented in the paper call this script:
```
bash scripts/train_and_eval.sh
```

For every language, this train and evaluate script will train the following models with seeds 1, 2 and 3:
* base: a baseline transformer without monotonicity loss
* mono_all_heads: transformer with monotonicity loss on all heads in all layers, lambda=0.1 and margin=0.1
* mono_1_head: a transformer with monotonicity loss on one head in all layers, lambda=0.1 and margin=0.1
* rnn_base: a baseline RNN without monotonicity loss
* rnn_mono: an RNN with monotonicity loss, lambda=0.1, margin=0

You can then look at the evaluation results in the `eval_log` files created in the model folders, e.g. the transformer baseline for Arabic-English:
```
cat data/ArEn/base_1/eval_log
```

References:
* Shijie Wu, Pamela Shapiro, and Ryan Cotterell. 2018. Hard non-monotonic Attention for Character-level Transduction. In: Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing, pages 4425–4438, Brussels, Belgium. ACL.
* Shijie Wu and Ryan Cotterell. 2019. Exact Hard Monotonic Attention for Character-level Transduction. In: Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics, pages 1530–1537, Florence, Italy. ACL
* Shijie Wu, Ryan Cotterell, and Mans Hulden. 2020. Applying the Transformer to Character-level Transduction. Computing Research Repository, arXiv:2005.10213