## Morphological Inflection

For morphological inflection, we use the 
[CoNLL-SIGMORPHON 2017](https://github.com/sigmorphon/conll2017) shared task dataset.
We perform experiments on all languages in the high and medium resource settings (without Scottish Gaelic to be comparable to previous work).

We preprocess the data to insert a separator token between the morphological
tags and the lemma in the input. We learn morphological inflection on 
character-level and substitute all space characters with the diameter character 
(which does not occur in the training data otherwise). The monotonicity loss is 
then only computed on the positions to the right of the separator token's position.

The input to our models looks like this:
`V V.PTCP PRS <sep> z g j o j`

The output of our models looks like this: 
`d u k e ⌀ z g j u a r`

To download the data and create the format for training and evaluating the models call this script:
```
bash scripts/preprocess.sh
```

To train and evaluate the models presented in the paper call this script:
```
bash scripts/train_and_eval.sh
```

For every language, this script will train the following models with seeds 1, 2 and 3:
* base: a baseline transformer without monotonicity loss
* mono_all_heads: transformer with monotonicity loss on all heads in all layers, lambda=0.1 and margin=0.1
* mono_1_head: a transformer with monotonicity loss on one head in all layers, lambda=0.1 and margin=0.1
* rnn_base: a baseline RNN without monotonicity loss
* rnn_mono: an RNN with monotonicity loss, lambda=0.1, margin=0

You can then look at the evaluation results in the `eval_log` files created in the model folders, e.g. the transformer baseline for high-resource albanian:
```
cat high/albanian/base_1/eval_log
```