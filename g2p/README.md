Scripts and data used for the grapheme-to-phoneme experiments. 

Data sets are CMUdict (https://github.com/cmusphinx/cmudict) and Nettalk (Sejnowski 1987, see  https://archive.ics.uci.edu/ml/datasets/Connectionist+Bench+%28Nettalk+Corpus%29). For specific data splits, see ./data.

See ./scripts for examples on how to train and evaluate with sockeye.

 - data splits obtained from Shijie Wu, see Wu et al. (2018), (2019) and (2020).
 - hyperparameters for RNNs: as close as possible to to Wu et al. (2018), large setting. 
 - hyperparameters for transformers: as close as possible to Wu et al. (2020), except for dropout. We use dropout=0.3 on Nettalk, dropout=0.2 on CMUdict for all experiments (instead of 0.1 and 0.3 on both)
 - code to calculate PER/WER in ./scripts/eval.py based on: https://github.com/shijie-wu/neural-transducer/blob/master/src/util.py


References:
- Shijie Wu and Ryan Cotterell. 2019. Exact Hard Monotonic Attention for Character-level Transduction. In: Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics, pages 1530–1537, Florence, Italy. ACL
- Shijie Wu, Pamela Shapiro, and Ryan Cotterell. 2018. Hard non-monotonic Attention for Character-level Transduction. In: Proceedings of the 2018 Conference on  Empirical Methods in Natural Language Processing, pages 4425–4438, Brussels, Belgium. ACL.
- Shijie Wu, Ryan Cotterell, and Mans Hulden. 2020. Applying the Transformer to Character-level Transduction. Computing Research Repository, arXiv:2005.10213
- Sejnowski, T.J., and Rosenberg, C.R. 1987. Parallel networks that learn to pronounce English text. In: Complex Systems, 1, 145-168.
