Scripts and data used for grapheme-to-phoneme experiments:

 - Data splits obtained from Shijie Wu, see Wu et al. (2018), (2019) and (2020)
 - hyperparameters for RNNs: according to Wu et al. (2018), large setting. 
 - hyperparameters for transformers: according to Wu et al. (2020), except for dropout. We use dropout=0.3 on Nettalk, dropout=0.2 on CMUdict for all experiments (instead of 0.1 and 0.3 on both)
 - code to calculate PER taken from: https://github.com/shijie-wu/neural-transducer/blob/master/src/util.py


References:
- Shijie Wu and Ryan Cotterell. 2019. Exact Hard Monotonic Attention for Character-level Transduction. In: Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics, pages 1530–1537, Florence, Italy. ACL
- Shijie Wu, Pamela Shapiro, and Ryan Cotterell. 2018. Hard non-monotonic Attention for Character-level Transduction. In: Proceedings of the 2018 Conference on  Empirical Methods in Natural Language Processing, pages 4425–4438, Brussels, Belgium. ACL.
- Shijie Wu, Ryan Cotterell, and Mans Hulden. 2020. Applying the Transformer to Character-level Transduction. Computing Research Repository, arXiv:2005.10213
