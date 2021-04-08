##Code and scripts for the paper "On Biasing Transformer Attention Towards Monotonicity".

Changes to sockeye include the additional loss, drophead (Zhou 2020), phoneme-error-rate as a metric, negative positional encodings for morphology tags, scoring functions to print monotonicity and the percentage of target positions with increased average attention, and also a function to visualize attention scores with the loss graphically.

**Data used in the paper**:
  - Grapheme-to-Phoneme: CMUdict and NETTalk, data splits as in Wu and Cotterell (2019), Wu et al. (2018) and Wu et al. (2020)
  - Morphological Inflection: [CoNLL-SIGMORPHON 2017](https://github.com/sigmorphon/conll2017) shared task dataset
  - Transliteration: NEWS 2015 shared task (Zhang 2015), own data split (ids published)
  - Dialect Normalization: data split as in [Swiss German UD](https://github.com/noe-eva/SwissGermanUD/tree/master/monotonicity_splits)

**Note**: requirements.txt contains some packages needed for preprocessing/visualization. To install the necessary packages for sockeye, use the relevant file for your cuda version in sockeye/requirements/*. 

**References**:
- Shijie Wu and Ryan Cotterell. 2019. Exact Hard Monotonic Attention for Character-level Transduction. In: Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics, pages 1530–1537, Florence, Italy. ACL
- Shijie Wu, Pamela Shapiro, and Ryan Cotterell. 2018. Hard non-monotonic Attention for Character-level Transduction. In: Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing, pages 4425–4438, Brussels, Belgium. ACL.
- Shijie Wu, Ryan Cotterell, and Mans Hulden. 2020. Applying the Transformer to Character-level Transduction. Computing Research Repository, arXiv:2005.10213
- Min Zhang, Haizhou Li, Rafael E. Banchs, and A. Kumaran. 2015.  Whitepaper of NEWS 2015 Shared Task on Machine Transliteration. In: Proceedings of the Fifth Named Entity Workshop, pages 1-9, Beijing, China. ACL.
- Wangchunshu  Zhou, Tao Ge, Furu Wei, Ming Zhou,and Ke Xu. 2020. Scheduled  DropHead: A regularization method for transformer models. In: Findings of the Association for Computational Linguistics: EMNLP 2020, pages 1971–1980, Online. ACL.
- Noëmi  Aepli  and  Simon  Clematide.  2018. Parsing Approaches for Swiss German. In Proceedings of the 3rd Swiss Text Analytics Conference, Winterthur, Switzerland.
