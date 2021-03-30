Code and scripts for the paper <insert url>. 
Changes to sockeye include the additional loss, drophead (https://www.aclweb.org/anthology/2020.findings-emnlp.178/), phoneme-error-rate as a metric, negative positional encodings for morphology tags, scoring functions to print monotonicity and the percentage of target positions with increased average attention, and also a function to print a attention scores graphically.

Data used in the paper:
  - Grapheme-to-Phoneme: CMUdict and NETTalk, data splits as in Wu and Cotterell (2019), Wu et al. (2018) and Wu et al. (2020)
  - Morphological Inflection: 
  - Transliteration: NEWS 2015 shared task (Zhang 2015), own data split (TODO: can be made public?)
  - Dialect Normalization: 



References:
Shijie Wu and Ryan Cotterell. 2019. Exact Hard Monotonic Attention for Character-level Transduction. In: Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics, pages 1530–1537, Florence, Italy. ACL
Shijie Wu, Pamela Shapiro, and Ryan Cotterell. 2018. Hard non-monotonic Attention for Character-level Transduction. In: Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing, pages 4425–4438, Brussels, Belgium. ACL.
Shijie Wu, Ryan Cotterell, and Mans Hulden. 2020. Applying the Transformer to Character-level Transduction. Computing Research Repository, arXiv:2005.10213
Min Zhang, Haizhou Li, Rafael E. Banchs, and A. Kumaran. 2015.  Whitepaper of NEWS 2015 Shared Task on Machine Transliteration. In: Proceedings of the Fifth Named Entity Workshop, pages 1-9, Beijing, China. ACL.


