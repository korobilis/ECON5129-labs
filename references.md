# Data and references

## Datasets

**FRED-MD.** A monthly panel of 126 US macroeconomic series beginning in 1959, maintained by the Federal Reserve Bank of St. Louis. The copy used here is the vintage ending 2025-09 and is frozen in the course repository so that results are reproducible. Each series carries a transformation code that renders it stationary; `econ5129_utils.fredmd_transform` applies them.

> McCracken, M. W. and S. Ng (2016). FRED-MD: A Monthly Database for Macroeconomic Research. *Journal of Business and Economic Statistics* 34(4), 574-589.

**NBER recession indicator (USREC).** Monthly binary indicator of US recession dates as determined by the NBER Business Cycle Dating Committee, obtained from FRED.

**FOMC corpus.** Transcripts of Federal Open Market Committee meetings from 1977 to 2008, used in the text labs. The course repository stores a compact meeting-level derivative of the ConvoKit distribution.

> Tan, C. and L. Lee (2016). Talk it up or play it down? (Un)expected correlations between (de-)emphasis and recurrence of discussion points in consequential U.S. economic policy meetings. *Text As Data*.

## Software

The labs use `numpy`, `pandas`, `scipy`, `matplotlib` and `scikit-learn` throughout, `torch` in the deep learning labs, and `transformers` in the final lab.

## Further reading

- Hastie, T., R. Tibshirani and J. Friedman (2009). *The Elements of Statistical Learning*, 2nd edition. Springer. Freely available from the authors.
- James, G., D. Witten, T. Hastie and R. Tibshirani (2021). *An Introduction to Statistical Learning*, 2nd edition. Springer. Freely available, and the gentler companion to the above.
- Bishop, C. M. (2006). *Pattern Recognition and Machine Learning*. Springer. The reference for the Bayesian material.
- Murphy, K. P. (2022). *Probabilistic Machine Learning: An Introduction*. MIT Press.
