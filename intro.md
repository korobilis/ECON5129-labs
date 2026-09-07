# ECON5129: Statistical Machine Learning

**Computer labs** &middot; Adam Smith Business School, University of Glasgow

Lectures run from 09:00 to 11:00 and the computer lab follows the same day from 14:00 to 16:00. Each lab implements the methods introduced that morning, first from first principles and then with the standard libraries you will use in your own work.

## How the labs work

Every lab is a Jupyter notebook. You can open it directly in Google Colab from the rocket icon at the top of the page, or download it and run it in Anaconda on a lab machine or your own laptop. Nothing needs to be installed for Labs 1 to 6.

Each lab alternates between worked material, which we go through together, and exercises, which you attempt yourself. Solutions are published on this site after each session. Attempt the exercises before reading them: the code is short, and the difficulty is in deciding what to write rather than in typing it.

The labs are not assessed. They exist to make the lecture material concrete and to give you the working knowledge the take-home exercises and the final examination assume.

## Schedule

| Week | Lecture | Lab |
|---|---|---|
| 1 | Regression | Linear regression, from algebra to Bayes |
| 2 | Classification | The Bayes classifier, discriminant analysis and logistic regression |
| 3 | Shrinkage I | Ridge, LASSO and the elastic net |
| 4 | Shrinkage II | Principal components, factors and the sparse-dense debate |
| 5 | Nonlinearity I | Kernels, splines and Gaussian processes |
| 6 | Nonlinearity II | Trees, forests and boosting |
| 7 | Deep Learning I | Neural networks from scratch |
| 8 | Deep Learning II | Networks for macroeconomics and finance |
| 9 | Text as Data I | Representing text, sentiment and topic models |
| 10 | Text as Data II | Attention, transfer learning and text for forecasting |

## Working conventions

Notation follows the lecture slides throughout. Cross-sectional problems are indexed $i = 1, \dots, n$ with $p$ predictors collected in the design matrix $\mathbf{X}$; from the shrinkage labs onward, time series problems are indexed $t = 1, \dots, T$.

Simulated data is used wherever the point is to compare an estimator against a known truth. Real data is used wherever the point is that we do not know it: the FRED-MD monthly macroeconomic panel from Lab 3 onwards, and the transcripts of Federal Open Market Committee meetings in Labs 9 and 10.

One forecasting problem recurs from Lab 3 to Lab 10: predicting the monthly change in the US unemployment rate from 363 macroeconomic predictors. Every method in the course is applied to it, against the same autoregressive benchmark, so that the comparisons across lectures are real comparisons rather than a change of subject.
