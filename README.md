# Amazon Review Classification with Apache Spark

Text classification of Amazon product reviews into product categories using Apache Spark. The assignment combines an RDD implementation of chi-square term ranking with Spark ML pipelines for TF-IDF features, feature selection, and multiclass linear SVM classification.

## Overview

The work has three parts:

1. **RDD feature extraction:** Count documents by category and term, then calculate chi-square statistics to rank informative terms. On the development dataset, the output matched Assignment 1 exactly after formatting adjustments.
2. **DataFrame feature pipeline:** Tokenize review text, remove stop words, create term-frequency vectors, compute IDF weights, and select features.
3. **Classification and evaluation:** Index category labels and train a `OneVsRest` classifier with `LinearSVC`; compare feature selectors and tune `regParam`, `standardization`, and `maxIter` using F1 score.

The report mentions 2,000 chi-square-selected terms and a comparison involving 500 features, but does not give a separate F1 breakdown by feature count. It also reports 561 terms overlapping with the Assignment 1 output (approximately one third of the Assignment 1 terms). These figures should not be interpreted as a measured classification improvement.

## Pipeline

| Stage | Spark ML component | Purpose |
| --- | --- | --- |
| Text preparation | `RegexTokenizer` | Split `reviewText` into tokens using a custom regex. |
| Filtering | `StopWordsRemover` | Remove common stop words. |
| Term counts | `CountVectorizer` | Create sparse term-frequency vectors. |
| Weighting | `IDF` | Produce TF-IDF features. |
| Labels | `StringIndexer` | Encode product categories as numeric labels. |
| Feature selection | `ChiSqSelector` or `VarianceThresholdSelector` | Select informative or sufficiently variable features. |
| Scaling | `Normalizer` | Apply L2 normalization before classification. |
| Classification | `OneVsRest` with `LinearSVC` | Train multiclass linear SVM classifiers. |
| Tuning | `CrossValidator` | Perform three-fold cross-validation, as described in the report. |

## Results

The following values are transcribed from the **two result tables** in the project report. They are reported as F1 scores; the report does not specify whether F1 is macro, micro, or weighted, or clearly distinguish validation from test scores in these tables.

### ChiSqSelector

| `regParam` | `standardization` | `maxIter` | F1 |
| ---: | :---: | ---: | ---: |
| 0.01 | True | 10 | 0.2867 |
| 0.01 | True | 20 | 0.2900 |
| 0.01 | False | 10 | 0.1605 |
| 0.01 | False | 20 | 0.2938 |
| 0.1 | True | 10 | **0.3122** |
| 0.1 | True | 20 | 0.3100 |
| 0.1 | False | 10 | 0.2897 |
| 0.1 | False | 20 | 0.2973 |
| 1.0 | True | 10 | 0.1391 |
| 1.0 | True | 20 | 0.2862 |
| 1.0 | False | 10 | 0.0424 |
| 1.0 | False | 20 | 0.0424 |

**Best listed configuration:** `regParam=0.1`, `standardization=True`, `maxIter=10`, **F1=0.3122**.

### VarianceThresholdSelector

| `regParam` | `standardization` | `maxIter` | F1 |
| ---: | :---: | ---: | ---: |
| 0.01 | True | 10 | 0.3291 |
| 0.01 | True | 20 | 0.3404 |
| 0.01 | False | 10 | 0.1937 |
| 0.01 | False | 20 | 0.2908 |
| 0.1 | True | 10 | 0.3293 |
| 0.1 | True | 20 | **0.3432** |
| 0.1 | False | 10 | 0.1935 |
| 0.1 | False | 20 | 0.0732 |
| 1.0 | True | 10 | 0.3262 |
| 1.0 | True | 20 | 0.3160 |
| 1.0 | False | 10 | 0.1935 |
| 1.0 | False | 20 | 0.0631 |

**Best listed configuration and overall highest reported F1:** `regParam=0.1`, `standardization=True`, `maxIter=20`, **F1=0.3432**. This is 0.0310 F1 points higher than the best listed ChiSqSelector result (0.3122). These are best scores among the displayed configurations, not evidence of performance on an independent test set.

### Interpretation and report corrections

- With standardization enabled, VarianceThresholdSelector scores 0.3160–0.3432 in six listed configurations; ChiSqSelector ranges from 0.1391 to 0.3122. Both selectors can perform poorly for some settings without standardization. The table does not support a blanket claim that ChiSqSelector is more stable across all configurations.
- More iterations do **not** always improve F1: for ChiSqSelector at `regParam=0.1`, `standardization=True`, F1 changes from 0.3122 at 10 iterations to 0.3100 at 20; for VarianceThresholdSelector at `regParam=0.1`, `standardization=False`, it drops from 0.1935 to 0.0732.
- The report's prose says `maxIter=12` for the 0.2900 ChiSqSelector result, while its table says **20**. Its conclusion calls 0.2900 the best ChiSqSelector result, but the table's best is **0.3122**. The tables above follow the tabulated values.

## Source

Based on *Assignment 2: Text Processing and Classification using Apache Spark* by Filip Aleksic, Lucija Aleksic, Anabel Dautovic, Ana Petrovic, and Matyas Prechl. Consult the project report for the original methodology, pipeline diagram, and results.
