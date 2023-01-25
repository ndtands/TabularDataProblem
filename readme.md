# [Dataset](https://archive.ics.uci.edu/ml/datasets/adult)

# Model
## 1. TabTransformer
Paper: [TabTransformer-2020](https://arxiv.org/pdf/2012.06678.pdf)

![alt text](images/TabTransformer.png)

1. Numerical features are normalised and passed forward
2. Categorical features are embedded
3. Embeddings are passed through Transformer blocks N times to get contextual embeddings
4. Contextual categorical embeddings are concatenated with numerical features
5. Concatenation gets passed through MLP to get the required prediction

## 2. FT-Transformer
Paper:  [FT-Transformer-2021](https://arxiv.org/pdf/2106.11959v2.pdf)

![alt text](images/FT-Transformer.png)

The difference from TabTransformer model is adding Embedding layer for continuous feature

## 3. FT-Transformer Variances
Paper:  [Embeddings for Numerical Features in Tabular Deep Learning-2022](https://arxiv.org/pdf/2203.05556.pdf)

![alt text](images/FT-Transformer-variance.png)
- Periodic Embeddings ( using concat sinv and cosv )
- Piecewise Linear Encoding (Quantile Binning)
- Target Binning Approach

Target binning involves using the decision tree algorithm to assist in construction of the bins. As we saw above, the quantile approach splits our feature into the bins of equal width but it might be suboptimal in certain cases. A decision tree would be able to find the most meaningful splits with regards to the target. For example, if there was more target variance towards the larger values of the feature, majority of the bins could move to the right.

## 4. Experiment with transformers
![alt text](images/evaluation.png)


Evaluation models in testing dataset
| Type model      | is_explainable| ROC AUC | PR AUC |  ACC | F1-score | Acc Possitive |
| ----------- | ----------- | ----------- | ----------- | ----------- | ----------- |----------- |
| TabTransformer (No numerical Embedding)      | No |0.8962       | 0.7361 | 0.8479 | 0.7653| 0.63 |
| Linear numerical Embedding   | Yes |0.9001       | 0.7393 | 0.8485 | 0.7796| 0.66 |
| Periodic numerical Embedding | Yes |0.8925       | 0.7275 | 0.8443 | 0.765 | 0.77 |
| Piecewise Linear Encoding (Quantile Binning)   | Yes |0.9117       | 0.7879 | 0.86 | 0.7942| 0.79  |
| Target Binning Approach   | yes |0.9122       | 0.7864 |  0.8592 | 0.7905| 0.79 |