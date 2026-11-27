# Ringkasan Metrik Evaluasi Model

| Model               |   CV_5Fold_F1_Mean |   CV_5Fold_F1_Std |   Test_Accuracy |   Test_Precision_Weighted |   Test_Recall_Weighted |   Test_F1_Weighted |
|:--------------------|-------------------:|------------------:|----------------:|--------------------------:|-----------------------:|-------------------:|
| Logistic Regression |             0.3242 |            0.0004 |          0.4919 |                    0.2419 |                 0.4919 |             0.3243 |
| Decision Tree       |             0.3646 |            0.0062 |          0.37   |                    0.3762 |                 0.37   |             0.3729 |
| Random Forest       |             0.3833 |            0.0135 |          0.3981 |                    0.3586 |                 0.3981 |             0.3684 |
| K-Nearest Neighbors |             0.3725 |            0.0098 |          0.3656 |                    0.3648 |                 0.3656 |             0.3637 |
| Naive Bayes         |             0.3688 |            0.0066 |          0.455  |                    0.3592 |                 0.455  |             0.3535 |

# Confusion Matrix Hasil Benchmark (`random_state=42`, `test_size=0.2`)

## Logistic Regression

|               |   Pred_High |   Pred_Low |   Pred_Medium |
|:--------------|------------:|-----------:|--------------:|
| Actual_High   |           0 |          0 |           494 |
| Actual_Low    |           0 |          0 |           319 |
| Actual_Medium |           0 |          0 |           787 |

## Decision Tree

|               |   Pred_High |   Pred_Low |   Pred_Medium |
|:--------------|------------:|-----------:|--------------:|
| Actual_High   |         153 |        105 |           236 |
| Actual_Low    |         106 |         68 |           145 |
| Actual_Medium |         240 |        176 |           371 |

## Random Forest

|               |   Pred_High |   Pred_Low |   Pred_Medium |
|:--------------|------------:|-----------:|--------------:|
| Actual_High   |         103 |         48 |           343 |
| Actual_Low    |          76 |         35 |           208 |
| Actual_Medium |         205 |         83 |           499 |

## K-Nearest Neighbors

|               |   Pred_High |   Pred_Low |   Pred_Medium |
|:--------------|------------:|-----------:|--------------:|
| Actual_High   |         163 |         78 |           253 |
| Actual_Low    |         132 |         48 |           139 |
| Actual_Medium |         299 |        114 |           374 |

## Naive Bayes

|               |   Pred_High |   Pred_Low |   Pred_Medium |
|:--------------|------------:|-----------:|--------------:|
| Actual_High   |          29 |         20 |           445 |
| Actual_Low    |          27 |         18 |           274 |
| Actual_Medium |          60 |         46 |           681 |
