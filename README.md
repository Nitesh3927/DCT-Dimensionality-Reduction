# 📘 Performance Analysis of PCA, LDA, and DCT for Dimensionality Reduction

## 🧩 Abstract
This project presents a comparative analysis of three dimensionality reduction techniques — **Principal Component Analysis (PCA)**, **Linear Discriminant Analysis (LDA)**, and **Discrete Cosine Transform (DCT)** — applied to high-dimensional data.  
The primary goal is to evaluate each method’s ability to reduce dimensionality while maintaining model performance, minimizing error, and reducing computation time.

---

## 🎯 Objective
- To perform dimensionality reduction on high-dimensional datasets using PCA, LDA, and DCT.  
- To evaluate and compare model performance based on metrics such as RMSE, MAE, R², and training time.  
- To analyze DCT as an efficient and interpretable alternative to classical techniques.

---

## 🧠 Introduction
High-dimensional datasets often suffer from redundancy, noise, and high computational costs — a phenomenon known as the **curse of dimensionality**.  
Dimensionality reduction techniques transform data into a compact, informative form by removing irrelevant or correlated features.  
This project compares:
- **PCA**: An unsupervised linear technique that captures maximum variance.
- **LDA**: A supervised linear technique that maximizes class separability.
- **DCT**: A transformation-based technique that converts spatial data to frequency domain, retaining key energy coefficients.

---

## ⚙️ Methodology

1. **Data Loading & Splitting**
   - Read dataset, identify target variable, split into training and test sets.

2. **Preprocessing**
   - Handle missing values using mean imputation.
   - Standardize data using Z-score normalization (fit on training set only).

3. **Baseline Model**
   - Train a **Linear Regression** model on full feature set for reference.

4. **PCA Pipeline**
   - Apply PCA to training data, retain top *k* components.
   - Train Linear Regression using reduced components.

5. **LDA Pipeline**
   - Discretize continuous target into quantile-based classes.
   - Fit LDA to find *n_classes − 1* components.
   - Train Linear Regression model on transformed features.

6. **DCT Pipeline**
   - Apply 1-D Discrete Cosine Transform (Type-II) to each sample.
   - Retain first *k* coefficients (high energy compaction).
   - Train Linear Regression on transformed data.

7. **Evaluation & Metrics**
   - Compute **RMSE**, **MAE**, **R²**, and **Training Time**.
   - Record PCA explained variance and DCT energy retention.

8. **Comparison & Visualization**
   - Compare all models through tabular metrics and visual plots (RMSE, MAE, R², training time, and feature counts).

---

## 📊 Results and Discussion

- **Accuracy**: DCT and PCA achieved comparable predictive accuracy, while DCT required fewer coefficients.
- **Training Time**: DCT showed reduced training time due to compact frequency representation.
- **Error Metrics**: RMSE and MAE for DCT-based models were close to PCA, outperforming LDA in most cases.
- **Energy Retention**: The majority of data variance was preserved using only a subset of DCT coefficients.

### Key Insights
| Technique | Type | Accuracy (R²) | RMSE | MAE | Training Time | Remarks |
|------------|------|---------------|------|------|----------------|----------|
| PCA | Unsupervised | High | Low | Low | Moderate | Sensitive to scaling |
| LDA | Supervised | Moderate | High | High | Moderate | Requires labeled data |
| DCT | Transform-based | High | Low | Low | Fastest | Energy-efficient & noise-tolerant |

---

## 🧮 Mathematical Foundation (Brief)
The **Discrete Cosine Transform (DCT)** represents data as a sum of cosine functions of varying frequencies:

\[
X_k = \sum_{n=0}^{N-1} x_n \cos\left[\frac{\pi}{N}\left(n + \frac{1}{2}\right)k\right], \quad k = 0,1,...,N-1
\]

- Most of the energy is concentrated in **low-frequency coefficients**, allowing efficient dimensionality reduction by retaining only the first few terms.
- The transformation is **linear**, **orthogonal**, and **real-valued**, making it computationally efficient and interpretable.

---

## 🧰 Tools and Libraries
- **Language:** Python
- **Libraries:** NumPy, Pandas, Scikit-learn, SciPy, Matplotlib
- **Notebook:** `DCT_Performance_Loop.ipynb`

---

## 💻 Execution Steps

1. **Install Dependencies**
   ```bash
   pip install numpy pandas scikit-learn scipy matplotlib
   ```

2. **Run the Notebook**
   - Open `DCT_Performance_Loop.ipynb` in Jupyter Notebook or VS Code.
   - Execute cells sequentially to:
     - Load dataset
     - Apply PCA, LDA, and DCT pipelines
     - Evaluate and visualize performance metrics

---

## 🧾 Conclusion
The study concludes that **DCT-based dimensionality reduction** is a **powerful, efficient, and interpretable** approach for handling high-dimensional data.  
Compared to PCA and LDA, DCT offers:
- Faster computation,
- Comparable accuracy,
- Effective noise reduction, and
- High information retention in fewer components.

Thus, DCT serves as a practical alternative for preprocessing and feature compression in modern machine learning pipelines.
