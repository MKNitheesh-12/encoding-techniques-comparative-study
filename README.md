# Label Encoding vs One-Hot Encoding — A Comparative Study

A hands-on comparison of the two most common techniques for converting categorical data into numerical form, demonstrated on a small hospital appointment dataset.

The goal is not just to show *how* each method works, but to make visible what each one does to the shape and the meaning of your data — and when that matters.

---

## Overview

Machine learning models cannot consume text. Categorical columns like `Gender`, `City`, or `AppointmentType` must be converted to numbers first. Two standard approaches:

| | Label Encoding | One-Hot Encoding |
|---|---|---|
| Tool | `sklearn.preprocessing.LabelEncoder` | `pandas.get_dummies()` |
| Output | One integer per category | One binary column per category |
| Shape change | None (5 × 4 → 5 × 4) | Expands (5 × 4 → 5 × 9) |
| Implied order | Yes (often false) | No |
| Memory | Compact | Grows with cardinality |

---

## Dataset

A small synthetic hospital dataset, built in-notebook — no external files required.

| Patient_ID | Gender | AppointmentType | City |
|---|---|---|---|
| P1 | M | Routine | New York |
| P2 | F | Emergency | Los Angeles |
| P3 | F | Routine | Chicago |
| P4 | M | Follow-up | New York |
| P5 | F | Emergency | Chicago |

All four columns are of dtype `object`. No missing values.

---

## Results

### After Label Encoding

Shape stays at **(5, 4)**. Each category is replaced by an integer, assigned alphabetically by default.

```
  Patient_ID  Gender  AppointmentType  City
0         P1       1                2     2
1         P2       0                0     1
2         P3       0                2     0
3         P4       1                1     2
4         P5       0                0     0
```

Mappings produced:

- `Gender`: F → 0, M → 1
- `AppointmentType`: Emergency → 0, Follow-up → 1, Routine → 2
- `City`: Chicago → 0, Los Angeles → 1, New York → 2

The catch: nothing about this data says New York (2) outranks Chicago (0). The encoder invented an ordering that a linear or distance-based model will take seriously.

### After One-Hot Encoding

Shape expands to **(5, 9)**.

- `Gender` (2 categories) → `Gender_F`, `Gender_M`
- `AppointmentType` (3 categories) → `AppointmentType_Emergency`, `AppointmentType_Follow-up`, `AppointmentType_Routine`
- `City` (3 categories) → `City_Chicago`, `City_Los Angeles`, `City_New York`

Total = 1 identifier + 2 + 3 + 3 = 9 columns. Values come back as booleans, which behave as 1/0 in numeric operations.

No false hierarchy is introduced — but the feature space grew by more than double on a dataset with only three small categorical columns.

---

## When to use which

**Label Encoding fits:**
- Ordinal data with a real order (education level, severity, satisfaction rating)
- Tree-based models (decision trees, random forests, gradient boosting), which split on values rather than assuming linear relationships
- High-cardinality features where dimensionality is a genuine constraint

**One-Hot Encoding fits:**
- Nominal data with no intrinsic order (city, gender, product category)
- Linear models, neural networks, and distance-based algorithms (k-NN, SVM), where arbitrary integers mislead the model
- Features with a manageable number of unique categories

---

## Known drawbacks

**Label Encoding**
- Implies ordinality where none exists, which can degrade model performance
- Unsuitable for linear models, which read the integers as a continuous scale
- The mapping is arbitrary and depends on internal sorting — it can shift if new categories appear

**One-Hot Encoding**
- Dimensionality explosion on high-cardinality features (ZIP codes, product SKUs)
- Sparsity — most values are zero, which is inefficient for some algorithms
- Multicollinearity in linear models (the dummy variable trap), avoidable with `drop_first=True`

---

## Running the notebook

```bash
git clone https://github.com/<your-username>/<your-repo>.git
cd <your-repo>
pip install pandas numpy scikit-learn
jupyter notebook Comparative_Study_of_Label_Encoding_vs_One_Hot_Encoding.ipynb
```

Or open it directly in Google Colab — no local setup needed.

**Requirements:** Python 3.8+, `pandas`, `numpy`, `scikit-learn`

---

## Repository contents

```
.
├── Comparative_Study_of_Label_Encoding_vs_One_Hot_Encoding.ipynb   # Notebook with code and outputs
├── Comparative_Study_of_Label_Encoding_vs_One-Hot_Encoding.pdf     # Written report
└── README.md
```

---

## Conclusion

Both techniques solve the same problem in fundamentally different ways. Label Encoding is compact but risks introducing relationships that do not exist. One-Hot Encoding preserves the nominal nature of categories at the cost of dimensionality.

The usual practice: one-hot for nominal features, ordinal encoding for genuinely ordered ones. For high-cardinality columns, consider frequency encoding or learned embeddings instead of either.

---

## Possible extensions

- Train a linear model and a tree-based model on both encodings and compare performance
- Add `drop_first=True` and examine the effect on multicollinearity
- Scale the dataset up to test where one-hot encoding stops being practical
- Benchmark alternatives: target encoding, frequency encoding, `CatBoostEncoder`
