![](images/cover.jpeg)

## Data Preprocessing & Exploratory Data Analysis (EDA)

The `EDA.ipynb` notebook covers data cleaning, feature engineering, and exploratory data analysis across the portfolio, profile, and transcript datasets.

### Data Preprocessing
- **Portfolio Dataset**:
  - Unpacked and expanded the `channels` list into one-hot encoded dummy variables (`email`, `mobile`, `social`, `web`).
  - Renamed `id` to `offer_id` for consistency across datasets.
  - Mapped lengthy 32-character offer IDs to concise aliases (e.g., `B1`–`B4` for BOGO, `D1`–`D4` for Discount, `I1`–`I2` for Informational).

- **Profile Dataset**:
  - Converted `became_member_on` to datetime format.
  - Renamed `id` to `customer_id`.
  - Handled missing demographic values (where `age == 118`) by creating a `valid` boolean flag to easily filter out incomplete records.
  - One-hot encoded `gender` (`F`, `M`, `O`).
  - Engineered categorical feature groups:
    - **`age_group`**: `<25`, `25-35`, `35-45`, `45-55`, `55-65`, `65+`
    - **`income_group`**: `<50k`, `50k-80k`, `80-100k`, `100k+`
    - **`membership_length_group`**: `<0.5 yr`, `0.5-1 yr`, `1-2 yr`, `2-5 yr`, `5+ yr`

- **Transcript Dataset**:
  - Extracted `offer_id` and `amount` values from the nested `value` dictionary column.
  - One-hot encoded event types (`offer_received`, `offer_viewed`, `offer_completed`, `transaction`).
  - Renamed `person` to `customer_id`.

- **Data Integration**:
  - Left-joined `transcript`, `profile`, and `portfolio` into a single dataset (`df_merged.csv`) filtered to valid customer records.

### Exploratory Data Analysis
- **Automated Reporting**: Generated automated EDA reports using [Sweetviz](https://pypi.org/project/sweetviz/) (`sweetviz_report.html`) and Pandas Profiling (`Pandas Profile Report.html`).
- **Offer Performance Analysis**: Evaluated receive, view, and completion rates grouped by offer type (`bogo`, `discount`, `informational`), customer age demographics, and income groups.

## Model Building & Evaluation

The [`model.ipynb`](https://github.com/kirawei2025/starbucks-capstone-project/blob/main/model.ipynb) notebook details the feature prep, hyperparameter tuning, model training, and comparative evaluation for predicting customer offer response (`offer_viewed`).

### Model Preprocessing & Feature Engineering
- **Filtering**: Removed transaction records (`offer_id` as NaN) and non-responder entries where neither `offer_received` nor `offer_viewed` occurred. Excluded non-specified gender entries (`O == True`).
- **Feature Derivation & Encoding**:
  - Calculated **`membership_days`** relative to `2019-01-01`.
  - Aggregated event counts (`offer_received`, `offer_viewed`) by customer and offer.
  - Converted gender flags (`F`, `M`) to integer indicators.
  - One-hot encoded categorical columns including `offer_type`, `offer_id`, `duration`, `reward`, and `difficulty`.
- **Feature Scaling**: Applied `MinMaxScaler` across continuous features (`income`, `membership_days`, and `age`).
- **Correlation Analysis**: Conducted Spearman correlation analysis and rendered a heatmap across numerical and encoded categorical features.

---

### Machine Learning Models & Hyperparameter Tuning
Five classification algorithms were trained and evaluated using **5-Fold Cross-Validation** optimized for **ROC-AUC** scores (`scoring='roc_auc'`) via `GridSearchCV`:

1. **XGBoost Classifier** (`XGBClassifier`)
   - **Tuned Parameters**: `colsample_bytree: 0.8`, `learning_rate: 0.1`, `max_depth: 5`, `n_estimators: 100`, `subsample: 1`
   - **Best CV ROC-AUC Score**: **`0.8641`**

2. **Random Forest Classifier** (`RandomForestClassifier`)
   - **Tuned Parameters**: `max_depth: 10`, `max_features: 'sqrt'`, `min_samples_split: 5`
   - **Best CV ROC-AUC Score**: **`0.8613`**
   - **Feature Importance**: Evaluated cumulative importance scores to identify key variables driving target prediction.

3. **K-Nearest Neighbors** (`KNeighborsClassifier`)
   - **Tuned Parameters**: `metric: 'manhattan'`, `n_neighbors: 50`, `weights: 'uniform'`
   - **Best CV ROC-AUC Score**: **`0.8569`**

4. **Logistic Regression** (`LogisticRegression`)
   - **Tuned Parameters**: `C: 0.1`, `l1_ratio: 1`, `solver: 'liblinear'`
   - **Best CV ROC-AUC Score**: **`0.8389`**
   - **Coefficient Analysis**: Extracted feature coefficients to determine directional impact on offer view probability.

5. **Support Vector Classifier** (`SVC`)
   - **Tuned Parameters**: `C: 100`, `gamma: 'scale'`, `kernel: 'linear'`
   - **Best CV ROC-AUC Score**: **`0.8338`**

---

### Key Findings & Model Evaluation Metrics
- **Performance Evaluation**: Models were evaluated on test data ($30\%$ split) using standard metrics: **Accuracy**, **Precision**, **Recall**, **F1-Score**, and **ROC-AUC Score**.
- **Best Performing Model**: **XGBoost Classifier** achieved the highest cross-validation ROC-AUC score (`0.8641`), closely followed by **Random Forest** (`0.8613`).
- **Artifacts & Visualizations Generated**:
  - `correlation_heatmap.png`
  - `confusion_matrix_rf.png` & `confusion_matrix_lr.png`
  - `auc_roc_curve.png` & `auc_roc_curve_lr.png`
  - `feature_importance_rf.png` & `feature_importance_lr.png`

## Repo structure
```
├── data
│   ├── df_merged.csv
│   ├── portfolio.json
│   ├── profile.json
│   └── transcript.json
├── docs
│   ├── _config.yml
│   ├── images
│   │   ├── b1_by_expense.png
│   │   ├── b2_by_expense.png
│   │   ├── b2_gender.png
│   │   ├── b3_by_expense.png
│   │   ├── b3_by_income.png
│   │   ├── b3_gender.png
│   │   ├── d1_gender.png
│   │   ├── expense_age.png
│   │   ├── expense_gender.png
│   │   ├── expense_income.png
│   │   ├── expense.png
│   │   ├── offer_distro.png
│   │   └── population.png
│   └── index.md
├── EDA.ipynb
├── images
│   ├── clustering
│   │   ├── cluster_duration.png
│   │   ├── cluster_offer_type.png
│   │   ├── cluster_platform.png
│   │   ├── kmeans_cluster_duration.png
│   │   ├── kmeans_cluster_offer_type.png
│   │   └── kmeans_cluster_platform.png
│   ├── cover.jpeg
│   ├── eda
│   │   ├── correlation_heatmap.png
│   │   ├── offer_metrics_by_age.png
│   │   ├── offer_metrics_by_distribution_channel.png
│   │   ├── offer_metrics_by_income.png
│   │   ├── offer_metrics_by_membership_length.png
│   │   ├── offer_metrics_by_offer_types.png
│   │   └── scatter_completeRate.png
│   └── model
│       ├── accuracy_scors.png
│       ├── auc_roc_curve_lr.png
│       ├── auc_roc_curve_models.png
│       ├── auc_roc_curve_rf.png
│       ├── confusion_matrix_KNeighborsClassifier.png
│       ├── confusion_matrix_lr.png
│       ├── confusion_matrix_models.png
│       ├── confusion_matrix_rf.png
│       ├── confusion_matrix_SVC.png
│       ├── confusion_matrix_XGBClassifier.png
│       ├── correlation_heatmap_rf.png
│       ├── feature_importance_lr.png
│       ├── feature_importance_rf.png
│       └── feature_importance_xgboost.png
├── model.ipynb
├── Pandas Profile Report.html
├── README.md
└── sweetviz_report.html
```