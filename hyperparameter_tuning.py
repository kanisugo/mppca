import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from mppca import MPPCA
from data_utils import load_magnetic_csv, extract_features

if __name__ == "__main__":
    CSV_PATH = "/to/file/path"
    try:
        df = load_magnetic_csv(CSV_PATH)
    except FileNotFoundError:
        print("Error: magnetic.csv not found.")
        exit()

    data = extract_features(df)

    # Scale Full Dataset
    mean = np.mean(data, axis=0)
    std = np.std(data, axis=0)
    data_scaled = (data - mean) / std

    # Apply Magnetic Weight
    MAG_WEIGHT = 5.0
    data_scaled[:, 3] *= MAG_WEIGHT

    # Subsample for CV
    np.random.seed(42)
    subset_idx = np.random.choice(len(data_scaled), 5000, replace=False)
    X_subset = data_scaled[subset_idx]

    K_FOLDS = 5
    kf = KFold(n_splits=K_FOLDS, shuffle=True, random_state=42)

    M_values = [i for i in range(3, 20)]
    q_values = [1, 2, 3]
    results = []

    for M in M_values:
        for q in q_values:
            fold_test_lls = []
            fold_bics = []
            for train_index, val_index in kf.split(X_subset):
                X_train_fold = X_subset[train_index]
                X_val_fold = X_subset[val_index]
                model = MPPCA(n_components=M, latent_dim=q, max_iter=30)
                try:
                    model.fit(X_train_fold)
                    val_ll = model.score(X_val_fold) / len(X_val_fold)
                    fold_test_lls.append(val_ll)
                    bic = model.get_bic(X_train_fold)
                    fold_bics.append(bic)
                except np.linalg.LinAlgError:
                    fold_test_lls.append(-np.inf)
                    fold_bics.append(np.inf)

            results.append({
                'M': M,
                'q': q,
                'Avg Val LL': np.mean(fold_test_lls),
                'Std Val LL': np.std(fold_test_lls),
                'Avg BIC': np.mean(fold_bics)
            })

    res_df = pd.DataFrame(results)
    print(res_df.head())