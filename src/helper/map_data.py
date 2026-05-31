def convert_raw_to_macro(y_raw_list):
    """Converte predições y_raw de volta para y_macro (0, 1, 2)"""
    macro_map = {
        "c1_p1": 0,
        "c2_p1": 1,
    }
    y_macro_pred = []
    for y in y_raw_list:
        if y in macro_map:
            y_macro_pred.append(macro_map[y])
        else:
            y_macro_pred.append(2)  # Tudo que não for c1_p1 ou c2_p1 é "Ruim" (2)
    return np.array(y_macro_pred)


def train_test_split_hierarchical(X, y_macro, y_raw, test_size=0.2, seed=42):
    np.random.seed(seed)

    train_idx = []
    test_idx = []

    subclasses = np.unique(y_raw)

    for c in subclasses:
        idx = np.where(y_raw == c)[0]
        np.random.shuffle(idx)

        n = len(idx)

        # split proporcional
        n_test = int(np.round(n * test_size))

        # garante presença mínima em ambos os conjuntos (se n >= 2)
        if n >= 2:
            n_test = max(1, min(n - 1, n_test))
        else:
            n_test = 0

        n_train = n - n_test

        train_idx.extend(idx[:n_train])
        test_idx.extend(idx[n_train:])

    train_idx = np.array(train_idx)
    test_idx = np.array(test_idx)

    np.random.shuffle(train_idx)
    np.random.shuffle(test_idx)

    y_raw_arr = np.array(y_raw)
    return (
        X[train_idx],
        X[test_idx],
        y_macro[train_idx],
        y_macro[test_idx],
        y_raw_arr[train_idx],
        y_raw_arr[test_idx],
    )


def load_and_prepare_data():
    """Carrega e pré-processa os dados uma única vez."""
    print("[1] Carregando dados...")
    df = pd.read_csv("data/Dados.csv", header=None)

    count_by_class = df.iloc[:, -1].value_counts()
    print(count_by_class)

    y_raw = df.iloc[:, -1]

    # ── Agrupar em 3 macro-classes ──────────────────────────────────
    def _map_to_group(label: str) -> int:
        if label.startswith("c1"):
            return 0  # Ótimo
        elif label.startswith("c2"):
            return 1  # Normal
        else:
            return 2  # Ruim

    species_names = ["Ótimo", "Normal", "Ruim"]
    species_int = y_raw.map(_map_to_group).values
    print(f"   Macro-classes: {species_names}")
    print(f"   Distribuição: {dict(zip(*np.unique(species_int, return_counts=True)))}")

    # remover colunas indesejadas + label
    cols_to_drop = [5, 12, 13, 0, 3, 6, 14, 15, 16, 17, 18, 19, 22, 23]
    all_cols = list(range(df.shape[1] - 1))  # excluindo label
    valid_cols = [c for c in all_cols if c not in cols_to_drop]
    feature_names = [f"F_{c}" for c in valid_cols]

    X = df.drop(df.columns[cols_to_drop + [-1]], axis=1).values.astype(float)
    print(f"   {X.shape[0]} amostras | dim={X.shape[1]}")

    # Normalizando os dados (Z-score)
    X = (X - np.mean(X, axis=0)) / np.std(X, axis=0)

    return X, species_int, y_raw.values, species_names, feature_names
