def plot_accuracy_comparison(
    all_results, species_names, save_path, topology_text=None, x_labels=None
):
    """
    Gráfico 1: Acurácia de treino vs teste por realização,
    com linha de média móvel contínua entre as topologias.
    """
    n_runs = len(all_results)
    acc_trains = [r["acc_train"] * 100 for r in all_results]
    acc_tests = [r["acc_test"] * 100 for r in all_results]

    x = np.arange(1, n_runs + 1)

    # Tamanho fixo adequado para slides (aprox 16:9), sem alargar ao infinito
    fig, ax = plt.subplots(figsize=(14, 6))

    # Agrupar os resultados por topologia para calcular a média móvel
    topo_indices = {}
    for i, r in enumerate(all_results):
        t_idx = r.get("topo_idx", i)
        if t_idx not in topo_indices:
            topo_indices[t_idx] = []
        topo_indices[t_idx].append(i)

    num_topos = len(topo_indices)

    if num_topos > 1:
        # Média móvel (rolling) ao longo de todas as realizações sequenciais.
        # A janela é o número de treinos por topologia para suavizar dentro de cada bloco.
        window_size = max(2, len(list(topo_indices.values())[0]))

        train_ma = (
            pd.Series(acc_trains).rolling(window=window_size, min_periods=1).mean()
        )
        test_ma = pd.Series(acc_tests).rolling(window=window_size, min_periods=1).mean()

        # Amostrar o valor da média móvel no último treino de cada topologia
        topo_keys = list(topo_indices.keys())
        ma_train_vals = []
        ma_test_vals = []
        topo_labels = []

        for t_idx in topo_keys:
            indices = topo_indices[t_idx]
            last_i = indices[-1]  # último treino da topologia
            ma_train_vals.append(train_ma.iloc[last_i])
            ma_test_vals.append(test_ma.iloc[last_i])
            # Buscar label da topologia a partir dos x_labels ou gerar
            if x_labels is not None:
                topo_labels.append(x_labels[indices[0]].split("-")[0])  # ex: "T1"
            else:
                topo_labels.append(f"T{t_idx}")

        # Eixo X sequencial: 1, 2, 3, ..., num_topos
        x_topo = np.arange(1, num_topos + 1)

        ax.plot(
            x_topo,
            ma_train_vals,
            color="#0D47A1",
            linewidth=2.5,
            marker="o",
            markersize=5,
            label=f"Treino (Média Móvel w={window_size})",
        )
        ax.plot(
            x_topo,
            ma_test_vals,
            color="#B71C1C",
            linewidth=2.5,
            marker="s",
            markersize=5,
            label=f"Teste (Média Móvel w={window_size})",
        )

        # Adicionar textos sobre os pontos das médias móveis
        if num_topos <= 200:
            for i in range(num_topos):
                val_train = ma_train_vals[i]
                val_test = ma_test_vals[i]

                ax.text(
                    x_topo[i],
                    val_train - 1.5,
                    f"{val_train:.1f}%",
                    color="#0D47A1",
                    fontsize=7,
                    ha="center",
                    va="bottom",
                    fontweight="bold",
                    alpha=0.9,
                )
                ax.text(
                    x_topo[i],
                    val_test + 1.5,
                    f"{val_test:.1f}%",
                    color="#B71C1C",
                    fontsize=7,
                    ha="center",
                    va="top",
                    fontweight="bold",
                    alpha=0.9,
                )

        # Configurar eixo X com labels de topologia
        ax.set_xticks(x_topo)
        rotation_angle = 90 if num_topos > 15 else (45 if num_topos > 8 else 0)
        ha_align = "center" if rotation_angle == 0 else "right"
        ax.set_xticklabels(
            topo_labels, rotation=rotation_angle, ha=ha_align, fontsize=9
        )

    else:
        # Se for apenas 1 topologia (ex: best_results), mostramos os valores reais de cada treino
        ax.plot(
            x,
            acc_trains,
            color="#2196F3",
            linewidth=2,
            marker="o",
            markersize=5,
            label="Treino (Real)",
        )
        ax.plot(
            x,
            acc_tests,
            color="#F44336",
            linewidth=2,
            marker="s",
            markersize=5,
            label="Teste (Real)",
        )

        if n_runs <= 30:
            for i in range(n_runs):
                ax.text(
                    x[i],
                    acc_trains[i] - 1.0,
                    f"{acc_trains[i]:.1f}%",
                    color="#2196F3",
                    fontsize=7,
                    ha="center",
                    va="top",
                    fontweight="bold",
                )
                ax.text(
                    x[i],
                    acc_tests[i] + 1.0,
                    f"{acc_tests[i]:.1f}%",
                    color="#F44336",
                    fontsize=7,
                    ha="center",
                    va="bottom",
                    fontweight="bold",
                )

        ax.set_xticks(x)

    mean_train = np.mean(acc_trains)
    mean_test = np.mean(acc_tests)

    ax.axhline(
        y=mean_train,
        color="#2196F3",
        linestyle=":",
        linewidth=2,
        alpha=0.8,
        label=f"Média Geral Treino ({mean_train:.2f}%)",
    )
    ax.axhline(
        y=mean_test,
        color="#F44336",
        linestyle=":",
        linewidth=2,
        alpha=0.8,
        label=f"Média Geral Teste ({mean_test:.2f}%)",
    )

    ax.set_title(
        f"Evolução da Acurácia por Realização — {n_runs} Treinos ({K_BMU}-BMU)",
        fontsize=14,
        fontweight="bold",
    )
    ax.set_xlabel("Topologia", fontsize=12)
    ax.set_ylabel("Acurácia (%)", fontsize=12)

    # Ajustar limites do Y para os textos não cortarem
    y_min = min(min(acc_trains), min(acc_tests), mean_train, mean_test)
    y_max = max(max(acc_trains), max(acc_tests), mean_train, mean_test)
    ax.set_ylim(max(0, y_min - 4), min(100, y_max + 4))

    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(fontsize=10, loc="lower right")

    if topology_text:
        props = dict(boxstyle="round", facecolor="white", alpha=0.9, edgecolor="gray")
        ax.text(
            1.01,
            1.0,
            topology_text,
            transform=ax.transAxes,
            fontsize=9,
            verticalalignment="top",
            bbox=props,
        )
        plt.subplots_adjust(right=0.85)
    else:
        plt.tight_layout()

    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"   Gráfico de acurácia salvo em: {save_path}")
    plt.close(fig)


def plot_sorted_accuracies(all_results, save_path, x_labels=None, custom_title=None):
    """
    Gráfico Extra: Gráfico de barras horizontais ordenado pela Acurácia de Teste.
    Ideal para visualizar muitos treinos de uma vez.
    """
    n_runs = len(all_results)

    # Agrupar dados
    data = []
    for i, r in enumerate(all_results):
        label = x_labels[i] if x_labels else f"Realização {i+1}"
        data.append((r["acc_test"], r["acc_train"], label))

    # Ordenar por acurácia de teste
    data.sort(key=lambda x: x[0])

    acc_tests = [d[0] * 100 for d in data]
    acc_trains = [d[1] * 100 for d in data]
    labels = [d[2] for d in data]

    # Altura dinâmica para acomodar muitos resultados
    fig_height = max(6, n_runs * 0.25)
    fig, ax = plt.subplots(figsize=(10, fig_height))

    y = np.arange(n_runs)
    height = 0.4

    ax.barh(y - height / 2, acc_trains, height, color="#2196F3", label="Treino")
    ax.barh(y + height / 2, acc_tests, height, color="#F44336", label="Teste")

    title_str = (
        custom_title if custom_title else f"Ranking de Acurácia — {n_runs} Treinos"
    )
    ax.set_title(title_str, fontsize=13, fontweight="bold")
    ax.set_xlabel("Acurácia (%)", fontsize=11)

    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8)

    # Limitar x até um pouco mais que o máximo para caber o texto
    max_acc = max(max(acc_tests), max(acc_trains))
    ax.set_xlim(0, min(100 + 10, max_acc + 10))

    # Adicionar textos nas barras de teste e treino
    for i in range(n_runs):
        val_test = acc_tests[i]
        val_train = acc_trains[i]

        # Texto para Teste
        ax.text(
            val_test + 0.5,
            y[i] + height / 2,
            f"{val_test:.1f}%",
            va="center",
            fontsize=7,
            color="#F44336",
            fontweight="bold",
        )

        # Texto para Treino
        ax.text(
            val_train + 0.5,
            y[i] - height / 2,
            f"{val_train:.1f}%",
            va="center",
            fontsize=7,
            color="#2196F3",
            fontweight="bold",
        )

    ax.grid(True, axis="x", linestyle="--", alpha=0.4)
    ax.legend(fontsize=9, loc="lower right")

    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"   Gráfico de ranking salvo em: {save_path}")
    plt.close(fig)


def plot_mse_comparison(all_results, save_path):
    """
    Gráfico 2: Dois subplots lado a lado.
    - Esquerda: EQM médio por época de TREINO (média de todas as realizações).
    - Direita: EQM médio por época de TESTE (média de todas as realizações).
    """
    n_runs = len(all_results)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    max_epochs = max(len(r["mse_train_history"]) for r in all_results)
    epochs = list(range(1, max_epochs + 1))

    # ── Esquerda: EQM médio de TREINO ──
    raw_mse_train = [r["mse_train_history"] for r in all_results]
    all_mse_train = np.array(
        [np.pad(arr, (0, max_epochs - len(arr)), mode="edge") for arr in raw_mse_train]
    )
    mean_train = np.mean(all_mse_train, axis=0)
    std_train = np.std(all_mse_train, axis=0)

    axes[0].plot(
        epochs,
        mean_train,
        color="purple",
        linewidth=2,
        marker="o",
        markersize=3,
        label=f"EQM Treino (média de {n_runs} realizações)",
    )
    axes[0].fill_between(
        epochs,
        mean_train - std_train,
        mean_train + std_train,
        alpha=0.2,
        color="purple",
        label="± 1 Desvio Padrão",
    )
    axes[0].set_title(
        f"Evolução do EQM de Treino (média de {n_runs} realizações)",
        fontsize=12,
        fontweight="bold",
    )
    axes[0].set_xlabel("Época", fontsize=11)
    axes[0].set_ylabel("EQM", fontsize=11)
    axes[0].grid(True, linestyle="--", alpha=0.4)
    axes[0].legend(fontsize=9)

    # ── Direita: EQM médio de TESTE ──
    raw_mse_test = [r["mse_test_history"] for r in all_results]
    all_mse_test = np.array(
        [np.pad(arr, (0, max_epochs - len(arr)), mode="edge") for arr in raw_mse_test]
    )
    mean_test = np.mean(all_mse_test, axis=0)
    std_test = np.std(all_mse_test, axis=0)

    axes[1].plot(
        epochs,
        mean_test,
        color="#F44336",
        linewidth=2,
        marker="s",
        markersize=3,
        label=f"EQM Teste (média de {n_runs} realizações)",
    )
    axes[1].fill_between(
        epochs,
        mean_test - std_test,
        mean_test + std_test,
        alpha=0.2,
        color="#F44336",
        label="± 1 Desvio Padrão",
    )
    axes[1].set_title(
        f"Evolução do EQM de Teste (média de {n_runs} realizações)",
        fontsize=12,
        fontweight="bold",
    )
    axes[1].set_xlabel("Época", fontsize=11)
    axes[1].set_ylabel("EQM", fontsize=11)
    axes[1].grid(True, linestyle="--", alpha=0.4)
    axes[1].legend(fontsize=9)

    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"   Gráfico de EQM salvo em: {save_path}")
    plt.close(fig)


def plot_confusion_matrix_mean(
    all_results,
    species_names,
    save_path,
    custom_title_prefix="Matriz de Confusão Acumulada",
    subtitle_suffix=None,
):
    """
    Gráfico 3: Duas matrizes de confusão.
    Esquerda = Treino, Direita = Teste.
    """
    num_classes = len(species_names)
    n_runs = len(all_results)

    total_train = np.zeros((num_classes, num_classes), dtype=int)
    total_test = np.zeros((num_classes, num_classes), dtype=int)

    for r in all_results:
        total_train += r["conf_matrix_train"]
        total_test += r["conf_matrix_test"]

    acc_train = (
        np.trace(total_train) / np.sum(total_train) if np.sum(total_train) > 0 else 0
    )
    acc_test = (
        np.trace(total_test) / np.sum(total_test) if np.sum(total_test) > 0 else 0
    )

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    for ax, matrix, title, acc in [
        (axes[0], total_train, f"{custom_title_prefix} — TREINO", acc_train),
        (axes[1], total_test, f"{custom_title_prefix} — TESTE", acc_test),
    ]:
        im = ax.imshow(matrix, interpolation="nearest", cmap="Blues")
        fig.colorbar(im, ax=ax)
        ax.set_xticks(range(num_classes))
        ax.set_yticks(range(num_classes))
        ax.set_xticklabels(species_names, rotation=45, ha="right", fontsize=10)
        ax.set_yticklabels(species_names, fontsize=10)
        ax.set_xlabel("Predito", fontsize=11)
        ax.set_ylabel("Real", fontsize=11)
        subtitle = subtitle_suffix if subtitle_suffix else f"{n_runs} realizações"
        ax.set_title(
            f"{title}\n{subtitle} (Acc={acc:.2%})",
            fontsize=12,
            fontweight="bold",
        )

        for i in range(num_classes):
            for j in range(num_classes):
                val = matrix[i, j]
                color = "white" if val > matrix.max() * 0.6 else "black"
                ax.text(
                    j,
                    i,
                    str(val),
                    ha="center",
                    va="center",
                    fontsize=12,
                    color=color,
                    fontweight="bold",
                )

    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"   Matriz de confusão salva em: {save_path}")
    plt.close(fig)


def plot_decay_comparison(all_results, save_path):
    """
    Gráfico 4: Decaimento médio do raio de vizinhança (σ) e da taxa de
    aprendizado (α) ao longo das épocas, considerando todas as realizações.
    """
    n_runs = len(all_results)

    raw_sigma = [r["history_sigma"] for r in all_results]
    max_len_sigma = max(len(arr) for arr in raw_sigma)
    all_sigma = np.array(
        [np.pad(arr, (0, max_len_sigma - len(arr)), mode="edge") for arr in raw_sigma]
    )

    raw_lr = [r["history_lr"] for r in all_results]
    max_len_lr = max(len(arr) for arr in raw_lr)
    all_lr = np.array(
        [np.pad(arr, (0, max_len_lr - len(arr)), mode="edge") for arr in raw_lr]
    )

    mean_sigma = np.mean(all_sigma, axis=0)
    std_sigma = np.std(all_sigma, axis=0)
    mean_lr = np.mean(all_lr, axis=0)
    std_lr = np.std(all_lr, axis=0)

    epochs = list(range(1, len(mean_sigma) + 1))

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(
        f"Decaimento dos Parâmetros da SOM (média de {n_runs} realizações)",
        fontsize=13,
        fontweight="bold",
    )

    # ── Esquerda: Sigma ──
    axes[0].plot(
        epochs,
        mean_sigma,
        color="darkorange",
        linewidth=2,
        marker="o",
        markersize=3,
        label=f"σ médio ({n_runs} realizações)",
    )
    axes[0].fill_between(
        epochs,
        mean_sigma - std_sigma,
        mean_sigma + std_sigma,
        alpha=0.2,
        color="darkorange",
        label="± 1 Desvio Padrão",
    )
    axes[0].set_title("Decaimento do Raio de Vizinhança (σ)", fontweight="bold")
    axes[0].set_xlabel("Época", fontsize=11)
    axes[0].set_ylabel("σ (sigma)", fontsize=11)
    axes[0].grid(True, linestyle="--", alpha=0.4)
    axes[0].legend(fontsize=9)

    # ── Direita: Learning Rate ──
    axes[1].plot(
        epochs,
        mean_lr,
        color="purple",
        linewidth=2,
        marker="o",
        markersize=3,
        label=f"α médio ({n_runs} realizações)",
    )
    axes[1].fill_between(
        epochs,
        mean_lr - std_lr,
        mean_lr + std_lr,
        alpha=0.2,
        color="purple",
        label="± 1 Desvio Padrão",
    )
    axes[1].set_title("Decaimento da Taxa de Aprendizado (α)", fontweight="bold")
    axes[1].set_xlabel("Época", fontsize=11)
    axes[1].set_ylabel("α (learning rate)", fontsize=11)
    axes[1].grid(True, linestyle="--", alpha=0.4)
    axes[1].legend(fontsize=9)

    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"   Gráfico de decaimento salvo em: {save_path}")
    plt.close(fig)
