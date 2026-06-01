import pandas as pd
import numpy as np
import os
import sys
import json
import joblib
from datetime import datetime

# Adiciona o diretório 'src' ao sys.path para permitir a importação de helpers
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from helper.map_data import load_and_prepare_data, train_test_split_hierarchical
from helper.plot_data import plot_mlp_loss, plot_mlp_confusion_matrix

# Usamos o scikit-learn apenas para a matriz de confusão
from sklearn.metrics import confusion_matrix

# Imports para PyTorch
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# =====================================================================
# HIPERPARÂMETROS
# =====================================================================
# Gerais
RANDOM_SEED = 42
TEST_SIZE = 0.2


# Hiperparâmetros da MLP PyTorch
PT_HIDDEN_SIZE = 100  # Quantidade de neurônios na camada oculta
PT_NUM_EPOCHS = 100  # Número de épocas
PT_BATCH_SIZE = 26  # Tamanho do lote (batch)
PT_LEARNING_RATE = 0.001  # Taxa de aprendizado
# =====================================================================


# Definição do modelo PyTorch com 1 camada oculta
class SimpleMLP(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(SimpleMLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out


def run_pytorch_mlp(
    X_train,
    X_test,
    y_train,
    y_test,
    species_names,
    exp_dir,
    hidden_size,
    num_epochs,
    batch_size,
    learning_rate,
):
    print("\n--- Treinando MLP com PyTorch ---")

    # Convertendo os dados para tensores do PyTorch
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.long)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.long)

    # Criando DataLoaders
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # Parâmetros da rede
    input_size = X_train.shape[1]
    num_classes = len(species_names)

    # Instanciando o modelo, função de perda e otimizador
    model = SimpleMLP(input_size, hidden_size, num_classes)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Treinamento com coleta da perda (Loss) por época
    loss_history_train = []
    loss_history_test = []

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        for batch_X, batch_y in train_loader:
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(train_loader)
        loss_history_train.append(avg_loss)

        # Calcula test loss
        model.eval()
        with torch.no_grad():
            outputs_test = model(X_test_tensor)
            test_loss = criterion(outputs_test, y_test_tensor).item()
            loss_history_test.append(test_loss)

    # Avaliação e Matrizes de Confusão
    model.eval()
    with torch.no_grad():
        # Treino
        outputs_train = model(X_train_tensor)
        _, predicted_train = torch.max(outputs_train.data, 1)
        cm_train = confusion_matrix(y_train_tensor.numpy(), predicted_train.numpy())

        # Teste
        outputs_test = model(X_test_tensor)
        _, predicted_test = torch.max(outputs_test.data, 1)
        y_test_pred = predicted_test.numpy()
        y_test_true = y_test_tensor.numpy()
        cm_test = confusion_matrix(y_test_true, y_test_pred)

        total = y_test_true.shape[0]
        correct = (y_test_pred == y_test_true).sum()
        acc = correct / total
        print(f"Acurácia (Teste): {100 * acc:.2f}%")

    # --- SALVAMENTO DOS DADOS DO TREINO ---

    # 1. Salvar os pesos do modelo
    model_path = os.path.join(exp_dir, "pytorch_mlp_model.pth")
    torch.save(model.state_dict(), model_path)

    # 2. Salvar os parâmetros e resultados em JSON
    meta = {
        "framework": "PyTorch",
        "input_size": input_size,
        "hidden_size": hidden_size,
        "num_classes": num_classes,
        "num_epochs": num_epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "accuracy": acc,
    }
    with open(os.path.join(exp_dir, "pytorch_mlp_meta.json"), "w") as f:
        json.dump(meta, f, indent=4)

    # 3. Plotagem
    plot_mlp_loss(
        train_loss=loss_history_train,
        test_loss=loss_history_test,
        title="Curva de Perda (Loss) - PyTorch",
        save_path=os.path.join(exp_dir, "pytorch_mlp_loss.png"),
    )

    plot_mlp_confusion_matrix(
        cm_train=cm_train,
        cm_test=cm_test,
        species_names=species_names,
        title="Matrizes de Confusão - PyTorch",
        save_path=os.path.join(exp_dir, "pytorch_mlp_cm.png"),
    )


def main():
    # Cria uma pasta para o experimento baseada no Timestamp (como no seu SOM)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_dir = os.path.join("src", "experiments", "mlp", timestamp)
    os.makedirs(exp_dir, exist_ok=True)

    print(f"==================================================")
    print(f" Iniciando experimento: {timestamp}")
    print(f" Diretório de salvamento: {exp_dir}")
    print(f"==================================================")

    # 1. Carrega e prepara os dados
    X, y_macro, y_raw, species_names, feature_names = load_and_prepare_data()

    # 2. Divisão de treino e teste
    print("\n[2] Dividindo os dados em Treino e Teste...")
    X_train, X_test, y_train, y_test, y_raw_train, y_raw_test = (
        train_test_split_hierarchical(
            X, y_macro, y_raw, test_size=TEST_SIZE, seed=RANDOM_SEED
        )
    )

    print(f"   Treino: {X_train.shape[0]} amostras")
    print(f"   Teste: {X_test.shape[0]} amostras")

    # Salvar metadados gerais do dataset
    dataset_meta = {
        "timestamp": timestamp,
        "train_size": X_train.shape[0],
        "test_size": X_test.shape[0],
        "features": feature_names,
        "classes": species_names,
        "split_seed": RANDOM_SEED,
    }
    with open(os.path.join(exp_dir, "dataset_meta.json"), "w") as f:
        json.dump(dataset_meta, f, indent=4)

    # 3. Execução da MLP com os hiperparâmetros globais
    run_pytorch_mlp(
        X_train,
        X_test,
        y_train,
        y_test,
        species_names,
        exp_dir,
        hidden_size=PT_HIDDEN_SIZE,
        num_epochs=PT_NUM_EPOCHS,
        batch_size=PT_BATCH_SIZE,
        learning_rate=PT_LEARNING_RATE,
    )


if __name__ == "__main__":
    main()
