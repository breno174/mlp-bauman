import pandas as pd
import numpy as np
from helper.map_data import load_and_prepare_data, train_test_split_hierarchical

# Imports para scikit-learn
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, accuracy_score

# Imports para PyTorch
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

def run_sklearn_mlp(X_train, X_test, y_train, y_test):
    print("\n--- Treinando MLP com scikit-learn ---")
    
    # Configurando uma MLP com 1 camada oculta de 100 neurônios
    # Você pode alterar 'hidden_layer_sizes' para modificar a arquitetura
    mlp = MLPClassifier(hidden_layer_sizes=(100,), max_iter=500, random_state=42)
    
    # Treinamento
    mlp.fit(X_train, y_train)
    
    # Predição
    y_pred = mlp.predict(X_test)
    
    # Avaliação
    print(f"Acurácia: {accuracy_score(y_test, y_pred):.4f}")
    print("\nRelatório de Classificação:")
    print(classification_report(y_test, y_pred))

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

def run_pytorch_mlp(X_train, X_test, y_train, y_test):
    print("\n--- Treinando MLP com PyTorch ---")
    
    # Convertendo os dados para tensores do PyTorch
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.long)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.long)
    
    # Criando DataLoaders
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    
    # Parâmetros da rede
    input_size = X_train.shape[1]
    hidden_size = 100 # Tamanho da camada oculta
    num_classes = len(np.unique(y_train))
    
    # Instanciando o modelo, função de perda e otimizador
    model = SimpleMLP(input_size, hidden_size, num_classes)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Treinamento
    num_epochs = 50
    for epoch in range(num_epochs):
        model.train()
        for batch_X, batch_y in train_loader:
            # Forward pass
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            
            # Backward e otimização
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
    # Avaliação
    model.eval()
    with torch.no_grad():
        outputs = model(X_test_tensor)
        _, predicted = torch.max(outputs.data, 1)
        
        total = y_test_tensor.size(0)
        correct = (predicted == y_test_tensor).sum().item()
        
        print(f"Acurácia: {100 * correct / total:.2f}%")


def main():
    # 1. Carrega e prepara os dados
    X, y_macro, y_raw, species_names, feature_names = load_and_prepare_data()
    
    # 2. Divisão de treino e teste
    print("\n[2] Dividindo os dados em Treino e Teste...")
    X_train, X_test, y_train, y_test, y_raw_train, y_raw_test = train_test_split_hierarchical(
        X, y_macro, y_raw, test_size=0.2, seed=42
    )
    
    print(f"   Treino: {X_train.shape[0]} amostras")
    print(f"   Teste: {X_test.shape[0]} amostras")
    
    # 3. Execução das MLPs
    run_sklearn_mlp(X_train, X_test, y_train, y_test)
    
    run_pytorch_mlp(X_train, X_test, y_train, y_test)


if __name__ == "__main__":
    main()
