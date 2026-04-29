import numpy as np
from dataset import set_random, generete_data
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch
from collections import OrderedDict

#set random seed for reproducibility
set_random(42)

#generate data
X_train, y_train, X_test, y_test = generete_data()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

#define custom dataset class
class CIFAR10Dataset(Dataset):
    def __init__(self, data, labels, transform=None, target_transform=None):
        self.data = torch.tensor(data, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        x = self.data[idx]
        y = self.labels[idx]

        if self.transform:
            x = self.transform(x)
        if self.target_transform:
            y = self.target_transform(y)
        return x, y

# Create dataset and dataloader
train_dataset = CIFAR10Dataset(X_train, y_train)
test_dataset = CIFAR10Dataset(X_test, y_test)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

def set_parameters(model, std, lr, momentum):
    def init_weights_gaussian(m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=std)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)
    loss_fn = nn.CrossEntropyLoss()
    return init_weights_gaussian, optimizer, loss_fn

# set model
model = nn.Sequential(
    OrderedDict([
        ('fc1', nn.Linear(3072, 256)),
        ('relu1', nn.ReLU()),
        ('fc2', nn.Linear(256, 10))
    ])
)

# training loop
def train_and_evaluate(model, epochs, train_loader, test_loader, optimizer, loss_fn, init_weights, device):
    history = {
        'train_loss': [], 'train_acc': [],
        'test_loss': [], 'test_acc': []
    }
    model.to(device)
    model.apply(init_weights)

    epoch_bar = tqdm(range(epochs), desc="Training Progress", unit="epoch")

    for epoch in epoch_bar:
        # --- TRAINING PHASE ---
        model.train()
        running_loss, correct_train, total_train = 0.0, 0, 0

        for data, target in train_loader:
            data, target = data.to(device), target.to(device)

            optimizer.zero_grad()
            output = model(data)
            loss = loss_fn(output, target)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * data.size(0)
            _, predicted = torch.max(output.data, 1)
            total_train += target.size(0)
            correct_train += (predicted == target).sum().item()

        # --- EVALUATION PHASE ---
        model.eval()
        running_test_loss, correct_test, total_test = 0.0, 0, 0

        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                loss = loss_fn(output, target)

                running_test_loss += loss.item() * data.size(0)
                _, predicted = torch.max(output.data, 1)
                total_test += target.size(0)
                correct_test += (predicted == target).sum().item()

        # Metrics Calculation
        metrics = {
            'train_loss': running_loss / len(train_loader.dataset),
            'train_acc': correct_train / total_train,
            'test_loss': running_test_loss / len(test_loader.dataset),
            'test_acc': correct_test / total_test
        }

        for key in history:
            history[key].append(metrics[key])

        epoch_bar.set_postfix(train_acc=f"{metrics['train_acc']:.4f}", test_acc=f"{metrics['test_acc']:.4f}")

    return history

# hyperparameters grid search
def grid_search(model, param_grid, train_loader, test_loader, device):
    best_acc = 0.0
    best_params = None
    best_history = None

    for std in param_grid['std']:
        for lr in param_grid['lr']:
            for momentum in param_grid['momentum']:
                for epoch in param_grid['epochs']:
                    print(f"Testing std={std}, lr={lr}, momentum={momentum}, epochs={epoch}")

                    init_weights, optimizer, loss_fn = set_parameters(model, std=std, lr=lr, momentum=momentum)
                    history = train_and_evaluate(
                        model, epochs=epoch, train_loader=train_loader, test_loader=test_loader,
                        optimizer=optimizer, loss_fn=loss_fn, init_weights=init_weights, device=device
                    )

                    # avrage train accuracy at the end of training
                    final_train_acc = sum(history['train_acc'][-5:]) / 5  # average of last 5 epochs

                    if final_train_acc > best_acc:
                        best_acc = final_train_acc
                        best_params = (std, lr, momentum, epoch)
                        best_history = history

    print(f"Best Params: std={best_params[0]}, lr={best_params[1]}, momentum={best_params[2]}, epochs={best_params[3]} with Train Acc: {best_acc:.4f}")
    return best_params, best_acc, best_history



if __name__ == "__main__":
    param_grid = {
        'std': [0.01, 0.05],
        'lr': [0.001, 0.01],
        'momentum': [0.8, 0.9, 0.99],
        'epochs': [80]
    }
    best_params, best_acc, best_history = grid_search(model, param_grid, train_loader, test_loader, device)

    # Plotting results
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(best_history['train_loss'], label='Train Loss')
    plt.plot(best_history['test_loss'], label='Test Loss')
    plt.title('Loss Curves')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(best_history['train_acc'], label='Train Accuracy')
    plt.plot(best_history['test_acc'], label='Test Accuracy')
    plt.title('Accuracy Curves')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.show()