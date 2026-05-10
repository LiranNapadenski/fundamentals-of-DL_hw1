import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch.nn as nn

# Custom Dataset class
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

# Training and evaluation loop
def train_and_evaluate(model, epochs, train_loader, test_loader, optimizer, loss_fn, device):
    history = {
        'train_loss': [], 'train_acc': [],
        'test_loss': [], 'test_acc': []
    }

    epoch_bar = tqdm(range(epochs), desc="Training Progress", unit="epoch")

    model.to(device)

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

# Hyperparameter grid search
def grid_search(param_grid, train_loader, test_loader, device):
    best_acc = 0.0
    best_params = None
    best_history = None

    for lr in param_grid['lr']:
        for momentum in param_grid['momentum']:
            for epoch in param_grid['epochs']:
                print(f"Testing lr={lr}, momentum={momentum}, epochs={epoch}")

                model = SimpleCNN().to(device)
                optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)
                loss_fn = nn.CrossEntropyLoss()

                history = train_and_evaluate(
                    model, epochs=epoch, train_loader=train_loader, test_loader=test_loader,
                    optimizer=optimizer, loss_fn=loss_fn, device=device
                )

                # Average test accuracy at the end of training
                final_test_acc = sum(history['test_acc'][-5:]) / 5  # average of last 5 epochs

                if final_test_acc > best_acc:
                    best_acc = final_test_acc
                    best_params = (lr, momentum, epoch)
                    best_history = history

    print(f"Best Params: lr={best_params[0]}, momentum={best_params[1]}, epochs={best_params[2]} with Test Acc: {best_acc:.4f}")
    return best_params, best_acc, best_history


class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv2d(in_channels=64, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.fc1 = nn.Linear(16 * 8 * 8, 784)  # Assuming input images are 32x32
        self.fc2 = nn.Linear(784, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)

        x = x.view(x.size(0), -1)  # Flatten the tensor
        x = self.fc1(x)
        x = self.fc2(x)
        return x

