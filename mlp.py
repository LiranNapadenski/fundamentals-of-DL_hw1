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

def set_parameters(model, lr, momentum, init_weights, adam_flag=False, weight_decay=0.0):
    model.apply(init_weights)
    if adam_flag:
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    else:
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
    loss_fn = nn.CrossEntropyLoss()
    return model, optimizer, loss_fn

# set model
model = nn.Sequential(
    OrderedDict([
        ('fc1', nn.Linear(3072, 256)),
        ('relu1', nn.ReLU()),
        ('fc2', nn.Linear(256, 10))
    ])
)

# training loop
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

                    model = nn.Sequential(
                        OrderedDict([
                            ('fc1', nn.Linear(3072, 256)),
                            ('relu1', nn.ReLU()),
                            ('fc2', nn.Linear(256, 10))
                        ])
                    ).to(device)
                    # We create an init_weights function to apply BEFORE the optimizer is set
                    def init_weights(m):
                        if isinstance(m, nn.Linear):
                            nn.init.normal_(m.weight, mean=0.0, std=std)
                            if m.bias is not None:
                                nn.init.zeros_(m.bias)
                    
                    _, optimizer, loss_fn = set_parameters(model, lr=lr, momentum=momentum, init_weights=init_weights)
                    history = train_and_evaluate(
                        model, epochs=epoch, train_loader=train_loader, test_loader=test_loader,
                        optimizer=optimizer, loss_fn=loss_fn, device=device
                    )

                    # avrage test accuracy at the end of training
                    final_test_acc = sum(history['test_acc'][-5:]) / 5  # average of last 5 epochs

                    if final_test_acc > best_acc:
                        best_acc = final_test_acc
                        best_params = (std, lr, momentum, epoch)
                        best_history = history

    print(f"Best Params: std={best_params[0]}, lr={best_params[1]}, momentum={best_params[2]}, epochs={best_params[3]} with Test Acc: {best_acc:.4f}")
    return best_params, best_acc, best_history



def grid_serach_exp():
    param_grid = {
        'std': [0.01, 0.05, 0.001],
        'lr': [0.001, 0.01, 0.05],
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

def adam_exp():
    model = nn.Sequential(
        OrderedDict([
            ('fc1', nn.Linear(3072, 256)),
            ('relu1', nn.ReLU()),
            ('fc2', nn.Linear(256, 10))
        ])
    ).to(device)

    def init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=0.05)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    _, optimizer, loss_fn = set_parameters(model, lr=0.001, momentum=0.9, init_weights=init_weights, adam_flag=True)
    history = train_and_evaluate(
        model, epochs=80, train_loader=train_loader, test_loader=test_loader,
        optimizer=optimizer, loss_fn=loss_fn, device=device
    )
    
    # Plotting results
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['test_loss'], label='Test Loss')
    plt.title('Loss Curves (Adam)')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history['train_acc'], label='Train Accuracy')
    plt.plot(history['test_acc'], label='Test Accuracy')
    plt.title('Accuracy Curves (Adam)')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.show()

def xaivier_init_exp():
    model = nn.Sequential(
        OrderedDict([
            ('fc1', nn.Linear(3072, 256)),
            ('relu1', nn.ReLU()),
            ('fc2', nn.Linear(256, 10))
        ])
    ).to(device)

    def init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    _, optimizer, loss_fn = set_parameters(model, lr=0.01, momentum=0.9, init_weights=init_weights)
    history = train_and_evaluate(
        model, epochs=80, train_loader=train_loader, test_loader=test_loader,
        optimizer=optimizer, loss_fn=loss_fn, device=device
    )
    
    # Plotting results
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['test_loss'], label='Test Loss')
    plt.title('Loss Curves (Xavier)')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history['train_acc'], label='Train Accuracy')
    plt.plot(history['test_acc'], label='Test Accuracy')
    plt.title('Accuracy Curves (Xavier)')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.show()

def regularization_exp():
    # Experiment with weight decay and dropout regularization
    weight_decay = 1e-3
    dropout_prob = 0.3
    model = nn.Sequential(
        OrderedDict([
            ('fc1', nn.Linear(3072, 256)),
            ('relu1', nn.ReLU()),
            ('dropout1', nn.Dropout(p=dropout_prob)),  # Dropout layer with dropout_prob probability
            ('fc2', nn.Linear(256, 10))
        ])
    ).to(device)

    def init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=0.05)
            if m.bias is not None:
                nn.init.zeros_(m.bias)


    _, optimizer, loss_fn = set_parameters(model, lr=0.01, momentum=0.9, init_weights=init_weights, weight_decay=weight_decay)
    history = train_and_evaluate(
        model, epochs=80, train_loader=train_loader, test_loader=test_loader,
        optimizer=optimizer, loss_fn=loss_fn, device=device
    )

    # Plotting results
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['test_loss'], label='Test Loss')
    plt.title('Loss Curves (Regularization)')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history['train_acc'], label='Train Accuracy')
    plt.plot(history['test_acc'], label='Test Accuracy')
    plt.title('Accuracy Curves (Regularization)')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.show()


def pca_preprocessing_exp():
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    # Standardize the data before PCA
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Apply PCA to reduce dimensionality to 256
    pca = PCA(n_components=256)
    X_train_pca = pca.fit_transform(X_train_scaled)
    X_test_pca = pca.transform(X_test_scaled)

    # Create new datasets and dataloaders with PCA-transformed data
    train_dataset_pca = CIFAR10Dataset(X_train_pca, y_train)
    test_dataset_pca = CIFAR10Dataset(X_test_pca, y_test)
    train_loader_pca = DataLoader(train_dataset_pca, batch_size=64, shuffle=True)
    test_loader_pca = DataLoader(test_dataset_pca, batch_size=64, shuffle=False)

    # Define a new model that takes 256 input features instead of 3072
    model_pca = nn.Sequential(
        OrderedDict([
            ('fc1', nn.Linear(256, 256)),
            ('relu1', nn.ReLU()),
            ('fc2', nn.Linear(256, 10))
        ])
    ).to(device)

    def init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=0.05)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    _, optimizer, loss_fn = set_parameters(model_pca, lr=0.01, momentum=0.9, init_weights=init_weights)
    history = train_and_evaluate(
        model_pca, epochs=80, train_loader=train_loader_pca, test_loader=test_loader_pca,
        optimizer=optimizer, loss_fn=loss_fn, device=device
    )

    # Plotting results
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['test_loss'], label='Test Loss')
    plt.title('Loss Curves (PCA Preprocessing)')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history['train_acc'], label='Train Accuracy')
    plt.plot(history['test_acc'], label='Test Accuracy')
    plt.title('Accuracy Curves (PCA Preprocessing)')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.tight_layout()
    plt.show()


def width_exp():
    # Experiment with wider MLP architecture

    results = {'width': [], 'train_acc': [], 'test_acc': [], 'train_loss': [], 'test_loss': []}

    for i in [6, 10, 12]:
        width = 2 ** i
        model_wide = nn.Sequential(
            OrderedDict([
                ('fc1', nn.Linear(3072, width)),
                ('relu1', nn.ReLU()),
                ('fc2', nn.Linear(width, 10))
            ])
        ).to(device)

        def init_weights(m):
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=0.05)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        _, optimizer, loss_fn = set_parameters(model_wide, lr=0.01, momentum=0.9, init_weights=init_weights)
        history = train_and_evaluate(
            model_wide, epochs=80, train_loader=train_loader, test_loader=test_loader,
            optimizer=optimizer, loss_fn=loss_fn, device=device
        )

        # Store results for each width (average of last 5 epochs)
        results['width'].append(width)
        results['train_acc'].append(sum(history['train_acc'][-5:]) / 5)
        results['test_acc'].append(sum(history['test_acc'][-5:]) / 5)
        results['train_loss'].append(sum(history['train_loss'][-5:]) / 5)
        results['test_loss'].append(sum(history['test_loss'][-5:]) / 5)

    # Plot results
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(results['width'], results['train_loss'], label='Train Loss', marker='o')
    plt.plot(results['width'], results['test_loss'], label='Test Loss', marker='o')
    plt.title('Loss vs Width')
    plt.xlabel('Width')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(results['width'], results['train_acc'], label='Train Accuracy', marker='o')
    plt.plot(results['width'], results['test_acc'], label='Test Accuracy', marker='o')
    plt.title('Accuracy vs Width')
    plt.xlabel('Width')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.show()



def depth_exp():
    # Experiment with deeper MLP architecture 

    results = {'depth': [], 'train_acc': [], 'test_acc': [], 'train_loss': [], 'test_loss': []}

    for depth in [3, 4, 10]:
        layers = []
        input_size = 3072
        for i in range(depth):
            layers.append((f'fc{i+1}', nn.Linear(input_size, 64)))
            layers.append((f'relu{i+1}', nn.ReLU()))
            input_size = 64
        layers.append(('fc_out', nn.Linear(64, 10)))

        model_deep = nn.Sequential(OrderedDict(layers)).to(device)

        def init_weights(m):
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=0.05)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        _, optimizer, loss_fn = set_parameters(model_deep, lr=0.01, momentum=0.9, init_weights=init_weights)
        history = train_and_evaluate(
            model_deep, epochs=80, train_loader=train_loader, test_loader=test_loader,
            optimizer=optimizer, loss_fn=loss_fn, device=device
        )

        # Store results for each depth (average of last 5 epochs)
        results['depth'].append(depth)
        results['train_acc'].append(sum(history['train_acc'][-5:]) / 5)
        results['test_acc'].append(sum(history['test_acc'][-5:]) / 5)
        results['train_loss'].append(sum(history['train_loss'][-5:]) / 5)
        results['test_loss'].append(sum(history['test_loss'][-5:]) / 5)

    # Plot results
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(results['depth'], results['train_loss'], label='Train Loss', marker='o')
    plt.plot(results['depth'], results['test_loss'], label='Test Loss', marker='o')
    plt.title('Loss vs Depth')
    plt.xlabel('Depth')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(results['depth'], results['train_acc'], label='Train Accuracy', marker='o')
    plt.plot(results['depth'], results['test_acc'], label='Test Accuracy', marker='o')
    plt.title('Accuracy vs Depth')
    plt.xlabel('Depth')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    depth_exp()