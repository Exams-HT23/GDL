import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from typing import List
from models import get_device, GCN

from torch.nn.functional import cross_entropy


def accuracy(
        y: torch.Tensor,
        y_hat: torch.Tensor,
        mask: torch.Tensor) -> float:

    # Get the class label with greatest probability for each node.
    pred = torch.argmax(y_hat, dim=1)

    # Compute the accuracy using the ground truth labels.
    return torch.mean((pred[mask] == y[mask]).float())


def train(
        model: nn.Module,
        X: torch.Tensor,
        y: torch.Tensor,
        edge_index: torch.Tensor,
        train_mask: torch.Tensor,
        val_mask: torch.Tensor,
        test_mask: torch.Tensor,
        learning_rate: float = 0.01,
        max_epochs: int = 500,
        patience: int = 20,
        verbose: bool = True) -> float:

    # Move the model to GPU if available.
    model = model.to(get_device())

    # Initialise all model parameters.
    for param in model.parameters():
        if param.requires_grad and param.dim() > 1:
            nn.init.xavier_uniform_(param)

    # Use the Adam optimiser when training.
    optimiser = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate
    )

    count = 0
    best_model_state = None
    best_val_acc = -float('inf')

    # Iterate for the given number of maximum epochs.
    train_accs, val_accs = [], []
    for epoch in range(max_epochs):
        optimiser.zero_grad()

        # Get the model's predictions.
        if isinstance(model, GCN):
            y_hat = model(X, edge_index)
        else:
            y_hat = model(X)

        # Compute the loss and backpropagate.
        loss = cross_entropy(y_hat[train_mask], y[train_mask])
        loss.backward()
        optimiser.step()

        # Compute the training and validation accuracies.
        train_acc = accuracy(y, y_hat, train_mask)
        val_acc = accuracy(y, y_hat, val_mask)

        train_accs.append(train_acc.cpu().detach().numpy())
        val_accs.append(val_acc.cpu().detach().numpy())

        # If the validation accuracy was not an improvement, increment the count.
        if val_acc < best_val_acc:
            count += 1
        else:
            # Otherwise, reset the count and save the model state.
            count = 0
            best_val_acc = val_acc
            best_model_state = model.state_dict()

        # If the validation accuracy has not improved after `patience` epochs, stop early.
        if epoch > 40 and count >= patience:
            if verbose:
                print(f'Early stopping after {epoch} epochs')
            break

        if verbose and epoch % 10 == 0:
            print(f"Epoch {str(epoch)+':':<4} train_loss={loss.item():.4f} | train_acc={train_acc:.4f} | val_acc={val_acc:.4f}")

    # Load the best model state for the final prediction.
    model.load_state_dict(best_model_state)
    if isinstance(model, GCN):
        y_hat = model(X, edge_index)
    else:
        y_hat = model(X)

    # Compute the test accuracy.
    test_acc = accuracy(y, y_hat, test_mask)

    if verbose:
        print(f"Test accuracy: {test_acc:.4f}")
        plot_train_val_accs(train_accs, val_accs)

    return test_acc.item()


def plot_train_val_accs(train_accs: List[float], val_accs: List[float]) -> None:
    # Plot the training and validation accuracies versus epoch.
    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111)
    ax.plot(train_accs, label='Train')
    ax.plot(val_accs, label='Validation')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Accuracy')
    ax.legend()
    plt.show()
