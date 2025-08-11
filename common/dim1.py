import random
import math

import torch
import torch.optim
import torch.nn as nn
import torch.nn.functional as functional

import matplotlib.pyplot as plt
import matplotlib.animation as animation

from typing import List, Callable, Optional
from torch.utils.data import TensorDataset, DataLoader
from IPython.display import Image, display
from dataclasses import dataclass, field

@dataclass
class FnnConfig:
    learning_rate: float
    epochs: int
    batch_size: int
    x: torch.tensor
    y: torch.tensor
    animation_filename: str
    layer_dimensions: List[List[int]] = field(default_factory=list)
    seed: int = 10
    activation_function: Callable = functional.relu
    optimizer_function: Callable = torch.optim.Adam
    loss_function: Callable = nn.MSELoss()
    device: str = "cpu"


class Fnn(nn.Module):
    def __init__(self, fnn_config):
        super(Fnn, self).__init__()

        self.fnn_config = fnn_config
        self.layers = nn.ModuleList()

        for i in range(0, len(fnn_config.layer_dimensions)):
            in_dim = fnn_config.layer_dimensions[i][0]
            out_dim = fnn_config.layer_dimensions[i][1]
            self.layers.append(nn.Linear(in_dim, out_dim))

    def forward(self, x):
        for i in range(0, len(self.layers)):
            layer = self.layers[i]

            if i == len(self.layers) - 1:
                # Last layer does not use an activation function
                x = layer(x)
            else:
                x = self.fnn_config.activation_function(layer(x))

        return x


@dataclass
class TrainResponse:
    training_losses: List[float] = field(default_factory=list)
    validation_losses: List[float] = field(default_factory=list)
    x: List[float] = field(default_factory=list)
    y_actual: List[float] = field(default_factory=list)
    y_predicted: List[List[float]] = field(default_factory=list)
    model: Optional[Fnn] = None


def train_fnn(fc: FnnConfig):
    random_object = random.Random(fc.seed)

    model = Fnn(fc)
    model.to(fc.device)

    assert len(fc.x.shape) == 1
    assert len(fc.y.shape) == 1
    assert fc.x.shape[0] == fc.y.shape[0]

    x_indexes = list(range(0, fc.x.shape[0]))
    random_object.shuffle(x_indexes)

    split_index = math.floor(0.8 * len(fc.x))

    x_train = fc.x[x_indexes[0:split_index]].unsqueeze(1)
    y_train = fc.y[x_indexes[0:split_index]].unsqueeze(1)

    x_validation = fc.x[x_indexes[split_index:]].unsqueeze(1)
    y_validation = fc.y[x_indexes[split_index:]].unsqueeze(1)

    x_all = fc.x.unsqueeze(1)
    
    x_train = x_train.to(fc.device)
    y_train = y_train.to(fc.device)
    
    dataset_train = TensorDataset(x_train, y_train)
    train_loader = DataLoader(dataset_train, batch_size=fc.batch_size)

    optimizer = fc.optimizer_function(model.parameters(), lr=fc.learning_rate)

    response = TrainResponse(
        model=model,
        x=fc.x.tolist(),
        y_actual=fc.y.tolist(),
    )

    for epoch in range(1, fc.epochs + 1):
        model.train()

        for x_batch, y_batch in train_loader:
            x_batch = x_batch.to(fc.device)
            y_batch = y_batch.to(fc.device)
            loss_train = fc.loss_function(model(x_batch), y_batch)
            optimizer.zero_grad()
            loss_train.backward()
            optimizer.step()

        model.eval()

        with torch.no_grad():
            loss_train = fc.loss_function(model(x_train), y_train).detach().cpu().item()
            loss_train /= x_train.shape[0]
            response.training_losses.append(loss_train)

            loss_validation = fc.loss_function(
                model(x_validation), y_validation).detach().cpu().item()
            loss_validation /= x_validation.shape[0]
            response.validation_losses.append(loss_validation)

            y_epoch_predicted = model(x_all)
            response.y_predicted.append(y_epoch_predicted.tolist())

        if epoch != 0:
            print("\r", end="")

        print(f"Epoch {epoch:5d} | loss_train={loss_train:.5f} | loss_validation={loss_validation:.5f}", end="")

    print("")
    return response


def plot_train_results(tr: TrainResponse, fc: FnnConfig):
    epoch_list = list(range(0, fc.epochs))

    fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(12, 12))

    axs[0][0].set_title("Actual vs Predicted Animation")
    axs[0][0].set_xlabel("x")
    axs[0][0].set_ylabel("y")
    actual_plot, = axs[0][0].plot(tr.x, tr.y_actual, 'b-', label="Actual")
    predicted_plot, = axs[0][0].plot([], [], 'r-', label="Predicition")
    axs[0][0].legend()
    axs[0][1].margins(0.1)

    axs[0][1].set_title("Learning Curve Animation")
    axs[0][1].set_xlabel("Epoch")
    axs[0][1].set_ylabel("Loss")
    axs[0][1].set_xlim([0, fc.epochs])
    axs[0][1].set_ylim(
        [0, max([max(tr.training_losses), max(tr.validation_losses)])])
    training_loss_plot, = axs[0][1].plot([], [], 'b-', label="Training")
    validation_loss_plot, = axs[0][1].plot([], [], 'r-', label="Validation")
    axs[0][1].legend()
    axs[0][1].margins(0.1)

    axs[1][0].set_title("Actual vs Predicted Final")
    axs[1][0].set_xlabel("x")
    axs[1][0].set_ylabel("y")
    axs[1][0].plot(tr.x, tr.y_actual, 'b-', label="Actual")
    axs[1][0].plot(tr.x, tr.y_predicted[-1], 'r-', label="Predicition")
    axs[1][0].legend()

    axs[1][1].set_title("Learning Curve Final")
    axs[1][1].set_xlabel("Epoch")
    axs[1][1].set_ylabel("Loss")
    axs[1][1].plot(epoch_list, tr.training_losses, 'b-', label="Training")
    axs[1][1].plot(epoch_list, tr.validation_losses, 'r-', label="Validation")
    axs[1][1].legend()

    def update_actual_predicted(index):
        print("\r", end="")
        print(f"Animating frame {index + 1}/{fc.epochs}", end="")
        
        predicted_plot.set_data(tr.x, tr.y_predicted[index])
        training_loss_plot.set_data(epoch_list[0:index], tr.training_losses[0:index])
        validation_loss_plot.set_data(epoch_list[0:index], tr.validation_losses[0:index])

        return (actual_plot, predicted_plot, training_loss_plot, validation_loss_plot)

    anim = animation.FuncAnimation(fig=fig, func=update_actual_predicted, frames=fc.epochs, interval=20)
    anim.save(fc.animation_filename, writer='pillow', fps=10)
    plt.close(fig)
    print("")