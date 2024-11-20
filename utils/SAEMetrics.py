import torch
import pickle
import numpy as np
from tqdm import tqdm

import matplotlib.pyplot as plt
import plotly.express as px
import plotly


def calc_freq(autoencoder, dataloader, device, no_batches=100):
    activation_counts = torch.zeros(autoencoder.m)
    total_samples = 0

    for idx, batch in enumerate(tqdm(dataloader)):
        X = batch.to(device)
        X_hat, f = autoencoder(X)

        activated_features_batch = (f.detach().cpu() > 0)
        activation_counts += torch.sum(activated_features_batch, dim=0)
        total_samples += dataloader.batch_size

        if no_batches != None and idx >= no_batches - 1:
            break

    return activation_counts / total_samples


def save_freqs_img_for_training(autoencoder, dataloader, device, no_batches=1000, xaxis="log", yaxis="log", filename="save.png"):
    activation_count = calc_freq(autoencoder, dataloader, device, no_batches=no_batches)
    nonzero_mask = (activation_count > 1e-6) & (activation_count < 1e-0)
    activation_count = activation_count[nonzero_mask]

    plt.hist(activation_count, bins=np.logspace(-5, 0, 50), density=True)
    plt.xscale(xaxis)
    plt.yscale(yaxis)
    plt.xlabel("Frequency")
    plt.ylabel("Occurences")
    plt.savefig(filename)
    plt.clf()


def calc_freq_from_tokens(autoencoder, batches, device, no_batches=1000):
    activation_sum = torch.zeros(autoencoder.m)
    activation_counts = torch.zeros(autoencoder.m)
    total_samples = 0

    #Feed Transformer Activations into AutoEncoder
    for idx, batch in tqdm(enumerate(batches)):
        X = batch.to(device)
        X_hat, f = autoencoder(X)

        activation_sum += torch.sum(f.detach().cpu(), dim=0)

        activated_features_batch = (f.detach().cpu() > 0)
        activation_counts += torch.sum(activated_features_batch, dim=0)
        total_samples += len(batch)

        if no_batches != None and idx >= no_batches - 1:
            break

    no_dead_neurons = torch.sum(activation_sum == 0)

    return activation_counts / total_samples, no_dead_neurons


def save_freqs_img_for_training_from_tokens(autoencoder, batches, device, no_batches=1000, xaxis="log", yaxis="linear", filename="./save.html"):
    activation_count, no_dead_neurons = calc_freq_from_tokens(autoencoder, batches, device, no_batches=no_batches)
    #nonzero_mask = (activation_count > 1e-8) & (activation_count < 1e-0)
    #activation_count = activation_count[nonzero_mask]

    fig = px.histogram(
        activation_count.log10(),
        histnorm="percent",
        title=f"{no_dead_neurons} dead Neurons"
    )
    plotly.offline.plot(fig, filename=filename)

    #plt.hist(activation_count, bins=np.logspace(-7, -2, 100), density=True)
    #plt.hist(activation_count, bins=np.logspace(0, 7, 100), density=True)

    # plt.xscale(xaxis)
    # plt.yscale(yaxis)
    # plt.xlabel("Frequency")
    # plt.ylabel("Occurences")
    # plt.savefig(filename)
    # plt.clf()


def plot_loss_curve(reconstruction_losses, sparsity_losses, filename="loss_curve.png"):
    plt.plot(reconstruction_losses, label="Reconstruction")
    plt.plot(sparsity_losses, label="Sparsity")
    plt.yscale("log")
    plt.legend()
    plt.savefig(filename)
    plt.clf()

