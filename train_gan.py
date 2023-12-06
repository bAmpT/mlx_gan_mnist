import argparse
import time
from typing import Any

import numpy as np
import torch
from torchvision.utils import make_grid, save_image

import mlx.core as mx
import mlx.nn as nn
from mlx.optimizers import Adam
from optimizer import AdamW 

import mnist

def leakyrelu(x, neg_slope=0.01): 
    return nn.relu(x) - nn.relu(-neg_slope * x)

def _softmax(x, axis):
    m = x - mx.max(x, axis=axis, keepdims=True)
    e = m.exp()
    return m, e, e.sum(axis=axis, keepdims=True)

def softmax(x, axis=-1):
    _, e, ss = _softmax(x, axis)
    return e.div(ss)

def log_softmax(x, axis=-1):
    m, _, ss = _softmax(x, axis)
    return m - ss.log()

class Generator(nn.Module):
    def __init__(self, g_input_dim, g_output_dim, hidden_dim = 256):
        super().__init__()
        self.fc1 = nn.Linear(g_input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 2*hidden_dim)
        self.fc3 = nn.Linear(2*hidden_dim, 4*hidden_dim)
        self.fc4 = nn.Linear(4*hidden_dim, g_output_dim)

    def __call__(self, x) -> Any:
        x = leakyrelu(self.fc1(x), 0.2)
        x = leakyrelu(self.fc2(x), 0.2)
        x = leakyrelu(self.fc3(x), 0.2)
        x = mx.tanh(self.fc4(x))
        return x

class Discriminator(nn.Module):
    def __init__(self, d_input_dim, hidden_dim = 256):
        super().__init__()
        self.d1 = nn.Dropout(0.3)
        self.d2 = nn.Dropout(0.3)
        self.d3 = nn.Dropout(0.3)
        self.fc1 = nn.Linear(d_input_dim, 4*hidden_dim)
        self.fc2 = nn.Linear(4*hidden_dim, 2*hidden_dim)
        self.fc3 = nn.Linear(2*hidden_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, 2)
    
    def __call__(self, x) -> Any:
        x = self.d1(leakyrelu(self.fc1(x+1.0), 0.2))
        x = self.d2(leakyrelu(self.fc2(x), 0.2))
        x = self.d3(leakyrelu(self.fc3(x), 0.2))
        x = log_softmax(self.fc4(x))
        return x

def make_labels(bs, col, val=-2.0):
  y = np.zeros((bs, 2), np.float32)
  y[range(bs), [col] * bs] = val  # Can we do label smoothing? i.e -2.0 changed to -1.98789.
  return mx.array(y)

def loss_fn_d(model_g, model_d, X, z_dim):
    batch_size = X.shape[0]
    labels_real = make_labels(batch_size, 1)
    loss_real = mx.mean(model_d(X) * labels_real)
    
    z = mx.random.normal(shape=(batch_size, z_dim))
    labels_fake = make_labels(batch_size, 0)
    loss_fake = mx.mean(model_d(mx.stop_gradient(model_g(z))) * labels_fake)

    return loss_real + loss_fake

def loss_fn_g(model_g, model_d, batch_size, z_dim):
    z = mx.random.normal(shape=(batch_size, z_dim))
    labels_real = make_labels(batch_size, 1)
    return mx.mean(model_d(model_g(z)) * labels_real)

def batch_iterate(batch_size, X, y):
    perm = mx.array(np.random.permutation(y.size))
    for s in range(0, y.size, batch_size):
        ids = perm[s : s + batch_size]
        yield X[ids], y[ids]


def main(args):
    seed = 1
    hidden_dim = 256
    z_dim = 128
    batch_size = 128
    num_epochs = 50
    learning_rate_d = 2e-4
    learning_rate_g = 2e-4
    sample_interval = 5

    np.random.seed(seed)
    ds_noise = mx.random.normal(shape=(64, 128))

    # Load the data
    train_images, train_labels, test_images, test_labels = map(mx.array, mnist.mnist())
    train_images = train_images * 2.0 - 1.0

    # Load the model
    model_g = Generator(z_dim, train_images.shape[-1], hidden_dim=hidden_dim)
    model_d = Discriminator(train_images.shape[-1], hidden_dim=hidden_dim)    
    mx.eval(model_g.parameters())
    mx.eval(model_d.parameters())

    loss_and_grad_fn_g = nn.value_and_grad(model_g, loss_fn_g)
    loss_and_grad_fn_d = nn.value_and_grad(model_d, loss_fn_d)
    optimizer_g = Adam(learning_rate=learning_rate_g, betas=[0.5, 0.999]) 
    optimizer_d = Adam(learning_rate=learning_rate_d, betas=[0.5, 0.999]) 
    if args.adamw:
        optimizer_g = AdamW(learning_rate=learning_rate_g, betas=[0.5, 0.999],weight_decay=0.01)
        optimizer_d = AdamW(learning_rate=learning_rate_d, betas=[0.5, 0.999], weight_decay=0.01)

    for e in range(num_epochs):
        tic = time.perf_counter()
        for X, y in batch_iterate(batch_size, train_images, train_labels):
            
            # train discriminator
            loss_d, grads_d = loss_and_grad_fn_d(model_g, model_d, X, z_dim)
            optimizer_d.update(model_d, grads_d)
            mx.eval(model_d.parameters(), optimizer_d.state)

            # train generator
            loss_g, grads_g = loss_and_grad_fn_g(model_g, model_d, batch_size, z_dim)
            optimizer_g.update(model_g, grads_g)
            mx.eval(model_g.parameters(), optimizer_g.state)

        toc = time.perf_counter()
        print(
            f"Epoch {e}, Losses D={loss_d.item():.3f} G={loss_g.item():.3f},",
            f"Time {toc - tic:.3f} (s)"
        )

        # plot some samples        
        if (e+1) % sample_interval == 0:
            fake_images = np.array(model_g(ds_noise))
            fake_images = (fake_images.reshape(-1, 1, 28, 28) + 1) / 2  # 0 - 1 range.
            save_image(make_grid(torch.tensor(fake_images)),  f"./images/image_{e+1}.jpg")


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Train a simple GAN network on MNIST with MLX.")
    parser.add_argument("--gpu", action="store_true", help="Use the Metal back-end.")
    parser.add_argument("--adamw", action="store_true", help="Use the AdamW optimizer.")
    args = parser.parse_args()
    if not args.gpu:
        mx.set_default_device(mx.cpu)
    main(args)