# MNIST GAN with mlX

This code implements a GAN (Generative Adversarial Network) for the MNIST dataset using the new Apple machine learning framework mlX, that is designed for Apple silicon.

The GAN consists of a generator network and a discriminator network. The generator network generates fake images, while the discriminator network tries to distinguish between real and fake images. The two networks are trained together in an adversarial manner.

This code is just to explores the capabilities of the mlX framework for machine learning tasks.

Run the code with:
```
python3 train_gan.py  --gpu
```

--------------------
Credits: This code is basically borrowed from this [Tinygrad example](https://github.com/tinygrad/tinygrad/blob/master/examples/mnist_gan.py).

About the mlX framework:
Checkout the [mlX repository](https://github.com/ml-explore/mlx) for more information or take a look at the documentation [mlX documentation](https://ml-explore.github.io/mlx).