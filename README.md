# wgan


Example experiment for the Wasserstein GAN on toy data.

Most of the code is a copy from https://github.com/martinarjovsky/WassersteinGAN, however I wrote it in myself and therefore any bugs or issues are my fault.

All hyperparameters are roughly the same as the original wgan, except for:

- weights are initialised to N(0, 0.2)

Some notes:

1. The critic output for real data doesn't really change
2. The critic loss (real - fake) has basically the 'right' shape, yet the generator distribution doesn't really match the input distribution at all.
