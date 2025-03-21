# Second Order Volterra Variational Autoencoder (V2VAE)
This repository contains the implementation of the Volterra Variational Autoencoder (V2VAE), i.e. a variational autoencoder where the decoder is a Volterra filter.
The work is based on the VAE for equalization of communication systems [1, 2].

Check out our paper which is currently under review:
https://arxiv.org/abs/2410.16125

If you find it helpful/intestering, please consider throwing a reference [3].

## Getting started

Install the requirements using pip
```
pip install -r requirements.txt
```

This should enable you to run the main, that fits a V2VAE to a non-linear ISI channel:
```
python main_v2vae.py
```

NB! The V2VAE can see some significant speedups if using a GPU. If possible, prioritize installing pytorch with a good GPU setup.

## Acknowledgements

The work carried out in this repository is part of a research project [MAchine leaRning enaBLEd fiber optic communication](https://veluxfoundations.dk/en/villum-synergy-2021) (MARBLE) funded by the Villum foundation.

## References

[1] A. Caciularu and D. Burshtein, “Unsupervised Linear and Nonlinear Channel Equalization and Decoding Using Variational Autoencoders,” IEEE Transactions on Cognitive Communications and Networking, vol. 6, no. 3, pp. 1003–1018, Sep. 2020, doi: 10.1109/TCCN.2020.2990773.

[2] V. Lauinger, F. Buchali, and L. Schmalen, “Blind Equalization and Channel Estimation in Coherent Optical Communications Using Variational Autoencoders,” IEEE Journal on Selected Areas in Communications, vol. 40, no. 9, pp. 2529–2539, Sep. 2022, doi: 10.1109/JSAC.2022.3191346.

[3] S. F. Nielsen, D. Zibar, and M. N. Schmidt, “Blind Equalization using a Variational Autoencoder with Second Order Volterra Channel Model,” Oct. 21, 2024, arXiv: arXiv:2410.16125. doi: 10.48550/arXiv.2410.16125.

[4] J. Song et al., “Blind Channel Equalization Using Vector-Quantized Variational Autoencoders,” Feb. 22, 2023, arXiv: arXiv:2302.11687. doi: 10.48550/arXiv.2302.11687.

Code for [2]: https://github.com/kit-cel/vae-equalizer
