# Pytorch-bcvae
Pytorch implement of bvae、fvae<br>
**version 2**<br>
# Enviornment
```
python  : 3.8.3
cuda    : 10.1
pytorch : 1.6.0(stable)
nvidia  : Tesla K80
GPU     : cuda 1
packages: torch, numpy, os, pickle, opencv
```
# Coefficients
```python
dataset      : dSprites'
batch_size   : 576
data_path    : './dsprites-dataset/dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz'
img_size     : (64, 64)
k-cross-val  : 10(k=10)

results_path : './results/'
models_path  : './models/'

#model 1
model_name   : bcvae2D
sch_gamma    : 0.95
lr           : 5e-3
epoch        : 80
gamma        : 10
beta         : 7
max_capacity      : 25
capacity_max_iter : 1e5
```
# Run
```bash
git clone https://github.com/scarletteshu/Pytorch-bcvae.git

cd Pytorch-bcvae

python run.py
```
# Model explain
<ul>
<li> 1 validation after 1 epoch training</li>
<li> save model dicts after 1 epoch training</li>
<li> lr decays for every 10 epoch</li>
<li> while training: <br>
1. record original imgs, reconstruction imgs and latents for every 100 iter<br>
2. record loss, reconstruction loss and kld loss for every 50 iter<br>
<li> while validating:<br>
1. record original imgs, reconstruction imgs and latents for every 200 iter<br>
2. record avg_loss, avg_recons_loss and avg_kld_loss once<br>
</ul>

# File explain
- model files
```python
dataload.py
betavae2d.py
run.py
```
- datasets
```
./dsprites-dataset/
```
- results
```
./results/
 |__./plot/
 |__./train/
    |__./latents/
    |__./loss/
    |__./origins/
    |__./recons/
 |__./val
    |__./latents/
    |__./loss/
    |__./origins/
    |__./recons/
```
>note 1 : plot file in python is not included for now, will be updated in the future<br>
>note 2 : for github storage concern, ./results only inlcudes partial results
# Papers
<a href="https://openreview.net/pdf?id=Sy2fzU9gl"> beta-VAE: Learning Basic Visual Concepts with a Constrained Variational Framework </a><br>
<a href="https://arxiv.org/pdf/1804.03599.pdf">Understanding disentangling in β-VAE</a>
# Reference
### dataset refered to:
<a href="https://deepmind.com/research/open-source/dsprites-disentanglement-testing-sprites-dataset">dsprites official </a><br>
or<br>
<a href="https://github.com/deepmind/dsprites-dataset">dsprites github ver</a>, which disclaims
```
The images were generated using the LOVE framework, which is licenced under zlib/libpng licence:

LOVE is Copyright (c) 2006-2016 LOVE Development Team

This software is provided 'as-is', without any express or implied
warranty. In no event will the authors be held liable for any damages
arising from the use of this software.

Permission is granted to anyone to use this software for any purpose,
including commercial applications, and to alter it and redistribute it
freely, subject to the following restrictions:

1. The origin of this software must not be misrepresented; you must not
claim that you wrote the original software. If you use this software
in a product, an acknowledgment in the product documentation would be
appreciated but is not required.

2. Altered source versions must be plainly marked as such, and must not be
misrepresented as being the original software.

3. This notice may not be removed or altered from any source
distribution.
```
### part of the codes refered to:
```
@misc{Subramanian2020,
  author = {Subramanian, A.K},
  title = {PyTorch-VAE},
  year = {2020},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/AntixK/PyTorch-VAE}}
}
```
# Cite
```
@misc{
  author = {scarlette},
  title = {PyTorch-bvae},
  year = {2020},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {https://github.com/scarletteshu/Pytorch-bcvae.git}
}
```
