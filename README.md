<center><img src="https://raw.githubusercontent.com/gully/ynot/master/docs/_static/ynot_logo_0p1.svg" width="150"></center>

# `ynot`
#### _An interpretable machine learning approach to échellogram forward modeling_

[![Documentation Status](https://readthedocs.org/projects/ynot/badge/?version=latest)](https://ynot.readthedocs.io/en/latest/?badge=latest)

## Rationale: A spectrograph's digital twin

Virtually all modern insights derived from astronomical spectroscopy first arrive to the world on 2D digital camera chips such as visible [CCDs](https://en.wikipedia.org/wiki/Charge-coupled_device) or near-infrared [focal plane arrays](https://en.wikipedia.org/wiki/Staring_array). Data reduction pipelines translate these raw 2D images into 1D extracted spectra, the more nimble and natural data format for practitioners. These data pipelines work great, and for the vast majority of practitioners, pipelines are a mere afterthought.

But some science use cases---such as spatially extended objects, faint ultracool brown dwarfs, emission line objects, and Extreme Precision Radial Velocity (EPRV)---defy the assumptions built into these automated reduction pipelines. Heuristics may exist for some of these scenarios, with mixed performance, but in general, a turnkey solution is lacking for these and other non-standard use cases.

The $y_0$ framework aspires to address these problems through the creation of a _spectrograph [digital twin](https://en.wikipedia.org/wiki/Digital_twin)_. In this context, a digital twin indicates an ambition to forward model as many physically realistic attributes of the spectrograph system as possible. Above all, we seek a pixel-perfect scene model analogous to spectroperfectionism[^1], but with an improved ability to learn the physical state of the spectrograph by pooling information among observations. The physically interpretable and extensible underlying design should provide the stepping stone needed to simulate and ultimately fit EPRV data that may someday find Earth analogs.

[^1]: https://ui.adsabs.harvard.edu/abs/2010PASP..122..248B/abstract

## Architectural design and computational considerations

![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white)
![nVIDIA](https://img.shields.io/badge/nVIDIA-%2376B900.svg?style=for-the-badge&logo=nVIDIA&logoColor=white)

A typical modern astronomical echellogram has at least $1000 \times 1000$ pixels, and often mosaics or tiles of these can reach tens of millions of pixels in a single instrument. Scene modeling all of those pixels to arbitrary precision represents an enormous computational challenge. We therefore designed $y_0$ with these three uncompromising computational considerations in order to make the problem tractable:

1. Uses autodiff and Gradient Descent with PyTorch
2. Has hardware acceleration with NVIDIA GPUs
3. (Optionally) can use sparse matrices for target traces and sky background

## v0.1 demo on Keck NIRSPEC data

Our project is funded in part through NASA Astrophysics Data Analysis Program (ADAP) grant [80NSSC21K0650](https://www.highergov.com/grant/80NSSC21K0650/) to improve archival data from the [Keck NIRSPEC](https://www2.keck.hawaii.edu/inst/nirspec/) spectrograph. We are encouraged to see that version 0.1 of the code already delivers improved performance on an example brown dwarf spectrum in our first pass scene model. Below is a [tensorboard](https://www.tensorflow.org/tensorboard) screencap that shows how the PyTorch machine learning training proceeded, giving lower training loss and better 2D spectral reconstruction. This training process took a few minutes on my NVIDIA RTX2070 GPU. We think we can improve both the precision and speed of the `ynot` code in upcoming versions.

[![ynot with tensorboard](http://img.youtube.com/vi/mXToHEmq6hM/0.jpg)](http://www.youtube.com/watch?v=mXToHEmq6hM "ynot training demo")

## Project History, status, and how to get involved

The baseline PyTorch code and project scaffolding is based off of Ian Czekala's [MPoL project](https://github.com/iancze/MPoL). The project started in the early months of the COVID pandemic as a proof-of-concept demo, and was dormant as we built [blasé](https://github.com/gully/blase), which enables interpretable machine learning for 1D extracted spectra[^2].

[^2]: https://ui.adsabs.harvard.edu/abs/2022ApJ...941..200G/abstract

As of July 2023, we are returning to $y_0$ to make it a broadly applicable tool for a wide variety of scientific use cases. If that sounds good to you, join our open source project! We welcome contibutions from a wide range of experience levels.

To get involved, please introduce yourself on our [discussions](https://github.com/gully/ynot/discussions/22) page, or open an [Issue](https://github.com/gully/ynot/issues) describing your interest, or just [contact me](http://gully.github.io/) directly.

---

Copyright M. Gully-Santiago and contributors 2020, 2021, 2022, 2023

Version :
0.1
