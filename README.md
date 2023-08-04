# $y_0$

#### An interpretable machine learning approach to échellogram forward modeling.

![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white)
[![Documentation Status](https://readthedocs.org/projects/ynot/badge/?version=latest)](https://ynot.readthedocs.io/en/latest/?badge=latest)

## Rationale: A spectrograph's digital twin

Virtually all modern insights derived from astronomical spectroscopy first arrive to the world on 2D digital camera chips such as visible [CCDs](https://en.wikipedia.org/wiki/Charge-coupled_device) or near-infrared [focal plane arrays](https://en.wikipedia.org/wiki/Staring_array). Data reduction pipelines translate these raw 2D images into 1D extracted spectra, the more nimble and natural data format for practitioners. These data pipelines work great, and for the vast majority of practitioners, pipelines are a mere afterthought.

But some science use cases---such as spatially extended objects, faint ultracool brown dwarfs, emission line objects, and Extreme Precision Radial Velocity (EPRV)---defy the assumptions built into these automated reduction pipelines. Heuristics may exist for some of these scenarios, with mixed performance, but in general, a turnkey solution is lacking for these and other non-standard use cases.

The $y_0$ framework aspires to address these problems through the creation of a _spectrograph [digital twin](https://en.wikipedia.org/wiki/Digital_twin)_. In this context, a digital twin indicates an ambition to forward model as many physically realistic attributes of the spectrograph system as possible. Above all, we seek a pixel-perfect scene model analogous to spectroperfectionism[^1], but with an improved ability to learn the physical state of the spectrograph by pooling information among observations. The physically interpretable and extensible underlying design should provide the stepping stone needed to simulate and ultimately fit EPRV data that may someday find Earth analogs.

[^1]: https://ui.adsabs.harvard.edu/abs/2010PASP..122..248B/abstract

## Project History, status, and how to get involved

The baseline PyTorch code and project scaffolding is based off of Ian Czekala's [MPoL project](https://github.com/iancze/MPoL). The project started in the early months of the COVID pandemic as a proof-of-concept demo, and was dormant as we built [blasé](https://github.com/gully/blase), which enables interpretable machine learning for 1D extracted spectra[^2].

[^2]: https://ui.adsabs.harvard.edu/abs/2022ApJ...941..200G/abstract

As of July 2023, we are returning to $y_0$ to make it a broadly applicable tool for a wide variety of scientific use cases. If that sounds good to you, join our open source project! We welcome contibutions from a wide range of experience levels.

To get involved, please introduce yourself on our [discussions](https://github.com/gully/ynot/discussions/22) page, or open an [Issue](https://github.com/gully/ynot/issues) describing your interest, or just [contact me](http://gully.github.io/) directly.

<iframe width="560" height="315" src="https://www.youtube.com/embed/mXToHEmq6hM" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" allowfullscreen></iframe>

---

Copyright M. Gully-Santiago and contributors 2020, 2021, 2022, 2023

Version :
0.1
