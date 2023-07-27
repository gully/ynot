# Outline

- **Introduction**
	- The remarkable promise and pervasiveness of astronomical spectroscopy
	- The challenge: instrumental defects and imperfections
	- Previous work: optimal extraction
	- Previous work: spectroperfectionism
	- The epic hero: autodiff and GPUs, PyTorch
	- This work: A new autodiff-aware 2D echellogram modeling framework
- **The method**
	- A mapping of (x, y) pixel coordinates to (lambda, s) physical coordinates
	- How to represent the target spectrum
	- How to represent the sky spectrum
	- How to represent the target PSF
	- How to represent the sky spatial extent
	- How to represent the slit 
	