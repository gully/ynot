# Outline

- **Introduction**
	- The remarkable promise and pervasiveness of astronomical spectroscopy
	- The challenge: instrumental defects and imperfections
	- Previous work: optimal extraction
	- Previous work: spectroperfectionism
	- Common assumptions embedded into current techniques:
		- Single point source
		- Source has continuum
		- Continuum is high SNR
		- Relatively few segments of uninterrupted continuum
		- Trends adequately captured by polynomials
		- Symmetric, Gaussian-like PSF
	- Many heuristics designed to cope departures from these assumptions
	- Extreme or unusual spectra break these assumptions
		- brown dwarfs with low SNR and highly structured/missing spectra
		- emission line spectra (no continuum)
		- binary stars on the same slit
		- extended objects (non-point sources)
		- EPRV
	- Philosphy: Each previous spectrum should inform future spectra
	- Philosphy: Treat the instrument as a breathing, dynamic system
	- The epic hero: autodiff and GPUs, PyTorch, flexible models
	- Why this is only recently possible
	- Why this may be in some ways easier to reason about than heuristics
	- This work: A new autodiff-aware 2D echellogram modeling framework
- **The method**
	- A mapping of (x, y) pixel coordinates to (lambda, s) physical coordinates
	- How to represent the target spectrum
	- How to represent the sky spectrum
	- How to represent the target PSF
	- How to represent the sky spatial extent
	- How to represent the slit 
- **Models for multiple data sources**
	- Arcs: sparsely encode both wavelength and slit position
	- Flats: encode just slit position
	- Darks: encode just background
	- Target spectra: encode wavelength, slit position, target position
	- The likelihood function and per-pixel uncertainties
- **Validation: Injection/recovery**
	- Injection/recovery test with noisy data: generating fake data
	- Initializing of the model and optimization setup
	- Training computational performance, number of epochs, batching/sparsity
	- Best fit model comparison 1: injection/recovery of initial parameters
	- Best fit model comparison 2: spectrum as unbinned, weighted samples
- **Training on real data**
	- Introduction to real data
	- Data-preprocessing and heuristics
	- Outcome: reduced spectrum as unbinned, weighted samples
	- SNR improvement compared to previous methods (head-to-head)
- **Discussion**
	- The promise for EPRV
		- Simulation of minor RV shifts
		- Simulation of sub-pixel flat fields
		- Tracking spectrograph state across decades of operation
	- Ability to repurpose non-standard data (variable data quality)
	- Conceivable extensions
	
