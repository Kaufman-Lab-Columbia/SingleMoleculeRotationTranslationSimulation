# SingleMoleculeRotationTranslationSimulation
This code simulates simultaneous rotational and translational motion of single molecule fluorescent probes via random walks to approximate the heterogeneous dynamics exhibited by supercooled liquid systems. The simulations output as .tif files of two orthogonally polarized optical channels. These simulations are written to utilize GPU acceleration on CUDA cores but the computations could be adapted to run just on the CPU. 

# Getting Started
## Dependencies
The current version implementation of these simulations requires installation of the NVIDIA CUDA Toolkit on a computer with an NVIDIA graphics card installed. The NVIDIA CUDA Toolkit can be downloaded at the following link: https://developer.nvidia.com/cuda-downloads

This code can be adapted for use without CUDA and this repository may be updated in the future to include such an implementation. 

# Basic Usage
The user is able to modify the following parameters by editting their values from the .cu file: 

- General Parameters:

  - ***HEIGHT*** and ***WIDTH***: Determines the size (pixel height x pixel width) of the resulting movie.

  - ***init_pos***: Determines how the initial XY-positions of the features are determined.
    - *0*: The initial positions of all features are randomly determined.
    - *1*: The initial positions are placed upon a grid.
   
   - ***feats*** (int): Only applicable if *init_pos = 0*. Determines the number of molecules to simulate.
   - ***separation*** (int): Only applicable if *init_pos = 1*. Determines how far apart (pixels) simulated moviles must be on the grid. This also thus determines the number of molecules simulated.
 
   - ***frames*** (int): Determines the number of frames to simulate.
   - ***pixel_size*** (int): Determines the size (nm) of each pixel.
   - ***tbf*** (float): Determines the time between frame (s).
   - ***aperture_detect*** (float): Determines the numerical aperture (NA) of the simulated collection objective lens.
   - ***PSF_fwhm*** (int): Determines the full-width half-max (FWHM) of the simulated Point Spread Function (PSF)
   - ***add_Excitation*** (int): Determines whether excitation contributions are accounted for. (0) No and (1) Yes
 
   - ***photons_per_frame*** (float): Scales the resulting feature intensity by an anticipated number of photons per frame. 
   - ***photomultiplier_gain*** (float): Scales the resulting feature intensity by an anticipated photomultiplier gain value. 
   - ***bkg_constant*** (float): Adds a constant background value to all pixels. 
   - ***bkg_amp*** (float): Scales the amplitude of random background noise. 

   - ***ratio*** (float): **Typically not used.** Artificially scales the ratio of in-plane and out-of-plane rotation contributions; 1.0 is isotropic.
 
   - ***exch_type*** (int): Indicates the type of exchange you want to simulate. (1) Uniform Dynamic Exchange (2) Correlated Exchange following http://dx.doi.org/10.1063/1.4915267

- Translation Specific:
  - ***D_med*** (float):
  - ***D_fwhm*** (float):
  - ***D_corr*** (float):
 
  - ***Dxch_med*** (float):
  - ***Dxch_fwhm*** (float):
  - ***Dxch_corr***(float):

- Rotation Specific:
  - ***tau_med*** (float):
  - ***tau_fwhm*** (float):
  - ***tau_corr*** (float):
 
  - ***xch_med*** (float):
  - ***xch_fwhm*** (float):
  - ***xch_corr*** (float):
 
  - 


 










