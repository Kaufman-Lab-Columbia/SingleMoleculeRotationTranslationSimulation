# SingleMoleculeRotationTranslationSimulation
This code simulates simultaneous rotational and translational motion of single molecule fluorescent probes via random walks to approximate the heterogeneous dynamics exhibited by supercooled liquid systems. The simulations output as .tif files of two orthogonally polarized optical channels. These simulations are written to utilize GPU acceleration on CUDA cores but the computations could be adapted to run just on the CPU as C code.  

# Getting Started
## Dependencies
The current version implementation of these simulations requires installation of the NVIDIA CUDA Toolkit on a computer with an NVIDIA graphics card installed. The NVIDIA CUDA Toolkit can be downloaded at the following link: https://developer.nvidia.com/cuda-downloads

This code can be adapted for use without CUDA and this repository may be updated in the future to include such an implementation. 

# Basic Usage
  ## Modifying Parameters
The user is able to modify the following parameters by editting their values from the .cu file: 

- General Parameters:

  - ***HEIGHT*** and ***WIDTH***: Determines the size (pixel height x pixel width) of the resulting movie. <br/> <br/>

  - ***init_pos***: Determines how the initial XY-positions of the features are determined.
    - *0*: The initial positions of all features are randomly determined.
    - *1*: The initial positions are placed upon a grid. <br/> <br/>

   - ***feats*** (int): Only applicable if *init_pos = 0*. Determines the number of molecules to simulate.
   - ***separation*** (int): Only applicable if *init_pos = 1*. Determines how far apart (pixels) simulated moviles must be on the grid. This also thus determines the number of molecules simulated. <br/> <br/>

   - ***frames*** (int): Determines the number of frames to simulate.
   - ***pixel_size*** (int): Determines the size (nm) of each pixel.
   - ***tbf*** (float): Determines the time between frame (s).
   - ***aperture_detect*** (float): Determines the numerical aperture (NA) of the simulated collection objective lens.
   - ***PSF_fwhm*** (int): Determines the full-width half-max (FWHM) of the simulated Point Spread Function (PSF)
   - ***add_Excitation*** (int): Determines whether excitation contributions are accounted for. (0) No and (1) Yes <br/> <br/>

   - ***photons_per_frame*** (float): Scales the resulting feature intensity by an anticipated number of photons per frame. 
   - ***photomultiplier_gain*** (float): Scales the resulting feature intensity by an anticipated photomultiplier gain value. 
   - ***bkg_constant*** (float): Adds a constant background value to all pixels. 
   - ***bkg_amp*** (float): Scales the amplitude of random background noise.  <br/> <br/>

   - ***ratio*** (float): **Typically not used.** Artificially scales the ratio of in-plane and out-of-plane rotation contributions; 1.0 is isotropic. <br/> <br/>
 
   - ***exch_type*** (int): Indicates the type of exchange you want to simulate. (1) Uniform Dynamic Exchange (2) Correlated Exchange following http://dx.doi.org/10.1063/1.4915267 <br/> <br/>

- Translation Specific:
  - ***D_med*** (float): Median value of the sampled log_10(Dmsd) distribution.
  - ***D_fwhm*** (float): FWHM of the log_10(Dmsd) distribution.
  - ***D_corr*** (float): Correlation of Dmsd to the previous timescale (i.e.; fast molecules are more likely to remain fast and slow molecules are more likely to remain slow). ***THIS IS NOT CURRENTLY IMPLEMENTED***

  - ***Dxch_med*** (float): If *exch_type = 1*: this is the medium number of frames before exchanging to a new Dmsd. If *exch_type = 2*: ***THIS IS NOT CURRENTLY IMPLEMENTED***
  - ***Dxch_fwhm*** (float): FWHM of the log_10(D_exchange) distribution ***THIS IS NOT CURRENTLY IMPLEMENTED***
  - ***Dxch_corr***(float): Correlation of exchange time with Dmsd (i.e.; fast molecules exchange timescales frequently and slow molecules exchange timescales less frequently) ***THIS IS NOT CURRENTLY IMPLEMENTED***

- Rotation Specific:
  - ***tau_med*** (float): Median value of the sampled log_10(tauc) distribution.
  - ***tau_fwhm*** (float): FWHM of the log_10(tauc) distribution.
  - ***tau_corr*** (float): Correlation of tau to previous timescales (i.e.; fast molecules are more likely to remain fast and slow molecules are more likely to remain slow). ***THIS IS NOT CURRENTLY IMPLEMENTED***

  - ***xch_med*** (float): If *exch_type = 1*: This is the medium number of frames before exchanging to a new tauc. If *exch_type = 2*: This is the medium of the distribution upon which exchange times will be sampled.
  - ***xch_fwhm*** (float): FWHM of the log_10(tau_exchange) distribution.
  - ***xch_corr*** (float): Correlation of exchange time with tauc (i.e.; fast molecules exchange timescales frequently and slow molecules exchange timescales less frequently)
<br/>
 
## Running the Simualations
  ### Individual Simulations
  Individual instances of the simulation code can be compiled using the CUDA compiler and executed by opening the *RotationTranslationSimulation.cu* file location in a Bash terminal and entering the following commands:

```
  nvcc RotationTranslationSimulation.cu -o RotationTranslationSim.exe
  ./RotationTranslationSim.exe
```

  Once complete, the output frame files will be contained in two folders titled *lc_frames* and *rc_frames* which can be converted to *.tif* files by running the included *SimOut_to_Tif.py* code included in this repository. This may require the user to modify the first snippet of code whereby the path of the *lc_frames* or *rc_frames* folders is specified.

  ### Multiple Simulations with Shell Scripts
    
  More often than not the user will likely be interested in running a number of simulations with potentially varying parameters. As such, this code can also be run using the shell scripts included in this repository if you have many simulations to run. This will execute all simulations in series; to execute in parallel simply run this same process in a different working directory from another instance of the Bash terminal.
  <br/>
  <br/>
  In this case place each *RotationTranslationSimulation.cu* file (with its parameters adjusted to your preference) in its own folder and nest these folders within another folder. As such your file directory should look something like:

  *./
  <br/>
  ./toSimulate
  <br/>
  ./toSimulate/SimConditions1
  <br/>
  ./toSimulate/SimConditions2
  <br/>
  etc.*

 Then, in the working directory, *./*, execute the following command: 
 
 ```bash batchsim.sh```

 This will then check every folder in your working directory (*./toSimulate* above) to see if it contains subfolders (*./toSimulate/SimConditions1* for example) which contain a copy of the *RotationTranslationSimulation.cu* file and, if so, will compile and execute this code. 
 
All the output frame files can then be converted to *.tif* files with the following command once again executed from the working directory *./*:

```bash batchConvertToTif.sh```

***NOTE:*** The path of *SimOut_to_Tif.py* in this bash script will need to be modified to reflect the location of this Python code in your file directory. 

# License
SingleMoleculeRotationTranslationSimulation is licensed with an MIT license. See LICENSE file for more information.

# Referencing
If you use SingleMoleculeRotationTranslationSimulation for your work, cite it with the following:
```
Alec R. Meacham, Jaladhar Mahato, Han Yang, and Laura J. Kaufman
The Journal of Physical Chemistry B 2024 128 (38), 9233-9243
DOI: 10.1021/acs.jpcb.4c02097
```

# Contact
No Current Contact








