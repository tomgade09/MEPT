# PTEM - Particle Tracing model for Ensembles in the Magnetosphere

Developers: Tom Gade and Alex Gullickson

PTEM is a particle tracing simulation whose objective is to reproduce electron distributions under the influence of a number of physical phenomena (mirror force, quasi-static potential structures, alfven waves, etc) gathered by FAST.  It is written with C++/CUDA, and OpenMP.

This code is a modified form of the coupled pair of simulations, C-ADPIC-PTEM.  The code base for PTEM was separated from the pair to be used as the basis for project work for EE 5351 - Applied Parallel Programming.  The finished project will be merged with the existing codebase of C-ADPIC-PTEM when complete.


## Compatibility
Right now, PTEM runs on Windows and Linux.  As it is dependent on CUDA, it requires an NVIDIA graphics card with up-to-date(-ish, at least) drivers, as well as the CUDA libraries installed.  No other external dependencies for C++ exist.


## Dependencies
CUDA (see above - Compatibility)

## Getting Started

#### 1. Download Repository

##### Platform Agnostic

  ```
  git clone https://github.umn.edu/gadex007/PTEM
  cd PTEM
  ```
  
#### 2. Compile

##### Windows

Open the Visual Studio solution (`PTEM/vs/PTEM.sln`), ensure `Release` and `x64` is selected, and in the toolbar click `Build -> Build Solution`.

##### Linux
  
  ```
  chmod 777 ./configure
  ./configure
  make
  ```

Note: gcc compatible with the `-std=c++14` flag is required.


#### 3. Run an example simulation

##### 3a. Create a Particle Distribution
From a terminal in `%WHEREVER%/C-ADPIC-PTEM`

  ```
  cd bin
  ./PTEM
  ```


##### 3b. Run Simulation with Example Characteristics

  ```
  (from PTEM)
  cd bin
  ./PTEM
  ```
  
The Python script will create the appropriate directories for you (`PTEM/_dataout/%DATE-TIME-GROUP%`) and save data after the fact.  See the documentation for the save output folder structure.  The characteristics of the simulation will be printed to the output (BField model, EField model, etc).


## Additional Documentation
[Read the documentation here](./docs/README.md)