# PTEM - Particle Tracing model for Ensembles in the Magnetosphere

Developers: Tom Gade and Alex Gullickson

PTEM is a particle tracing simulation whose objective is to reproduce electron distributions under the influence of a number of physical phenomena (mirror force, quasi-static potential structures, alfven waves, etc) gathered by FAST.  It is written with C++/CUDA, and OpenMP.

This code is a modified form of the coupled pair of simulations, C-ADPIC-PTEM.  The code base for PTEM was separated from the pair to be used as the basis for project work for EE 5351 - Applied Parallel Programming.  The finished project will be merged with the existing codebase of C-ADPIC-PTEM when complete.


## Compatibility
PTEM runs on Windows and Linux.  As it is dependent on CUDA, it requires an NVIDIA graphics card with up-to-date(-ish, at least) drivers, as well as the CUDA libraries installed.  No other external dependencies for C++ exist.
NOTE: This does not compile properly on the gpulab machines due to gcc and nvcc version issues.  This is due to the use of the C++ filesystem library.  As a result of using this, the code must be compiled via a version of gcc (Linux) that supports C++17 and the filesystem library.  GCC 9 and up should work fine, but 10 is likely best.  Additionally, CUDA 11.1 was used, although 10 and up should work fine.


## Multi-GPU Execution
PTEM can be run on single or multiple GPU systems.  Every CUDA capable GPU on the system will be utilized unless limited by the environment variable

	```
	CUDA_DEVICES_VISIBLE
	```

On Windows, this can be set from powershell immediately prior to running the program.  From the bin folder:

	```
	$Env:CUDA_VISIBLE_DEVICES="0,1"; ./PTEM
	Note: the desired devices can be set through this flag.  e.g. using only 0 or only 1, etc
	Observe the "" if using a comma to separate devices
	```
	
On Linux, this is implemented by typing

	```
	CUDA_VISIBLE_DEVICES=0,1 ./PTEM
	Note: the same note above applies
	```
	

## Dependencies
CUDA (see above - Compatibility)

## Getting Started

#### 1. Unpack Repository

##### Platform Agnostic

Unzip the zip file using your preferred zip file program.
  
#### 2. Compile

##### Windows

Open the Visual Studio solution (`PTEM/vs/PTEM.sln`), ensure `Release` and `x64` is selected, and in the toolbar click `Build -> Build Solution`.

##### Linux
  
  ```
  chmod 777 ./configure
  ./configure
  make
  ```

Note: gcc compatible with the `-std=c++17` flag is required.


#### 3. Run an example simulation

##### 3a. Create a Particle Distribution
From a terminal in `%WHEREVER%/PTEM`

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
  
The program will create appropriate directories and save the data after execution in the folder _dataout.  The characteristics of the simulation (BField model, EField model, etc) will be printed to the logfile contained within the data folder.