# C-ADPIC-PTEM


### What is it?
C-ADPIC-PTEM is a simulation for generating test particle distributions as seen by a simulated satellite at a specified altitude under the influence of various B and E Field structures.


### Documentation
Click the links below to read about the various components of this software:

[0. Python](Python/README.md) - Python class that allows easy calling of the C API functions from an interactive interpreter, if desired.  Start here if you want the no-frills description of the easiest use of the library.

[1. Simulation](Simulation/README.md) - Container class that contains equation of motion for particles, as well as a number of control functions, access functions, and other useful things.  Manages lifetime of all of the below classes.

[2. BModel](BModel/README.md) - Abstract class for interfacing with various implementations of B Field models.

[3. EField](EField/README.md) - Abstract class for interfacing with various implementations of E Field models.  Can track numerous `EModel`s - E Field Elements.

[4. Particles](Particles/README.md) - Class that manages particles by tracking arrays of attributes, specified upon creation.  Also manages on GPU data, including cudaMalloc/cudaFree on initialization/destruction respectively.

[5. Satellite](Satellite/README.md) - Class that tracks whether or not a particle has passed a certain altitude from above or below.  Also manages on GPU data, including cudaMalloc/cudaFree on initialization/destruction respectively.

[6. API](API/README.md) - Mostly extern C-style API for interfacing through Python or other language.

[7. utils](utils/README.md) - Namespaced functions that accomplish a number of useful things, including generating a distribution of particles, generating a CSV from specified data, and loading/comparing particle distributions (including initial and final data, and satellite data).

[8. ErrorHandling](ErrorHandling/README.md) - Mostly macros to handle various error conditions, as well as make sure certain conditions are met.

[9. SimAttributes](SimAttributes/README.md) - Class that handles saving Simulation attributes to a file for later reference/recreation of an identical Simulation

[10. Log](Log/README.md) - A class with a number of member functions for writing a log file to disk along with elapsed time data.

[11. Examples](Examples/README.md) - Look here for examples of usage (Python and C++).

*Note: In this documentation, uppercase (and usually linked) names refer to classes, while lowercase names refer to non-class things.  For example: [Particles](Particles/README.md) refers to the class itself or an instance of the class which manages a large number of particles (lowercase).  particle(s) usually refers to a collection of attributes (ex: v_para, v_perp or mu, and s, as well as maybe time, index, etc) that represents a `real-world physical particle`.*
