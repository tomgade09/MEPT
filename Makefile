# Compiler binaries and settings
CC         := g++
NVCC       := nvcc
CXXFLAGS   := -std=c++17 -pedantic -O2 -fopenmp -Wall -Wno-unused-variable
NVCFLAGS   := -std=c++14 -ccbin=$(CC) -rdc=true -O2 -gencode=arch=compute_50,code=\"sm_50,compute_50\" -x cu -m64 -cudart static -Xlinker "-fopenmp" -Wno-deprecated-gpu-targets
LINKFLAGS  := -lcudadevrt -Xcompiler "-fopenmp"

# Build-essential directories and defines
CUDAINC    := /usr/local/cuda-12.1/include
CUDALIB    := /usr/local/cuda-12.1/lib64
LIBS       := -L$(CUDALIB)
INCS       := -I$(CUDAINC) -I./include
SRC        := ./src
OUT        := ./bin
BUILD      := $(OUT)/build
TARGET     := $(OUT)/MEPT

# Lists of files
SOURCES    := $(shell find $(SRC) -name "*.cpp")# | sed 's/^\.\/src\/SimAttributes\/SimAttributes\.cpp//')
OBJECTS    := $(patsubst $(SRC)/%.cpp,$(BUILD)/%.o,$(SOURCES))
CUSOURCES  := $(shell find $(SRC) -name "*.cu")
CUOBJECTS  := $(patsubst $(SRC)/%.cu,$(BUILD)/%.obj,$(CUSOURCES))

#Default rule
$(TARGET): $(OBJECTS) $(CUOBJECTS)
	$(NVCC) $(LINKFLAGS) -o $(TARGET) $(OBJECTS) $(CUOBJECTS)


# File rules
$(BUILD)/main.o: $(SRC)/main.cpp
	$(CC) $(CXXFLAGS) -c $(INCS) $(LIBS) $(SRC)/main.cpp -o $(BUILD)/main.o

$(BUILD)/BField/DipoleB.o: $(SRC)/BField/DipoleB.cpp
	$(CC) $(CXXFLAGS) -c $(INCS) $(LIBS) $(SRC)/BField/DipoleB.cpp -o $(BUILD)/BField/DipoleB.o

$(BUILD)/BField/DipoleBLUT.o: $(SRC)/BField/DipoleBLUT.cpp
	$(CC) $(CXXFLAGS) -c $(INCS) $(LIBS) $(SRC)/BField/DipoleBLUT.cpp -o $(BUILD)/BField/DipoleBLUT.o
	
$(BUILD)/EField/EField.o: $(SRC)/EField/EField.cpp
	$(CC) $(CXXFLAGS) -c $(INCS) $(LIBS) $(SRC)/EField/EField.cpp -o $(BUILD)/EField/EField.o
	
$(BUILD)/EField/QSPS.o: $(SRC)/EField/QSPS.cpp
	$(CC) $(CXXFLAGS) -c $(INCS) $(LIBS) $(SRC)/EField/QSPS.cpp -o $(BUILD)/EField/QSPS.o
	
$(BUILD)/Log/Log.o: $(SRC)/Log/Log.cpp
	$(CC) $(CXXFLAGS) -c $(INCS) $(LIBS) $(SRC)/Log/Log.cpp -o $(BUILD)/Log/Log.o

$(BUILD)/Particles/Particles.o: $(SRC)/Particles/Particles.cpp
	$(CC) $(CXXFLAGS) -c $(INCS) $(LIBS) $(SRC)/Particles/Particles.cpp -o $(BUILD)/Particles/Particles.o

$(BUILD)/Satellite/Detector.o: $(SRC)/Satellite/Detector.cpp
	$(CC) $(CXXFLAGS) -c $(INCS) $(LIBS) $(SRC)/Satellite/Detector.cpp -o $(BUILD)/Satellite/Detector.o

$(BUILD)/Satellite/Satellite.o: $(SRC)/Satellite/Satellite.cpp
	$(CC) $(CXXFLAGS) -c $(INCS) $(LIBS) $(SRC)/Satellite/Satellite.cpp -o $(BUILD)/Satellite/Satellite.o
	
$(BUILD)/Simulation/Simulation.o: $(SRC)/Simulation/Simulation.cpp
	$(CC) $(CXXFLAGS) -c $(INCS) $(LIBS) $(SRC)/Simulation/Simulation.cpp -o $(BUILD)/Simulation/Simulation.o

$(BUILD)/Simulation/iterateSimCPU.o: $(SRC)/Simulation/iterateSimCPU.cpp
	$(CC) $(CXXFLAGS) -c $(INCS) $(LIBS) $(SRC)/Simulation/iterateSimCPU.cpp -o $(BUILD)/Simulation/iterateSimCPU.o
	
$(BUILD)/utils/fileIO.o: $(SRC)/utils/fileIO.cpp
	$(CC) $(CXXFLAGS) -c $(INCS) $(LIBS) $(SRC)/utils/fileIO.cpp -o $(BUILD)/utils/fileIO.o

$(BUILD)/utils/numerical.o: $(SRC)/utils/numerical.cpp
	$(CC) $(CXXFLAGS) -c $(INCS) $(LIBS) $(SRC)/utils/numerical.cpp -o $(BUILD)/utils/numerical.o

$(BUILD)/utils/random.o: $(SRC)/utils/random.cpp
	$(CC) $(CXXFLAGS) -c $(INCS) $(LIBS) $(SRC)/utils/random.cpp -o $(BUILD)/utils/random.o

$(BUILD)/utils/readIOclasses.o: $(SRC)/utils/readIOclasses.cpp
	$(CC) $(CXXFLAGS) -c $(INCS) $(LIBS) $(SRC)/utils/readIOclasses.cpp -o $(BUILD)/utils/readIOclasses.o

$(BUILD)/utils/serializationHelpers.o: $(SRC)/utils/serializationHelpers.cpp
	$(CC) $(CXXFLAGS) -c $(INCS) $(LIBS) $(SRC)/utils/serializationHelpers.cpp -o $(BUILD)/utils/serializationHelpers.o

$(BUILD)/utils/strings.o: $(SRC)/utils/strings.cpp
	$(CC) $(CXXFLAGS) -c $(INCS) $(LIBS) $(SRC)/utils/strings.cpp -o $(BUILD)/utils/strings.o

$(BUILD)/utils/writeIOclasses.o: $(SRC)/utils/writeIOclasses.cpp
	$(CC) $(CXXFLAGS) -c $(INCS) $(LIBS) $(SRC)/utils/writeIOclasses.cpp -o $(BUILD)/utils/writeIOclasses.o

$(BUILD)/API/utilsAPI.o: $(SRC)/API/utilsAPI.cpp
	$(CC) $(CXXFLAGS) -c $(INCS) $(LIBS) $(SRC)/API/utilsAPI.cpp -o $(BUILD)/API/utilsAPI.o

$(BUILD)/API/LogAPI.o: $(SRC)/API/LogAPI.cpp
	$(CC) $(CXXFLAGS) -c $(INCS) $(LIBS) $(SRC)/API/LogAPI.cpp -o $(BUILD)/API/LogAPI.o

$(BUILD)/API/SimulationAPI.o: $(SRC)/API/SimulationAPI.cpp
	$(CC) $(CXXFLAGS) -c $(INCS) $(LIBS) $(SRC)/API/SimulationAPI.cpp -o $(BUILD)/API/SimulationAPI.o


#CUDA Source Files
$(BUILD)/BField/BModel.obj: $(SRC)/BField/BModel.cu
	$(NVCC) $(NVCFLAGS) -c $(INCS) $(LIBS) $(SRC)/BField/BModel.cu -o $(BUILD)/BField/BModel.obj

$(BUILD)/BField/DipoleB.obj: $(SRC)/BField/DipoleB.cu
	$(NVCC) $(NVCFLAGS) -c $(INCS) $(LIBS) $(SRC)/BField/DipoleB.cu -o $(BUILD)/BField/DipoleB.obj

$(BUILD)/BField/DipoleBLUT.obj: $(SRC)/BField/DipoleBLUT.cu
	$(NVCC) $(NVCFLAGS) -c $(INCS) $(LIBS) $(SRC)/BField/DipoleBLUT.cu -o $(BUILD)/BField/DipoleBLUT.obj

$(BUILD)/EField/QSPS.obj: $(SRC)/EField/QSPS.cu
	$(NVCC) $(NVCFLAGS) -c $(INCS) $(LIBS) $(SRC)/EField/QSPS.cu -o $(BUILD)/EField/QSPS.obj

$(BUILD)/EField/EField.obj: $(SRC)/EField/EField.cu
	$(NVCC) $(NVCFLAGS) -c $(INCS) $(LIBS) $(SRC)/EField/EField.cu -o $(BUILD)/EField/EField.obj

$(BUILD)/EField/EModel.obj: $(SRC)/EField/EModel.cu
	$(NVCC) $(NVCFLAGS) -c $(INCS) $(LIBS) $(SRC)/EField/EModel.cu -o $(BUILD)/EField/EModel.obj

$(BUILD)/Satellite/Detector.obj: $(SRC)/Satellite/Detector.cu
	$(NVCC) $(NVCFLAGS) -c $(INCS) $(LIBS) $(SRC)/Satellite/Detector.cu -o $(BUILD)/Satellite/Detector.obj

$(BUILD)/Simulation/Environment.obj: $(SRC)/Simulation/Environment.cu
	$(NVCC) $(NVCFLAGS) -c $(INCS) $(LIBS) $(SRC)/Simulation/Environment.cu -o $(BUILD)/Simulation/Environment.obj
	
$(BUILD)/Simulation/simulationphysics.obj: $(SRC)/Simulation/simulationphysics.cu
	$(NVCC) $(NVCFLAGS) -c $(INCS) $(LIBS) $(SRC)/Simulation/simulationphysics.cu -o $(BUILD)/Simulation/simulationphysics.obj

$(BUILD)/utils/arrayUtilsGPU.obj: $(SRC)/utils/arrayUtilsGPU.cu
	$(NVCC) $(NVCFLAGS) -c $(INCS) $(LIBS) $(SRC)/utils/arrayUtilsGPU.cu -o $(BUILD)/utils/arrayUtilsGPU.obj


.PHONY: clean
clean:
	find $(OUT) -name "*.o" -type f -delete
	find $(OUT) -name "*.obj" -type f -delete
	rm $(TARGET)

#.PHONY: exec
#exec:
#	TARGET     := $(OUT)/PTEM
#	CXXFLAGS   := -std=c++17 -pedantic -O2 -fopenmp -D_DEBUG -Wall -Wno-unused-variable
#	NVCFLAGS   := -std=c++14 -rdc=true -O2 -gencode=arch=compute_30,code=\"sm_30,compute_30\" -x cu -m64 -cudart static -Xlinker "-fopenmp"
#	LINKFLAGS  := -lcudadevrt -Xcompiler "-fopenmp" -Xlinker "-fopenmp"
