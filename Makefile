# Compiler and flags
NVCC = nvcc
CXXFLAGS = -Xcompiler /openmp -g -G -O3

# Executable name
TARGET = Hyperblocks

# Source files
CU_SRCS = \
    ./Host.cu \
    ./hyperblock_generation/MergerHyperBlock.cu \
    ./interval_hyperblock/IntervalHyperBlock.cu \
    ./simplifications/Simplifications.cu

CPP_SRCS = \
    ./hyperblock/HyperBlock.cpp \
    ./cuda_util/CudaUtil.cpp \
    ./data_utilities/DataUtil.cpp \
    ./knn/Knn.cpp \
    ./screen_output/PrintingUtil.cpp \
    ./ClassificationTesting/ClassificationTests.cpp

# Object files
CU_OBJS = $(CU_SRCS:.cu=.o)
CPP_OBJS = $(CPP_SRCS:.cpp=.o)
OBJS = $(CU_OBJS) $(CPP_OBJS)

# Default target
all: $(TARGET)

$(TARGET): $(OBJS)
	$(NVCC) $(CXXFLAGS) -o $@ $^

# Compilation rules
%.o: %.cu
	$(NVCC) $(CXXFLAGS) -c $< -o $@

%.o: %.cpp
	$(NVCC) $(CXXFLAGS) -c $< -o $@

# Clean rule
clean:
	rm -f $(OBJS) $(TARGET)