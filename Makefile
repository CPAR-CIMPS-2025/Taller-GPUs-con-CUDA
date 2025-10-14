NVCC ?= nvcc
CXXFLAGS = -O3 -std=c++17

all: vecadd reduce

vecadd: 01_vector_add.cu common.hpp
	$(NVCC) $(CXXFLAGS) $< -o $@

reduce: 02_reduction_shared.cu common.hpp
	$(NVCC) $(CXXFLAGS) $< -o $@

clean:
	rm -f vecadd reduce
