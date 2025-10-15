NVCC ?= nvcc
CXXFLAGS = -O3 -std=c++17

all: vecadd reduce

vecadd: 0a_vector_add.cu common.hpp
	$(NVCC) $(CXXFLAGS) $< -o $@

reduce: 0b_reduction_shared.cu common.hpp
	$(NVCC) $(CXXFLAGS) $< -o $@

clean:
	rm -f vecadd reduce
