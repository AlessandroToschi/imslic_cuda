OPENCV_LIB_PATH=/usr/local/lib/
OPENCV_INCLUDE_PATH=/usr/local/include/opencv4/
OPENCV_LIB= -lopencv_core -lopencv_imgcodecs -lopencv_highgui -lopencv_imgproc

CUDA_LIB_PATH=/usr/local/cuda/lib64/
CUDA_INCLUDE_PATH=/usr/local/cuda/include/
CUDA_LIB=-lcudart -lnppc -lnppicc -lnppidei -lnppif -lnppisu -lnppial

all:
	g++ \
	-O3 -g -std=c++11 -march=native \
	-I $(OPENCV_INCLUDE_PATH) \
	-I $(CUDA_INCLUDE_PATH) \
	imslic.cc \
	-o imslic \
	-L $(OPENCV_LIB_PATH) \
	-L $(CUDA_LIB_PATH) \
	$(OPENCV_LIB) \
	$(CUDA_LIB)

pippo:
	g++ \
	-O3 -g -std=c++11 -march=native \
	-I $(OPENCV_INCLUDE_PATH) \
	-I $(CUDA_INCLUDE_PATH) \
	test.cc \
	-o test \
	-L $(OPENCV_LIB_PATH) \
	-L $(CUDA_LIB_PATH) \
	$(OPENCV_LIB) \
	$(CUDA_LIB)