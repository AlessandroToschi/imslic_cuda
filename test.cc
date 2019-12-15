#include <iostream>
#include <memory>
#include <assert.h>
#include <string>
#include <fstream>

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include <cuda.h>
#include <cuda_runtime.h>

#include <npp.h>
#include <nppi.h>

#define NPP_SAFE_CALL(status){npp_check_status(status, __FILE__, __LINE__);}
#define CUDA_SAFE_CALL(status){cuda_check_status(status, __FILE__, __LINE__);}

inline void npp_check_status(const NppStatus status, const char *file, const int line)
{
    if(status != NppStatus::NPP_SUCCESS)
    {
        std::cerr <<    "NPP error: " << status << " at line "
                        << line << " in file " << file << std::endl;

        std::abort();
    }
}

inline void cuda_check_status(const cudaError_t status, const char *file, const int line)
{
    if(status != cudaError::cudaSuccess)
    {
        std::cerr <<    "Cuda error: " << cudaGetErrorString(status) << " at line "
                        << line << " in file " << file << std::endl;

        std::abort();
    }
}

template<typename T> void alloc_gpu_data(std::shared_ptr<T> &gpu_pointer, const int N)
{
    T *gpu_pointer_temp;
    static auto cuda_deleter = [&](T *pointer){CUDA_SAFE_CALL(cudaFree(pointer));};
    CUDA_SAFE_CALL(cudaMallocManaged<T>(&gpu_pointer_temp, sizeof(T) * N));
    gpu_pointer = std::shared_ptr<T>(gpu_pointer_temp, cuda_deleter);
}

void dump_to_file(const void *array, const size_t bytes, const std::string &file_path)
{
    auto binary_file = std::fstream(file_path, std::ios::out | std::ios::binary);
    binary_file.write((const char *)array, bytes);
    binary_file.close();
}

cv::Mat load_image(const std::string &image_path)
{
    cv::Mat bgr_image = cv::imread(image_path, cv::IMREAD_COLOR);

    if(bgr_image.data == NULL)
    {
        std::cerr << "Unable to open the image, aborting." << std::endl;
        std::abort();
    }
    else
    {
        std::cout << "Loaded image with size (" <<
                     bgr_image.rows << ", " <<
                     bgr_image.cols << "," <<
                     bgr_image.channels() << ")" << std::endl;
    }
    
    return bgr_image;
}

int main(int argc, char** argv)
{
    const int width = 3;
    const int height = 4;
    int image_step, filter_step;
    Npp32f *image = nppiMalloc_32f_C3(width, height, &image_step);
    Npp32f *filter = nppiMalloc_32f_C3(width, height, &filter_step);

    const int host_image_step = sizeof(Npp32f) * width * 3;
    Npp32f *host_image = new Npp32f[width * height * 3];

    for(int i = 0, counter = 0, index = 0; i < height; i++)
    {
        for(int j = 0; j < width; j++, counter++)
        {
            for(int k = 0; k < 3; k++, index++)
            {
                host_image[index] = (Npp32f)counter;
            }
        }
    }

    CUDA_SAFE_CALL(
        cudaMemcpy2D(
            image,
            image_step,
            host_image,
            host_image_step,
            host_image_step,
            height,
            cudaMemcpyHostToDevice
        )
    );

    NPP_SAFE_CALL(
        nppiFilterBoxBorder_32f_C3R(
            image,
            image_step,
            {width, height},
            {0, 0},
            filter,
            filter_step,
            {width, height},
            {2, 2},
            {1, 1},
            NppiBorderType::NPP_BORDER_REPLICATE
        )
    );

    CUDA_SAFE_CALL(
        cudaMemcpy2D(
            host_image,
            host_image_step,
            filter,
            filter_step,
            host_image_step,
            height,
            cudaMemcpyDeviceToHost
        )
    );

    for(int i = 0, index = 0; i < height; i++)
    {
        for(int j = 0; j < width; j++)
        {
            std::cout << "(";
            for(int k = 0; k < 3; k++, index++)
            {
                std::cout << host_image[index] << " ";
            }
            std::cout << ") ";
        }
        std::cout << std::endl;
    }

    delete[] host_image;

    nppiFree(image);
    nppiFree(filter);

    return EXIT_SUCCESS;
}