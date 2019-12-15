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
    cv::Mat image = cv::imread(image_path, cv::IMREAD_COLOR);

    if(image.data == NULL)
    {
        std::cerr << "Unable to open the image, aborting." << std::endl;
        std::abort();
    }
    else
    {
        std::cout << "Loaded image with size (" <<
                     image.rows << ", " <<
                     image.cols << ", " <<
                     image.channels() << ")" <<
                     " and " << image.step << " bytes of step." << std::endl;
    }
    
    return image;
}

Npp32f *bgr_to_lab_image(
    const Npp8u *bgr_image,
    const int bgr_image_step,
    int *lab_image_step, 
    const int width, 
    const int height
)
{
    int lab_image_uchar3_step;
    Npp8u *lab_image_uchar3 = nppiMalloc_8u_C3(width, height, &lab_image_uchar3_step);

    Npp32f *lab_image_float3 = nppiMalloc_32f_C3(width, height, lab_image_step);

    NPP_SAFE_CALL(
        nppiBGRToLab_8u_C3R(
            bgr_image,
            bgr_image_step,
            lab_image_uchar3,
            lab_image_uchar3_step,
            {width, height}
        )
    )

    NPP_SAFE_CALL(
        nppiConvert_8u32f_C3R(
            lab_image_uchar3,
            lab_image_uchar3_step,
            lab_image_float3,
            *lab_image_step,
            {width, height}
        )
    );

    nppiFree(lab_image_uchar3);

    return lab_image_float3;
}

Npp32f *get_area(const Npp32f *lab_image, const int lab_image_step, int *area_step, const int width, const int height)
{
    Npp32f *area = nppiMalloc_32f_C1(width, height, area_step);

    int nw_filter_step, sw_filter_step, se_filter_step, ne_filter_step;
    Npp32f *nw_filter = nppiMalloc_32f_C3(width, height, &nw_filter_step);
    Npp32f *sw_filter = nppiMalloc_32f_C3(width, height, &sw_filter_step);
    Npp32f *se_filter = nppiMalloc_32f_C3(width, height, &se_filter_step);
    Npp32f *ne_filter = nppiMalloc_32f_C3(width, height, &ne_filter_step);

    NPP_SAFE_CALL(
        nppiFilterBoxBorder_32f_C3R(
            lab_image,
            lab_image_step,
            {width, height},
            {0, 0},
            nw_filter,
            nw_filter_step,
            {width, height},
            {2, 2},
            {1, 1},
            NppiBorderType::NPP_BORDER_REPLICATE
        )
    );

    NPP_SAFE_CALL(
        nppiFilterBoxBorder_32f_C3R(
            lab_image,
            lab_image_step,
            {width, height},
            {0, 0},
            sw_filter,
            sw_filter_step,
            {width, height},
            {2, 2},
            {1, 0},
            NppiBorderType::NPP_BORDER_REPLICATE
        )
    );

        NPP_SAFE_CALL(
        nppiFilterBoxBorder_32f_C3R(
            lab_image,
            lab_image_step,
            {width, height},
            {0, 0},
            se_filter,
            se_filter_step,
            {width, height},
            {2, 2},
            {0, 0},
            NppiBorderType::NPP_BORDER_REPLICATE
        )
    );

    NPP_SAFE_CALL(
        nppiFilterBoxBorder_32f_C3R(
            lab_image,
            lab_image_step,
            {width, height},
            {0, 0},
            ne_filter,
            ne_filter_step,
            {width, height},
            {0, 1},
            {1, 1},
            NppiBorderType::NPP_BORDER_REPLICATE
        )
    );

    NPP_SAFE_CALL(
        nppiSub_32f_C3IR(
            nw_filter,
            nw_filter_step,
            se_filter,
            se_filter_step,
            {width, height}
        )
    );

    NPP_SAFE_CALL(
        nppiSub_32f_C3IR(
            nw_filter,
            nw_filter_step,
            se_filter,
            se_filter_step,
            {width, height}
        )
    );

    NPP_SAFE_CALL(
        nppiSub_32f_C3IR(
            ne_filter,
            ne_filter_step,
            se_filter,
            se_filter_step,
            {width, height}
        )
    );

    NPP_SAFE_CALL(
        nppiSub_32f_C3IR(
            nw_filter,
            nw_filter_step,
            sw_filter,
            sw_filter_step,
            {width, height}
        )
    );

    nppiFree(nw_filter);
    nppiFree(sw_filter);
    nppiFree(se_filter);
    nppiFree(ne_filter);

    return area;
}

int main(int argc, char** argv)
{
    cv::Mat bgr_image_host = load_image("./1.jpg");
    cv::Mat bgr_image_host_copy = bgr_image_host.clone();
    cv::cvtColor(bgr_image_host_copy, bgr_image_host_copy, cv::COLOR_BGR2Lab);

    int bgr_image_device_step;
    Npp8u *bgr_image_device = nppiMalloc_8u_C3(bgr_image_host.cols, bgr_image_host.rows, &bgr_image_device_step);

    CUDA_SAFE_CALL(
        cudaMemcpy2D(
            bgr_image_device,
            bgr_image_device_step,
            bgr_image_host.data,
            bgr_image_host.step,
            bgr_image_host.step,
            bgr_image_host.rows,
            cudaMemcpyHostToDevice
        )
    );

    int lab_image_device_step;
    Npp32f *lab_image_device = bgr_to_lab_image(
        bgr_image_device,
        bgr_image_device_step,
        &lab_image_device_step,
        bgr_image_host.cols,
        bgr_image_host.rows
    );

    Npp32f *lab_image_host = new Npp32f[bgr_image_host.total() * bgr_image_host.channels()];

    CUDA_SAFE_CALL(
        cudaMemcpy2D(
            lab_image_host,
            bgr_image_host.cols * bgr_image_host.elemSize(),
            lab_image_device,
            lab_image_device_step,
            bgr_image_host.cols * bgr_image_host.elemSize(),
            bgr_image_host.rows,
            cudaMemcpyDeviceToHost
        )
    );

    int area_device_step;
    auto area_device = get_area(lab_image_device, lab_image_device_step, &area_device_step, bgr_image_host.cols, bgr_image_host.rows);

    delete[] lab_image_host;

    nppiFree(bgr_image_device);
    nppiFree(lab_image_device);
    nppiFree(area_device);

    return EXIT_SUCCESS;
}