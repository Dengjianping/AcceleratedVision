# AcceleratedVision

This lib is for accelerating general computer vision algorithms, and fined-tuned.

## Platform Soppurt
- [x] Windows
- [ ] Linux(Coming)

Now, it fits for Windows only, but it's easy to port to supporting Linux. 
My new toy [jetson tx2](https://www.nvidia.com/en-us/autonomous-machines/embedded-systems-dev-kits-modules/) just didn't work, back to distributor now, I have to wait for a while.

## Dependencies
1. [Cmake](https://cmake.org/) 3.10+
2. [OpenCV](https://opencv.org/) 3.x(it doesn't need to compile, just download it and extract to folder ```C:\Program Files\opencv```
3. CUDA 8.0 or later
4. VS 2015 or later
Tips: Your video card should be capable of compute capability 2.1 at least.

## Installation

1. Make sure vs2015 and CUDA8.0, cmake3.8+ installed on your computer. And opencv is installed to 
   ```C:\Program Files\opencv\build```
2. Sync this repo, 
   ```git clone https://github.com/Dengjianping/AcceleratedVision.git```
3. Compile. create a folder named 'build', and get into it by command window.                   
   ```cmake -G "Visual Studio 14 Win64" ..```. 
4. Wait it until done. If all step are successful, then type 
   ```msbuild /t:rebuild /p:Configuration=Release /p:Platform=x64 accel_vision.sln```
   this command will generate a dll and lib file, named 'accel_vision.dll' and 'accel_vision.lib" under ..\build\Release
5. Enjoy it!

## APIs List

API Name | Description
------ | ---------
cudaThreshold | threshold a image
cudaGray | RGB to gray
cudaHSV | RGB color to HSV
cudaSplit | split a 3-ch image into 1-ch image
cudaMerge | merge three 1-ch images as 3-ch image
cudaErode | erode image
cudaDilation | dilate image
cudaLUT | look up table
cudaGaussianBlut | blur image by gaussian kernel
cudaLalace | laplace operator to extract edges
cudaSobel | sobel operator to extract edges
cudaHistogram | histogram for a image
cudaEqualizeHist | equalize a image
cudaADD/OR/Sqrt | some mathematical operations
updating|---



## Usage
Copy that folder named **include** to your project folder, as well as the dll and lib files you just compiled.

```cpp
#include "include\cuda_color_converter.h"

using namespace std;
using namespace cv;

#prgram comment(lib,"accel_vision.lib")

int main(void)
{
    string path = "2028.jpg";
    Mat img = imread(path), result;

    cudaGray(img, result); // gray this image

    string title = "CUDA";
    namedWindow(title);

    imshow(title, result);
    waitKey(0);

    return 0;
}
```

## Performance
I just test these api on my laptop(Thinkpad T420), which equips with **CPU I5-2520M** video card **Nvidia NVS 4200M**, that has only **1 SM, 48 CUDA cores, compute capability 2.1**.

Testing image is size 2048 * 2048.

API Name | Time Cost |
------ | ---------
cudaGray | 1.85ms(RGB to gray)
cudaHSV | 5.67ms(RGB to HSV)
cudaGaussianBlur | 3.81ms(5*5 kernel size, single channel)

More details see the excel.