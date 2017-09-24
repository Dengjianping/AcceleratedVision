#ifndef TIME_UTILITY_H
#define TIME_UTILITY_H

#include "accel_vision.h"


CUDA_EXPORTS class TimeRecorder
{
public:
    virtual void startRecord() = 0;
    virtual void stopRecord() = 0;
    virtual float timeCost() = 0;
};


CUDA_EXPORTS class DeviceTimeRecorder : public TimeRecorder
{
private:
    cudaEvent_t start, end;
public:
    DeviceTimeRecorder();
    virtual void startRecord();
    virtual void stopRecord();
    virtual float timeCost(); // ms
    ~DeviceTimeRecorder();
};

DeviceTimeRecorder::DeviceTimeRecorder()
{
    cudaEventCreate(&start);
    cudaEventCreate(&end);
}

void DeviceTimeRecorder::startRecord()
{
    cudaEventRecord(start);
}

void DeviceTimeRecorder::stopRecord()
{
    cudaEventRecord(end);
    cudaEventSynchronize(start);
    cudaEventSynchronize(end);
}

float DeviceTimeRecorder::timeCost()
{
    float time;
    cudaEventElapsedTime(&time, start, end);
    return time;
}

DeviceTimeRecorder::~DeviceTimeRecorder()
{
    cudaEventDestroy(start);
    cudaEventDestroy(end);
}


CUDA_EXPORTS class HostTimeRecorder : public TimeRecorder
{
private:
    double start, end;
public:
    HostTimeRecorder();
    virtual void startRecord();
    virtual void stopRecord();
    virtual float timeCost();
    ~HostTimeRecorder();
};

HostTimeRecorder::HostTimeRecorder()
{
    start = end = 0.0f;
}

void HostTimeRecorder::startRecord()
{
    start = (float)cv::getTickCount();
}

void HostTimeRecorder::stopRecord()
{
    end = (float)cv::getTickCount();
}

float HostTimeRecorder::timeCost()
{
    return (end - start) / cv::getTickFrequency() * 1000; // ms
}

HostTimeRecorder::~HostTimeRecorder()
{}


CUDA_EXPORTS class AdvancedHostTimeRecorder : public TimeRecorder
{
private:
    std::chrono::time_point<std::chrono::system_clock> start, end;
public:
    AdvancedHostTimeRecorder();
    virtual void startRecord();
    virtual void stopRecord();
    virtual float timeCost();
    ~AdvancedHostTimeRecorder();
};

AdvancedHostTimeRecorder::AdvancedHostTimeRecorder()
{

}

void AdvancedHostTimeRecorder::startRecord()
{
    start = std::chrono::system_clock::now();
}

void AdvancedHostTimeRecorder::stopRecord()
{
    end = std::chrono::system_clock::now();
}


//float AdvancedHostTimeRecorder::timeCost()
//{
//    std::chrono::duration_cast<std::chrono::microseconds>
//    return (end - start) / cv::getTickFrequency() * 1000; // ms
//}
//

#endif // !TIME_UTILITY_H
