#include <cstdio>
#include <ctime>
#include <csignal>
#include </usr/local/cuda-11.4/include/nvml.h>
#include <unistd.h>
#include <sys/time.h>
#include <sys/timeb.h>
//#include<iostream>
int DO = 1;

void signalHandler(int signum)
{
    //std::printf("aaaaa");
    DO = 0;
    std::fflush(stdout);
}

int main(int argc, char* argv[])
{
    DO = 1;
    std::signal(SIGINT,signalHandler);
    setvbuf(stdout,NULL,_IONBF,0);
    
    //std::ios_base::sync_with_stdio(false);
    //std::cin.tie(NULL);
    //std::cout.tie(NULL);

    nvmlReturn_t result;
    nvmlReturn_t memory_result;
    unsigned int device_count;
    struct tm curr_tm;
    time_t curr_time;
    struct timeb timer_msec;
    struct timeval start, fin;
    long operating_time;

    //localtime_r(&curr_time, &curr_tm);

    result = nvmlInit();
    if (result != NVML_SUCCESS)
        return 1;
    
    result = nvmlDeviceGetCount(&device_count);
    if (result != NVML_SUCCESS)
        return 2;

    while(DO){
	gettimeofday(&start, NULL);
	curr_time = time(NULL);
	localtime_r(&curr_time, &curr_tm);
	ftime(&timer_msec);
	for (int i = 0; i < device_count; ++i) {
        	nvmlDevice_t device;
      		result = nvmlDeviceGetHandleByIndex_v2(i, &device);
            //std::printf("1\n");
        	if (result != NVML_SUCCESS)
            	return 3;

        	char device_name[NVML_DEVICE_NAME_BUFFER_SIZE];
        	result = nvmlDeviceGetName(device, device_name, NVML_DEVICE_NAME_BUFFER_SIZE);
        	//std::printf("2\n");
            if (result != NVML_SUCCESS)
            		return 4;
		std::printf("%d:%d:%d:%d  ", curr_tm.tm_hour, curr_tm.tm_min, curr_tm.tm_sec,timer_msec.millitm);
        	std::printf("Device %d: %s  ", i, device_name);

        	nvmlUtilization_st device_utilization;
        	result = nvmlDeviceGetUtilizationRates(device, &device_utilization);
            //std::printf("3\n");
            
        nvmlMemory_t device_memory;
		memory_result = nvmlDeviceGetMemoryInfo(device, &device_memory);
            //std::printf("4\n");
            
        	if (result != NVML_SUCCESS && memory_result != NVML_SUCCESS)
            		return 5;

        	std::printf("GPU Util: %u  Mem Util: %u Mem Usage: %lli\n ", device_utilization.gpu, device_utilization.memory, device_memory.used);		
   	}
        //std::fflush(stdout);
	gettimeofday(&fin, NULL);
	operating_time = fin.tv_usec - start.tv_usec;
    //printf("%ld\n",operating_time);
	long sleeptime = 166667 - operating_time;
    sleeptime = (sleeptime>0?sleeptime:0);
    usleep(sleeptime);
    //std::printf("time : %ld\n",sleeptime);
    }
    
    nvmlShutdown();
    return 0;
}

