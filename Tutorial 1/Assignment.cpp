#define CL_USE_DEPRECATED_OPENCL_1_2_APIS
#define __CL_ENABLE_EXCEPTIONS

#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#ifdef __APPLE__
#include <OpenCL/cl.hpp>
#else
#include <CL/cl.hpp>
#endif

#include "Utils.h"

typedef struct temp {
	char location[14];
	char date[12];
	float value;
} temp_t;

void print_help() {
	std::cerr << "Application usage:" << std::endl;

	std::cerr << "  -s : select data source " << std::endl;
	std::cerr << "  -p : select platform " << std::endl;
	std::cerr << "  -d : select device" << std::endl;
	std::cerr << "  -l : list all platforms and devices" << std::endl;
	std::cerr << "  -h : print this message" << std::endl;
}

int main(int argc, char **argv) {
	//Part 1 - handle command line options such as device selection, verbosity, etc.
	int platformID = 0;
	int deviceID = 0;
	int sourceID = 0;

	int min = 0;
	int max = 0;
	int mean = 0;
	int stdDev = 0;
	int med[3] = { 0, 0, 0 };

	for (int i = 1; i < argc; i++)	{
		if ((strcmp(argv[i], "-p") == 0) && (i < (argc - 1))) { platformID = atoi(argv[++i]); }
		else if ((strcmp(argv[i], "-d") == 0) && (i < (argc - 1))) { deviceID = atoi(argv[++i]); }
		else if ((strcmp(argv[i], "-s") == 0) && (i < (argc - 1))) { sourceID = atoi(argv[++i]); }
		else if (strcmp(argv[i], "-l") == 0) { std::cout << ListPlatformsDevices() << std::endl; }
		else if (strcmp(argv[i], "-h") == 0) { print_help(); return 0;}
	}

	//detect any potential exceptions
	try {
		//Part 2 - host operations
		//2.1 Select computing devices
		cl::Context context = GetContext(platformID, deviceID);

		//display the selected device
		std::cout << "Running on " << GetPlatformName(platformID) << ", " << GetDeviceName(platformID, deviceID) << std::endl;

		//create a queue to which we will push commands for the device
		cl::CommandQueue queue(context, CL_QUEUE_PROFILING_ENABLE);

		//2.2 Load & build the device code
		cl::Program::Sources sources;

		AddSources(sources, "kernels.cl");

		cl::Program program(context, sources);

		//build and debug the kernel code
		try {
			program.build();
		}
		catch (const cl::Error& err) {
			std::cout << "Build Status: " << program.getBuildInfo<CL_PROGRAM_BUILD_STATUS>(context.getInfo<CL_CONTEXT_DEVICES>()[0]) << std::endl;
			std::cout << "Build Options:\t" << program.getBuildInfo<CL_PROGRAM_BUILD_OPTIONS>(context.getInfo<CL_CONTEXT_DEVICES>()[0]) << std::endl;
			std::cout << "Build Log:\t " << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(context.getInfo<CL_CONTEXT_DEVICES>()[0]) << std::endl;
			throw err;
		}

		//Part 3 - Read Data
		std::vector<temp> input;
		std::ifstream dataFile("temp_lincolnshire_short.txt");
		
		if (sourceID == 1) {
			std::ifstream dataFile("temp_lincolnshire.txt");
		}

		if (dataFile.good()) {
			std::string line;

			while (std::getline(dataFile, line)) {
				std::istringstream iss(line);

				std::vector<std::string> split((std::istream_iterator<std::string>{iss}),
												std::istream_iterator<std::string>());

				temp temp;
				strcpy(temp.location, split[0].c_str());
				strcpy(temp.date, (split[1] + split[2] + split[3] + split[4]).c_str());
				temp.value = std::stof(split[5]);

				input.push_back(temp);
			}

			dataFile.close();
		}
		else {
			std::cout << "Error reading data." << std::endl;
			_exit(0);
		}

		//Part 4 - Memory Allocation
		size_t inputElems = input.size();
		size_t inputSize = input.size() * sizeof(temp);

		std::vector<temp> output(inputElems);
		size_t outputSize = output.size() * sizeof(temp);

		cl::Buffer inputBuffer(context, CL_MEM_READ_WRITE, inputSize);
		cl::Buffer outputBuffer(context, CL_MEM_READ_WRITE, outputSize);

		//Part 5 - Device Operations

		queue.enqueueWriteBuffer(inputBuffer, CL_TRUE, 0, inputSize, &input[0]);
		
		cl::Kernel kernelIdentity = cl::Kernel(program, "identity");
		kernelIdentity.setArg(0, inputBuffer);
		kernelIdentity.setArg(1, outputBuffer);

		cl::Event profiler;

		cl::Device device = context.getInfo<CL_CONTEXT_DEVICES>()[deviceID];
		size_t prefWorkSize = kernelIdentity.getWorkGroupInfo<CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE>(device);

		queue.enqueueNDRangeKernel(kernelIdentity, cl::NullRange, cl::NDRange(inputElems), cl::NullRange);

		queue.enqueueReadBuffer(outputBuffer, CL_TRUE, 0, outputSize, &output[0]);

		//Part 6 - Output Results

		for (temp t : output) {
			std::cout << t.location << "\t" << t.date << "\t" << t.value << std::endl;
		}

		std::cout << std::endl;

		std::cout << "Total Number of Records:\t" << input.size() << std::endl;
		std::cout << "Min:\t" << min << "\tMax:\t" << max << std::endl;
		std::cout << "Mean:\t" << mean << std::endl;
		std::cout << "Standard Deviation:\t" << stdDev << std::endl;
		std::cout << "Median:\t" << med[1] << "\t(25th: " << med[0] << "\t75th: " << med[2] << ")" << std::endl;
	}
	catch (cl::Error err) {
		std::cerr << "ERROR: " << err.what() << ", " << getErrorString(err.err()) << std::endl;
	}

	getchar();
	return 0;
}
