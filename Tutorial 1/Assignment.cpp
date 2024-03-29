#define CL_USE_DEPRECATED_OPENCL_1_2_APIS
#define __CL_ENABLE_EXCEPTIONS

#include <algorithm>
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

void print_help() {
	std::cerr << "Application usage:" << std::endl;

	std::cerr << "  -s : select data source " << std::endl;
	std::cerr << "  -p : select platform " << std::endl;
	std::cerr << "  -d : select device" << std::endl;
	std::cerr << "  -l : list all platforms and devices" << std::endl;
	std::cerr << "  -h : print this message" << std::endl;
}

int main(int argc, char **argv) {
	cl_float padVal = 999;

	//Part 1 - handle command line options such as device selection, verbosity, etc.
	int platformID = 0;
	int deviceID = 0;
	int sourceID = 1;

	cl_float min = 999.0;
	cl_float max = -999.0;
	cl_float mean = 0.0;
	cl_float stdDev = 0.0;
	cl_float med[3] = { 0.0, 0.0, 0.0 };

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
		std::vector<cl_float> input;
		std::ifstream dataFile;
		
		if (sourceID == 1) {
			dataFile.open("temp_lincolnshire.txt");
		}
		else {
			dataFile.open("temp_lincolnshire_short.txt");
			//dataFile.open("test.txt");
		}

		if (dataFile.good()) {
			std::string line;

			while (std::getline(dataFile, line)) {
				std::istringstream iss(line);

				std::vector<std::string> split((std::istream_iterator<std::string>{iss}),
												std::istream_iterator<std::string>());

				input.push_back(std::stof(split[5]));
			}

			dataFile.close();
		}
		else {
			std::cout << "Error reading data." << std::endl;
			_exit(0);
		}

		//Part 4 - Memory Allocation
		cl::Device device = context.getInfo<CL_CONTEXT_DEVICES>()[deviceID];

		cl::Kernel kernel = cl::Kernel(program, "sum");

		size_t localSize = kernel.getWorkGroupInfo<CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE>(device) * 8;
		size_t numRecords = input.size();
		size_t paddingSize = input.size() % localSize;

		if (paddingSize) {
			std::vector<cl_float> inputExtra((localSize - paddingSize) / 2, padVal);
			std::vector<cl_float> inputExtra2((localSize - paddingSize) / 2, -padVal);

			input.insert(input.end(), inputExtra.begin(), inputExtra.end());
			input.insert(input.begin(), inputExtra2.begin(), inputExtra2.end());
		}

		size_t inputElems = input.size();
		size_t inputSize = input.size() * sizeof(cl_float);
		size_t floatSize = sizeof(cl_float);
		size_t numGroups = inputElems / localSize;

		std::vector<cl_float> output(numGroups);
		std::vector<cl_float> outputVar(inputElems);

		size_t outputSize = output.size() * sizeof(cl_float);
		size_t outputVarSize = outputVar.size() * sizeof(cl_float);

		cl::Buffer inputBuffer(context, CL_MEM_READ_ONLY, inputSize);
		cl::Buffer floatBuffer(context, CL_MEM_READ_ONLY, floatSize);
		cl::Buffer outputBuffer(context, CL_MEM_READ_WRITE, outputSize);
		cl::Buffer outputVarBuffer(context, CL_MEM_READ_WRITE, outputVarSize);

		queue.enqueueWriteBuffer(inputBuffer, CL_TRUE, 0, inputSize, &input[0]);
		queue.enqueueFillBuffer(outputBuffer, 0, 0, outputSize);
		queue.enqueueFillBuffer(outputVarBuffer, 0, 0, outputVarSize);

		//Part 5 - Perform Device Operations and Calculations
		//Mean
		kernel.setArg(0, inputBuffer);
		kernel.setArg(1, outputBuffer);
		kernel.setArg(2, cl::Local(localSize * sizeof(cl_float)));

		queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(inputElems), cl::NDRange(localSize));
		queue.enqueueReadBuffer(outputBuffer, CL_TRUE, 0, outputSize, &output[0]);

		for (size_t i = 0; i < numGroups; i++) {
			mean += output[i];
		}
		mean /= numRecords;

		//Median & Quartiles
		kernel = cl::Kernel(program, "median");
		kernel.setArg(0, inputBuffer);
		kernel.setArg(1, outputVarBuffer);
		kernel.setArg(2, cl::Local(localSize * sizeof(cl_float)));

		queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(inputElems), cl::NDRange(localSize));
		queue.enqueueReadBuffer(outputVarBuffer, CL_TRUE, 0, outputVarSize, &outputVar[0]);

		std::vector<cl_float> sorted;

		std::vector<cl_float> pos(numGroups, 0);

		while (sorted.size() <= inputElems) {
			cl_float low = padVal;
			size_t lowIndex = padVal;

			for (size_t i = 0; i < pos.size(); i++) {
				if (pos[i] < localSize) {
					if (outputVar[(localSize * i) + pos[i]] < low) {
						low = outputVar[(localSize * i) + pos[i]];
						lowIndex = i;
					}
				}
			}

			sorted.push_back(low);
			pos[lowIndex]++;
		}

		med[0] = (sorted[(inputElems / 4) - 1] + sorted[(inputElems / 4)]) / 2;
		med[1] = (sorted[(inputElems / 2) - 1] + sorted[(inputElems / 2)]) / 2;
		med[2] = (sorted[((inputElems / 4) * 3) - 1] + sorted[(inputElems / 4) * 3]) / 2;

		//Min & Max
		for (size_t i = 0; i < sorted.size(); i++) {
			if (sorted[i] != -padVal) {
				min = sorted[i];
				break;
			}
		}

		for (size_t i = sorted.size() - 1; i >= 0; i--) {
			if (sorted[i] != padVal) {
				max = sorted[i];
				break;
			}
		}

		//Standard Deviation
		queue.enqueueWriteBuffer(floatBuffer, CL_TRUE, 0, floatSize, &mean);

		kernel = cl::Kernel(program, "variance");
		kernel.setArg(0, inputBuffer);
		kernel.setArg(1, outputVarBuffer);
		kernel.setArg(2, floatBuffer);

		queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(inputElems), cl::NDRange(localSize));
		queue.enqueueReadBuffer(outputVarBuffer, CL_TRUE, 0, outputVarSize, &outputVar[0]);

		queue.enqueueWriteBuffer(inputBuffer, CL_TRUE, 0, inputSize, &outputVar[0]);

		kernel = cl::Kernel(program, "sum");
		kernel.setArg(0, inputBuffer);
		kernel.setArg(1, outputBuffer);
		kernel.setArg(2, cl::Local(localSize * sizeof(cl_float)));

		queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(inputElems), cl::NDRange(localSize));
		queue.enqueueReadBuffer(outputBuffer, CL_TRUE, 0, outputSize, &output[0]);

		for (size_t i = 0; i < numGroups; i++) {
			stdDev += output[i];
		}
		stdDev /= numRecords;
		stdDev = sqrt(stdDev);

		//Part 7 - Output Results
		cout.precision(2);
		std::cout << std::fixed;
		std::cout << std::endl << "Total Number of Records: " << numRecords << std::endl;
		std::cout << std::endl << "Average: " << mean << std::endl;
		std::cout << "Min: " << min << "\t\tMax: " << max << std::endl;
		std::cout << "Standard Deviation: " << stdDev << std::endl;
		std::cout << "Median: " << med[1] << "\t\t(25th: " << med[0] << "\t75th: " << med[2] << ")" << std::endl;
	}
	catch (cl::Error err) {
		std::cerr << "ERROR: " << err.what() << ", " << getErrorString(err.err()) << std::endl;
	}

	getchar();
	return 0;
}
