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

#include "temp.h"
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
	//Part 1 - handle command line options such as device selection, verbosity, etc.
	int platformID = 0;
	int deviceID = 0;
	int sourceID = 0;

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
		cl::CommandQueue queue(context);

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

		//Part 3 - memory allocation
		std::vector<temp> data;
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

				temp temp(split[0], split[1] + split[2] + split[3] + split[4], std::stof(split[5]));

				data.push_back(temp);
			}

			dataFile.close();
		}
		else {
			std::cout << "Error reading data." << std::endl;
			_exit(0);
		}

		std::cout << "Min:\t" << 0 << "\tMax:\t" << 0 << std::endl;
		std::cout << "Mean:\t" << 0 << std::endl;
		std::cout << "Standard Deviation:\t" << 0 << std::endl;
		std::cout << "Median:\t" << 0 << "\t(25th: " << 0 << "\t75th: " << 0 << ")" << std::endl;
	}
	catch (cl::Error err) {
		std::cerr << "ERROR: " << err.what() << ", " << getErrorString(err.err()) << std::endl;
	}

	getchar();
	return 0;
}
