#pragma once

#include <string>

class temp
{
public:
	temp(std::string location, std::string date, float value);

	std::string location;
	std::string date;
	float value;
};

temp::temp(std::string location, std::string date, float value)
{
	this->location = location;
	this->date = date;
	this->value = value;
}
