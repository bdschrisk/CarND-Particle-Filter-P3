project(PARTICLE_FILTER)

cmake_minimum_required (VERSION 3.5)

add_definitions(-std=c++0x)

# Build the particle filter project and solution.
# Use C++11
set(sources 
	src/main.cpp
	src/particle_filter.cpp)

# Create the executable
add_executable(particle_filter ${sources})

# Use C++11
#if [ ! -f ./src/particle_filter_sol.cpp]; then
#	echo "No solution file."
#else
#	echo "Solution file exists"
#fi

if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/src/particle_filter_sol.cpp")
	set(sources src/main.cpp src/particle_filter_sol.cpp)

	# Create the executable
	add_executable(particle_filter_solution ${sources})
endif()









