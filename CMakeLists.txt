cmake_minimum_required(VERSION 3.17)
set(CMAKE_CUDA_ARCHITECTURES 50)
project(cuda-algorithms LANGUAGES CXX CUDA)
cmake_policy(SET CMP0104 NEW)

# Enable CUDA
find_package(CUDA REQUIRED)
# Function to add CUDA executable for each .cu file in a subdirectory
function(add_cuda_executable target_name source_file)
    cuda_add_executable(${target_name} ${source_file})
    #set_target_properties(${target_name} PROPERTIES CUDA_ARCHITECTURES ${CMAKE_CUDA_ARCHITECTURES})
    #target_compile_features(${target_name} PUBLIC cuda_std_11)
endfunction()

# Get all subdirectories
file(GLOB SUBDIRS RELATIVE ${CMAKE_SOURCE_DIR} ${CMAKE_SOURCE_DIR}/*)

foreach(subdir ${SUBDIRS})
    if(IS_DIRECTORY ${CMAKE_SOURCE_DIR}/${subdir})
        # Get all .cu files in the subdirectory
        file(GLOB CU_FILES RELATIVE ${CMAKE_SOURCE_DIR}/${subdir} ${CMAKE_SOURCE_DIR}/${subdir}/*.cu)
        
        foreach(cu_file ${CU_FILES})
            # Extract the filename without extension
            get_filename_component(target_name ${cu_file} NAME_WE)
            # Add CUDA executable
            add_cuda_executable(${target_name} ${CMAKE_SOURCE_DIR}/${subdir}/${cu_file})
            # Set the output directory for the binary
            set_target_properties(${target_name} PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/${subdir}")
        endforeach()
    endif()
endforeach()
