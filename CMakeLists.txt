# CMakeLists file for FBX viewer

cmake_minimum_required(VERSION 3.13 FATAL_ERROR)

# Some settings
set(BINARY_PATH "${CMAKE_CURRENT_SOURCE_DIR}/bin")
set(SCYTHE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/deps/scythe")
set(SCYTHE_THIRDPARTY_DIR "${CMAKE_CURRENT_SOURCE_DIR}/deps/thirdparty")

# Dependencies
add_subdirectory(deps)

# The project itself
project(FBXViewer)

set(CMAKE_CXX_STANDARD 14)
set(SRC_DIRS
	src
)
set(include_directories
	${SCYTHE_PATH}/include
	${SCYTHE_PATH}/src
	deps/ufbx
)
#set(defines )
set(libraries
	scythe
	ufbx
)

foreach(DIR ${SRC_DIRS})
	file(GLOB DIR_SOURCE ${CMAKE_CURRENT_SOURCE_DIR}/${DIR}/*.cpp)
	set(SRC_FILES ${SRC_FILES} ${DIR_SOURCE})
endforeach(DIR)

if (WIN32)
	add_executable(${PROJECT_NAME} WIN32 ${SRC_FILES})
else()
	add_executable(${PROJECT_NAME} ${SRC_FILES})
endif()
target_include_directories(${PROJECT_NAME} PRIVATE ${include_directories})
#target_compile_definitions(${PROJECT_NAME} PRIVATE ${defines})
target_link_libraries(${PROJECT_NAME} PRIVATE ${libraries})

install(TARGETS ${PROJECT_NAME}
		RUNTIME DESTINATION ${BINARY_PATH})