# Dependencies
project(deps)

add_subdirectory(thirdparty)
add_subdirectory(scythe)

# -------------
# OpenFBX does not have CMake project
project(OpenFBX)

set(CMAKE_CXX_STANDARD 11)
set(SRC_DIRS OpenFBX/src)
set(include_directories OpenFBX/src)

foreach(DIR ${SRC_DIRS})
  file(GLOB DIR_SOURCE ${CMAKE_CURRENT_SOURCE_DIR}/${DIR}/*.c ${CMAKE_CURRENT_SOURCE_DIR}/${DIR}/*.cpp)
  set(SRC_FILES ${SRC_FILES} ${DIR_SOURCE})
endforeach(DIR)

add_library(${PROJECT_NAME} STATIC ${SRC_FILES})
target_include_directories(${PROJECT_NAME} PRIVATE ${include_directories})
#target_compile_definitions(${PROJECT_NAME} PRIVATE ${defines})
# -------------

project(ufbx)

set(SRC_DIRS ufbx)
set(include_directories ufbx)

foreach(DIR ${SRC_DIRS})
  file(GLOB DIR_SOURCE ${CMAKE_CURRENT_SOURCE_DIR}/${DIR}/*.c)
  set(SRC_FILES ${SRC_FILES} ${DIR_SOURCE})
endforeach(DIR)

add_library(${PROJECT_NAME} STATIC ${SRC_FILES})
target_include_directories(${PROJECT_NAME} PRIVATE ${include_directories})
#target_compile_definitions(${PROJECT_NAME} PRIVATE ${defines})

# -------------