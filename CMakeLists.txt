cmake_minimum_required(VERSION 3.10)

project(DelayEstimator)

find_package(Boost COMPONENTS program_options REQUIRED)
find_package(UHD REQUIRED UHD)

if(Boost_FOUND)
  message(STATUS "Boost found!")
  message(STATUS "Boost version: ${Boost_VERSION}")
  message(STATUS "Boost_INCLUDE_DIRS: ${Boost_INCLUDE_DIRS}")
  message(STATUS "Boost_LIBRARIES: ${Boost_LIBRARIES}")

  message(STATUS "UHD found!")
  message(STATUS "UHD version: ${UHD_VERSION}")
  message(STATUS "UHD_INCLUDE_DIRS: ${UHD_INCLUDE_DIRS}")
  message(STATUS "UHD_LIBRARIES: ${UHD_LIBRARIES}")

  add_executable(trx_timed_samples trx_timed_samples.cpp)

  target_link_libraries(trx_timed_samples PRIVATE ${UHD_LIBRARIES} ${Boost_LIBRARIES})
  target_include_directories(trx_timed_samples PRIVATE ${UHD_INCLUDE_DIRS} ${Boost_INCLUDE_DIRS})
else()
  message(FATAL_ERROR "Boost not found! Please install Boost.")
endif()
