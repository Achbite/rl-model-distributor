cmake_minimum_required(VERSION 3.16)
project(maze_model_distributor LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenSSL REQUIRED)
find_package(Protobuf REQUIRED)
find_package(PkgConfig REQUIRED)
pkg_check_modules(GRPC REQUIRED grpc++)

set(CONTRACT_CPP_DIR "" CACHE PATH "Directory containing generated C++ bindings")
if(NOT CONTRACT_CPP_DIR)
    message(FATAL_ERROR "CONTRACT_CPP_DIR must point to an rl-contracts C++ artifact")
endif()

set(PROTO_SOURCES
    ${CONTRACT_CPP_DIR}/common.pb.cc
    ${CONTRACT_CPP_DIR}/training.pb.cc
    ${CONTRACT_CPP_DIR}/training.grpc.pb.cc
)

set(SOURCES
    main/main.cpp
    src/config/config_loader.cpp
    src/grpc/model_distributor_service.cpp
    src/store/model_registry.cpp
)

add_executable(${PROJECT_NAME} ${SOURCES} ${PROTO_SOURCES})
target_include_directories(${PROJECT_NAME} PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${CONTRACT_CPP_DIR}
    ${Protobuf_INCLUDE_DIRS}
    ${GRPC_INCLUDE_DIRS}
)
target_link_libraries(${PROJECT_NAME} PRIVATE
    OpenSSL::Crypto
    protobuf::libprotobuf
    ${GRPC_LIBRARIES}
    pthread
)

include(CTest)
if(BUILD_TESTING)
    add_executable(model_distributor_development_test
        tests/model_distributor_rpc_test.cpp
        src/config/config_loader.cpp
        src/grpc/model_distributor_service.cpp
        src/store/model_registry.cpp
        ${PROTO_SOURCES}
    )
    target_include_directories(model_distributor_development_test PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
        ${CONTRACT_CPP_DIR}
        ${Protobuf_INCLUDE_DIRS}
        ${GRPC_INCLUDE_DIRS}
    )
    target_link_libraries(model_distributor_development_test PRIVATE
        OpenSSL::Crypto
        protobuf::libprotobuf
        ${GRPC_LIBRARIES}
        pthread
    )
    add_test(
        NAME model_distributor_development_contract
        COMMAND model_distributor_development_test
    )
endif()
