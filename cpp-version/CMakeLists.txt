cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(translator-ai-cpp)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Fetch SentencePiece
include(FetchContent)
FetchContent_Declare(
    sentencepiece
    GIT_REPOSITORY https://github.com/google/sentencepiece.git
    GIT_TAG        v0.2.0
)
FetchContent_MakeAvailable(sentencepiece)

# Find LibTorch
find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

# Source files
set(SOURCES
    src/main.cpp
    src/tokenizer.cpp
    src/data.cpp
    src/training.cpp
    src/inference.cpp
    src/e2b.cpp
    src/model/rope.cpp
    src/model/normalization.cpp
    src/model/attention.cpp
    src/model/block.cpp
    src/model/translator.cpp
)

# Add executable
add_executable(translator-ai-cpp ${SOURCES})

# Link libraries
target_link_libraries(translator-ai-cpp "${TORCH_LIBRARIES}" sentencepiece sentencepiece_train)

# Include directories (sentencepiece headers come from the FetchContent
# source/build trees; the sentencepiece targets do not export them)
target_include_directories(translator-ai-cpp PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${CMAKE_CURRENT_SOURCE_DIR}/src/model
    ${sentencepiece_SOURCE_DIR}/src
    ${sentencepiece_BINARY_DIR}/src
)

# Windows DLL copy
if (MSVC)
    file(GLOB TORCH_DLLS "${TORCH_INSTALL_PREFIX}/lib/*.dll")
    add_custom_command(TARGET translator-ai-cpp
        POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
        ${TORCH_DLLS}
        $<TARGET_FILE_DIR:translator-ai-cpp>)
endif()
