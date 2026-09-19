# Lightweight dependency-free C++ tests: every test is a small executable that
# returns a non-zero exit code on failure and is registered with CTest.

function(aegisq_add_cpp_test name)
    add_executable(${name} ${name}.cpp)
    target_link_libraries(${name} PRIVATE aegisq_core)
    target_include_directories(${name} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
    add_test(NAME cpp.${name} COMMAND ${name})
endfunction()

aegisq_add_cpp_test(test_version)
