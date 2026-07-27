# Configure the setup.py file
configure_file(setup.py.in ${CMAKE_CURRENT_BINARY_DIR}/setup.py)

# Create a custom target to build the python package during build stage
add_custom_target(autodiff-setuptools ALL
    DEPENDS autodiff4py
    COMMAND ${CMAKE_COMMAND} -E copy_directory ${CMAKE_CURRENT_SOURCE_DIR}/autodiff ${CMAKE_CURRENT_BINARY_DIR}/autodiff
    COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:autodiff4py> ${CMAKE_CURRENT_BINARY_DIR}/autodiff
    COMMAND ${PYTHON_EXECUTABLE} setup.py --quiet build --force
    COMMAND ${CMAKE_COMMAND} -E remove_directory ${CMAKE_CURRENT_BINARY_DIR}/autodiff/$<TARGET_FILE_NAME:autodiff4py>
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})

# Ensure the path where the python package is installed is not empty.
# Deliberately kept as a forward-slash CMake-style path (not converted via
# TO_NATIVE_PATH) since pip accepts forward slashes fine on Windows, and a
# native (backslash) path here would be embedded verbatim into the
# generated cmake_install.cmake below via install(CODE ...), which then
# fails to re-parse with "Invalid character escape" errors when that
# generated script is later run during the Install step.
if(NOT DEFINED AUTODIFF_PYTHON_INSTALL_PREFIX)
    set(AUTODIFF_PYTHON_INSTALL_PREFIX ${CMAKE_INSTALL_PREFIX})
endif()

# Create an install target for the autodiff python package
install(CODE "execute_process(COMMAND pip install . --no-deps --force-reinstall --no-compile --prefix=${AUTODIFF_PYTHON_INSTALL_PREFIX} WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})")
