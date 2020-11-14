enable_testing()

# Build a testing application
add_executable(checklinalg checklinalg.cc)
# Link it against the built library
target_link_libraries(checklinalg PUBLIC linalg)

add_test(NAME islinalg COMMAND checklinalg)
