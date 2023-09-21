add_executable(logistic_training logistic_training.cpp)
add_executable(training_all_labels training_all_labels.cpp)


find_package(Boost COMPONENTS json log program_options REQUIRED)

target_compile_features(logistic_training PRIVATE cxx_std_20)
target_compile_features(training_all_labels PRIVATE cxx_std_20)

target_link_libraries(logistic_training
    MOTION::motion
    Boost::json
    Boost::log
    Boost::program_options
)

target_link_libraries(training_all_labels
    MOTION::motion
    Boost::json
    Boost::log
    Boost::program_options
)


