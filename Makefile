build:
	clang++ -dynamiclib -fpic ./cxx/custom_op_library.cc -o custom_op_library_cxx.dylib -I ./inc
