#include <iostream>
#include <CL/sycl.hpp>

namespace gemmgen {
    void checkErr(const std::string &File, int Line) {

    }

  void synchDevice() {
    cl::sycl::queue{cl::sycl::host_selector{}}.wait_and_throw();
  }
}


