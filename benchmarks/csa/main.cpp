#include "gen_code/kernels.h"
#include "stop_watch.h"
#include "gemmforge_aux.h"
#include "yaml-cpp/yaml.h"
#include <device.h>
#include <iostream>
#include <tuple>
#include <vector>
#include <string>

using namespace gemmforge;
using namespace device;

int estimateNumElements(int SizeA, int SizeB, double AllowedSpaceInGB);

int main(int Argc, char* Arcv[]) {
  YAML::Node Params = YAML::LoadFile("./params.yaml");
  YAML::Node MatrixASpec = Params["MatA"];
  YAML::Node MatrixBSpec = Params["MatB"];

  int SizeA = MatrixASpec["num_rows"].as<int>() * MatrixASpec["num_cols"].as<int>();
  int SizeB = MatrixBSpec["num_rows"].as<int>() * MatrixBSpec["num_cols"].as<int>();
  
  std::vector<int> BboxA = MatrixASpec["bbox"].as<std::vector<int>>();
  std::vector<int> BboxB = MatrixBSpec["bbox"].as<std::vector<int>>();
  
  int M = BboxA[2] - BboxA[0];
  int N = BboxA[3] - BboxA[1];
  
  real Alpha = Params["alpha"].as<real>();
  real Beta = Params["beta"].as<real>();

  YAML::Node Config = YAML::LoadFile("./config.yaml");
  int NumElements = estimateNumElements(SizeA, SizeB, Config["allocate_mem"].as<double>());

  
  int additionOp{0};
  if ((Alpha != 0.0) && (Beta != 0.0)) additionOp = 1;
  
  int scaleAOp{0};
  if ((Alpha != 0.0) && (Alpha != 1.0)) scaleAOp = 1;
  
  int scaleBOp{0};
  if ((Beta != 0.0) && (Beta != 1.0)) scaleAOp = 1;
  
  int threadOperations = additionOp + scaleAOp + scaleBOp;
  long long FlopCounter = threadOperations * M * N;

  DeviceInstance &device = DeviceInstance::getInstance();
  device.api->setDevice(0);
  device.api->initialize();
  
  auto* DevA = static_cast<real*>(device.api->allocGlobMem(SizeA * NumElements * sizeof(real)));
  auto* DevB = static_cast<real*>(device.api->allocGlobMem(SizeB * NumElements * sizeof(real)));

  // Measure performance
  utils::StopWatch<std::chrono::duration<double, std::chrono::nanoseconds::period>> Timer;
  int NumRepeats = Config["num_repeats"].as<int>();
  Timer.start();
  for (int Repeat = 0; Repeat < NumRepeats; ++Repeat) {
    csa(Alpha, DevA, 0, DevB, 0, NumElements, nullptr, device.api->getDefaultStream());
  }
  device.api->synchDevice();
  Timer.stop();

  std::cout << "Num elements: " << NumElements << std::endl;
  std::cout << "Num repeats: " << NumRepeats << std::endl;
  std::cout << "Computed Flops: " << NumRepeats * FlopCounter * NumElements << std::endl;
  std::cout << "Spent time: " << Timer.getTime() << std::endl;
  std::cout << "GFLOPS: " << NumRepeats * FlopCounter / (Timer.getTime() / NumElements) << std::endl;

  device.api->freeMem(DevA);
  device.api->freeMem(DevB);

  device.api->finalize();
  return 0;
}


int estimateNumElements(int SizeA, int SizeB, double AllowedSpaceInGB) {
  long long ElementSizeInBytes = (SizeA + SizeB) * sizeof(real);
  constexpr double FACTOR = 1024 * 1024 * 1024;
  return int((AllowedSpaceInGB * FACTOR) / ElementSizeInBytes);
}