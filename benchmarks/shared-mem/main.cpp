#include "kernel.h"
#include <device.h>
#include <iostream>
#include <vector>

using namespace device;

int main() {
  constexpr int deviceId{0};

  DeviceInstance &device = DeviceInstance::getInstance();
  device.api->setDevice(deviceId);
  device.api->initialize();

  size_t laneSize = getLaneSize();
  auto devClocks = static_cast<uint32_t*>(device.api->allocGlobMem(2 * sizeof(uint32_t)));
  auto devScratch = static_cast<float*>(device.api->allocGlobMem(laneSize * sizeof(float)));


  float magicNumber{3.14};
  shrMemBW(devClocks, devScratch, magicNumber);

  std::vector<uint32_t> clocks(2, 0);
  device.api->copyFrom(const_cast<uint32_t*>(clocks.data()), devClocks, 2 * sizeof(uint32_t));

  std::vector<float> scratch(laneSize, 0.0f);
  device.api->copyFrom(const_cast<float*>(scratch.data()), devScratch, laneSize * sizeof(float));
  for (size_t i{0}; i < scratch.size(); ++i) {
    if (scratch[i] != magicNumber) {
      std::cout << "incorrect results at(" << i << "); expected: "
                << magicNumber << ", receivec: " << scratch[i]
                << std::endl;
    }
  }

  auto peakFrequency = getPeakGPUFrequency(deviceId);
  std::cout << "[clock] peak frequency, Hz: " << peakFrequency << std::endl;

  size_t clockDiff = clocks[END] - clocks[START];
  std::cout << "[clock] start: " << clocks[START] << std::endl;
  std::cout << "[clock] end: " << clocks[END] << std::endl;
  std::cout << "[clock] diff: " << clockDiff << std::endl;

  double time = static_cast<double>(clockDiff) / peakFrequency;
  std::cout << "time, sec: " << time << std::endl;

  constexpr double tera{1024.0 * 1024 * 1024 * 1024};
  double teraBytes = static_cast<double>(getNumTransferedBytes()) / tera;
  std::cout << "teraBytes: " << teraBytes << std::endl;
  double bandwidth = teraBytes / time;
  std::cout << "bandwidth, TB/s: " << bandwidth << std::endl;

  device.api->freeMem(devClocks);
  device.api->freeMem(devScratch);
  device.api->finalize();
  return 0;
}
