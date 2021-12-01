#ifndef SHRMEM_BANDWIDTH_TEST_KERNEL_H
#define SHRMEM_BANDWIDTH_TEST_KERNEL_H

#include <cstddef>
#include <cstdint>

#define START 0
#define END 1
#define REPEATS 100000000

void shrMemBW(uint32_t *clocks, float* scratch, float magicNumber);
double getPeakGPUFrequency(int deviceId);
long long getNumTransferedBytes();
size_t getLaneSize();

#endif //SHRMEM_BANDWIDTH_TEST_KERNEL_H
