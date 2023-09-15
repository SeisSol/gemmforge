from gemmforge.instructions.builders.kernels.log.log_kernels import ShrMemBasedLoopOverGemmKernelBuilder
from gemmforge.instructions.builders.kernels.gemms.factory import GemmKernelType

class LoopOverGemmKernelsFactory:
  def __init__(self, **kwargs):
    self._kwargs = kwargs
    self._vm = kwargs['vm']
    self._hw_descr = self._vm.get_hw_descr()
    self._gemm_kernel_type = kwargs['gemm_kernel_type']

  def _auto_select(self):
    model = self._hw_descr.model
    if model == 'pvc':
      raise Exception("Register Only Loop Over Gemm Kernel is not yet implemented")
    else:
      return GemmKernelType.SHR_MEM_BASED

  def get_builder(self):
    if self._gemm_kernel_type == GemmKernelType.AUTO:
      self._gemm_kernel_type = self._auto_select()

    if self._gemm_kernel_type == GemmKernelType.SHR_MEM_BASED:
      return ShrMemBasedLoopOverGemmKernelBuilder(**self._kwargs)
    elif self._gemm_kernel_type == GemmKernelType.REGISTER_ONLY_BASED:
      raise Exception("Register Only Loop Over Gemm Kernel is not yet implemented")
    else:
      raise RuntimeError('unknown gemm type')

  def gemm_kernel_type(self):
    """returns a concrete kernel type. It is relevant if the user requested
    to perform auto-selection"""
    return self._gemm_kernel_type
