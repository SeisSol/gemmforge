from gemmforge.instructions.builders.kernels.product.product_kernels import ShrMemBasedProductKernelBuilder
from gemmforge.instructions.builders.kernels.gemms.factory import GemmKernelType


class ProductKernelsFactory:
  def __init__(self, **kwargs):
    self._kwargs = kwargs
    self._vm = kwargs['vm']
    self._hw_descr = self._vm.get_hw_descr()
    self._product_kernel_type = kwargs['product_kernel_type']

  def _auto_select(self):
    model = self._hw_descr.model
    if model == 'pvc':
      raise Exception("Register Only Product Kernel is not yet implemented")
      # return GemmKernelType.REGISTER_ONLY_BASED
    else:
      return GemmKernelType.SHR_MEM_BASED

  def get_builder(self):
    if self._product_kernel_type == GemmKernelType.AUTO:
      self._product_kernel_type = self._auto_select()

    if self._product_kernel_type == GemmKernelType.SHR_MEM_BASED:
      return ShrMemBasedProductKernelBuilder(**self._kwargs)
    elif self._product_kernel_type == GemmKernelType.REGISTER_ONLY_BASED:
      raise Exception("Register Only Product Kernel is not yet implemented")
      # return RegisterOnlyProductKernelBuilder(**self._kwargs)
    else:
      raise RuntimeError('unknown gemm type')

  def product_kernel_type(self):
    """returns a concrete kernel type. It is relevant if the user requested
    to perform auto-selection"""
    return self._product_kernel_type
