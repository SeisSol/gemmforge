from gemmforge.instructions.builders.abstract_builder import AbstractBuilder
from gemmforge.instructions.builders import ShrMemAllocBuilder
from gemmforge.basic_types import ShrMemObject
from gemmforge.instructions.builders.abstract_builder import AbstractBuilder
from gemmforge.instructions.builders import GetElementPtrBuilder
from gemmforge.instructions.builders import RegistersAllocBuilder
from gemmforge.instructions import StoreRegToGlb
from gemmforge.instructions.builders.product.product_builder import ShrMemBasedProductBuilder
from gemmforge.instructions.store import StoreRegToGlbTensor

class ShrMemBasedProductKernelBuilder(AbstractBuilder):
  """ This is the base class for building complete gemm kernels."""

  def __init__(self, **kwargs):
    super(ShrMemBasedProductKernelBuilder, self).__init__(kwargs['vm'], kwargs['symbol_table'])
    self._trans_a = kwargs['trans_a']
    self._trans_b = kwargs['trans_b']
    self._tensor_a = kwargs['tensor_a']
    self._tensor_b = kwargs['tensor_b']
    self._tensor_c = kwargs['tensor_c']
    self._alpha = kwargs['alpha']
    self._num_compute_threads = kwargs['num_compute_threads']
    self._num_active_threads = kwargs['num_active_threads']

    self._reg_array_obj = None
    self._shr_mem_obj = None
    self._shr_mem_loads = []

  def get_reg_array_obj(self):
    return self._reg_array_obj

  def get_shr_mem_obj(self):
    return self._shr_mem_obj

  def get_shr_mem_loads(self):
    return self._shr_mem_loads

  def build_prologue(self):
    builder = GetElementPtrBuilder(self._vm, self._symbol_table)
    for symbol in self._symbol_table.from_global.values():
      builder.build(symbol)
      self._instructions.extend(builder.get_instructions())

    # create an array of registers
    builder = RegistersAllocBuilder(self._vm, self._symbol_table)
    print("WARNING: TODO: FIND A BETTER THREAD DISTRIBUTION FOR SUM OPERATOR")
    builder.build(self._tensor_c.get_size() / self._tensor_c.get_dimensions()[0], 0.0)
    self._instructions.extend(builder.get_instructions())
    self._reg_array_obj = builder.get_resultant_obj()

  def build_kernel(self):
    builder = ShrMemAllocBuilder(self._vm, self._symbol_table)
    builder.build(size=None)
    self._instructions.extend(builder.get_instructions())
    self._shr_mem_obj = builder.get_resultant_obj()

    # generate the rest instructions i.e., load to shr. mem, compute, store
    builder = ShrMemBasedProductBuilder(self._vm,
                                          self._symbol_table,
                                          self._reg_array_obj,
                                          self._shr_mem_obj,
                                          self._num_active_threads)

    builder.build(trans_a=self._trans_a,
                  trans_b=self._trans_b,
                  op1=self._symbol_table[self._tensor_a],
                  op2=self._symbol_table[self._tensor_b],
                  dest=self._symbol_table[self._reg_array_obj])

    self._shr_mem_loads = builder.get_srh_mem_loads()
    self._instructions.extend(builder.get_instructions())

  def build_epilogue(self):
    store = StoreRegToGlbTensor(self._vm,
                                self._symbol_table[self._tensor_c],
                                self._symbol_table[self._reg_array_obj],
                                self._alpha,
                                1.0,
                                self._num_compute_threads)
    self._instructions.append(store)


  def build(self):
    self.build_prologue()
    self.build_kernel()
    self.build_epilogue()