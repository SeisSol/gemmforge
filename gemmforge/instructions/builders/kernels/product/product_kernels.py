from gemmforge.instructions.builders.abstract_builder import AbstractBuilder
from gemmforge.instructions.builders import ShrMemAllocBuilder
from gemmforge.basic_types import DataFlowDirection, ShrMemObject
from gemmforge.instructions.builders.abstract_builder import AbstractBuilder
from gemmforge.instructions.builders import GetElementPtrBuilder
from gemmforge.instructions.builders import RegistersAllocBuilder
from gemmforge.instructions import StoreRegToGlb
from gemmforge.instructions.builders.alloctor_builder import ShrMemNewAllocBuilder
from gemmforge.instructions.builders.product.product_builder import ShrMemBasedProductBuilder
from gemmforge.instructions.store import StoreRegToGlbTensor

class ShrMemBasedProductKernelBuilder(AbstractBuilder):
  """ This is the base class for building complete gemm kernels."""

  def __init__(self, **kwargs):
    super(ShrMemBasedProductKernelBuilder, self).__init__(kwargs['vm'], kwargs['symbol_table'])
    self._tensors = kwargs['tensors']
    self._alpha = kwargs['alpha']
    self._num_compute_threads = kwargs['num_compute_threads']
    self._num_active_threads = kwargs['num_active_threads']
    self._operation_descriptions = kwargs['operation_descriptions']

    self._reg_array_obj = None
    self._shr_mem_obj = None
    self._shr_mem_loads = []
    self._tmp_objects = list()

  def get_reg_array_obj(self):
    return self._reg_array_obj

  def get_shr_mem_obj(self):
    return self._shr_mem_obj

  def get_shr_mem_loads(self):
    return self._shr_mem_loads

  def build_prologue(self):
    builder = GetElementPtrBuilder(self._vm, self._symbol_table)
    print(", ".join([str(x) for x in self._symbol_table.from_global.values()]))
    #print(", ".join([str(x) for x in self._symbol_table]))
    #raise Exception("uwu")
    for symbol in self._symbol_table.from_global.values():
      print(symbol)
      builder.build(symbol)
      self._instructions.extend(builder.get_instructions())

    # create an array of registers
    builder = RegistersAllocBuilder(self._vm, self._symbol_table)
    print("WARNING: TODO: FIND A BETTER THREAD DISTRIBUTION FOR SUM OPERATOR")
    # Max line size of every tmp result or final result
    max_line_width = 0
    for tensor in self._tensors:
      if tensor.direction == DataFlowDirection.SINK or tensor.direction == DataFlowDirection.SOURCESINK:
        line_width = tensor.get_size() / tensor.get_dimensions()[0]
        if line_width > max_line_width:
          max_line_width = line_width

    builder.build(max_line_width, 0.0)
    self._instructions.extend(builder.get_instructions())
    self._reg_array_obj = builder.get_resultant_obj()

    for tensor in self._tensors:
      if tensor.direction == DataFlowDirection.SOURCESINK:
        builder = ShrMemNewAllocBuilder(self._vm, self._symbol_table)
        symbol = builder.build(tensor.name, tensor.get_volume(), tensor)
        self._instructions.extend(builder.get_instructions())
        #self._tmp_objects.append(builder.get_resultant_obj())

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

    builder.build(ops=[self._symbol_table[tensor] for tensor in self._tensors],
                  dest=self._symbol_table[self._reg_array_obj],
                  operation_descriptions=self._operation_descriptions)

    self._shr_mem_loads = builder.get_srh_mem_loads()
    self._instructions.extend(builder.get_instructions())

  def build_epilogue(self):
    # Max line size of every tmp result or final result
    result_tensor = None
    for tensor in self._tensors:
      if tensor.direction == DataFlowDirection.SINK:
        result_tensor = tensor
    assert(result_tensor != None)

    store = StoreRegToGlbTensor(self._vm,
                                self._symbol_table[result_tensor],
                                self._symbol_table[self._reg_array_obj],
                                self._num_compute_threads)
    self._instructions.append(store)


  def build(self):
    self.build_prologue()
    self.build_kernel()
    self.build_epilogue()