from gemmforge.basic_types import DataFlowDirection
from gemmforge.instructions.builders import GetElementPtrBuilder, RegistersAllocBuilder, ShrMemAllocBuilder
from gemmforge.instructions.builders.alloctor_builder import ShrMemNewAllocBuilder
from gemmforge.instructions.builders.gemms.gemm_builder import ShrMemBasedDenseGemmBuilder
from gemmforge.instructions.builders.kernels.gemms.base_kernel import BaseGemmKernelBuilder
from gemmforge.symbol_table import SymbolType


class ShrMemBasedLoopOverGemmKernelBuilder(BaseGemmKernelBuilder):
  """ This is the base class for building complete gemm kernels."""

  def __init__(self, **kwargs):
    super(ShrMemBasedLoopOverGemmKernelBuilder, self).__init__(**kwargs)

  def build_kernel(self):
    builder = ShrMemAllocBuilder(self._vm, self._symbol_table)
    builder.build(size=None)
    self._instructions.extend(builder.get_instructions())
    self._shr_mem_obj = builder.get_resultant_obj()

    # generate the rest instructions i.e., load to shr. mem, compute, store
    builder = ShrMemBasedDenseGemmBuilder(self._vm,
                                          self._symbol_table,
                                          self._reg_array_obj,
                                          self._shr_mem_obj,
                                          self._num_active_threads)

    builder.build(trans_a=self._trans_a,
                  trans_b=self._trans_b,
                  op1=self._symbol_table[self._mat_a],
                  op2=self._symbol_table[self._mat_b],
                  dest=self._symbol_table[self._reg_array_obj])

    self._shr_mem_loads = builder.get_srh_mem_loads()
    self._instructions.extend(builder.get_instructions())

  def get_reg_array_obj(self):
    return self._reg_array_obj

  def get_shr_mem_obj(self):
    return self._shr_mem_obj

  def get_shr_mem_loads(self):
    return self._shr_mem_loads

  def build_prologue(self):
    builder = GetElementPtrBuilder(self._vm, self._symbol_table)
    print(", ".join([str(x) for x in self._symbol_table.from_global.values()]))

    for symbol in self._symbol_table.from_global.values():
      if symbol.stype == SymbolType.Batch:
        print(symbol)
        # try:
        builder.build(symbol)
        self._instructions.extend(builder.get_instructions())
        # except:
        #  pass

    # create an array of registers
    builder = RegistersAllocBuilder(self._vm, self._symbol_table)
    builder.build(self._mat_c.get_actual_num_cols(), 0.0)

    self._instructions.extend(builder.get_instructions())
    self._reg_array_obj = builder.get_resultant_obj()

    for matrix in [self._mat_a, self._mat_b, self._mat_c]:
      if matrix.direction == DataFlowDirection.SOURCESINK and "tmp" in matrix.name:
        builder = ShrMemNewAllocBuilder(self._vm, self._symbol_table)
        symbol = builder.build(matrix.name, matrix.get_actual_volume(), matrix)
        self._instructions.extend(builder.get_instructions())
        # self._tmp_objects.append(builder.get_resultant_obj())

  def build(self):
    self.build_prologue()
    self.build_kernel()
    self.build_epilogue()
