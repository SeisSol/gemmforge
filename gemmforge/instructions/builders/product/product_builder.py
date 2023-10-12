from typing import List

from gemmforge.basic_types import DataFlowDirection, GeneralLexicon
from gemmforge.instructions import ShrMemBasedProduct, SyncThreads
from gemmforge.instructions.builders.abstract_builder import AbstractBuilder
from gemmforge.instructions.builders.allocator_builder import ShrMemNewAllocBuilder
from gemmforge.instructions.loaders import shm_mem_loader_factory
from gemmforge.symbol_table import Symbol, SymbolType


class ShrMemBasedProductBuilder(AbstractBuilder):
  """This class helps to assemble all necessary instructions
  required to build a shared-memory-based dense gemm operation"""

  def __init__(self,
               vm,
               symbol_table,
               register_array,
               shr_mem,
               num_threads: int):
    super(ShrMemBasedProductBuilder, self).__init__(vm, symbol_table)
    self._dest_regs = register_array
    self._shr_mem = shr_mem
    self._num_threads = num_threads

    self._counter = 0
    self._load_instrs = []

    self._ops = list()
    self._dest = None

    self._mem_region_a = None
    self._mem_region_b = None

  def build(self,
            op1: Symbol,
            op2: Symbol,
            dest: Symbol,
            operation_description):
    self._reset()

    # Note: of trans_a==True than an operand is given as KxM instead of (MxK).
    # In this case, a loader will load an operand from glb. mem. to shr. mem
    # transposing it on the fly. In, short, the loader guaranties to deliver
    # an operand as (MxK) to shr. mem.
    # self._symbol_table.add_scope()
    print("EX1", ", ".join([str(op) for op in [op1, op2]]))
    if not op1.obj.temporary:
      print(op1)
      self._symbol_table.add_scope()
      self._op1 = self._make_loader_and_symbol(operand=op1, do_transpose=False)
    else:
      self._op1 = op2
    if not op2.obj.temporary:
      print(op1)
      self._symbol_table.add_scope()
      self._op2 = self._make_loader_and_symbol(operand=op2, do_transpose=False)
    else:
      self._op2 = op2
    print("EX2", ", ".join([str(op) for op in self._ops]))

    if not op1.obj.temporary and not op2.obj.temporary:
      self._insert_sync_threads()

    gemm_params = {'vm': self._vm,
                   'op1': self._op1,
                   'op2': self._op2,
                   'dest': dest,
                   'num_threads': self._num_threads,
                   'operation_description': operation_description}
    self._instructions.append(ShrMemBasedProduct(**gemm_params))

  def _make_loader_and_symbol(self, operand, do_transpose):
    shr_mem_region = Symbol(name=self._name_shr_reg(),
                            stype=SymbolType.SharedMem,
                            obj=operand.obj)

    self._symbol_table.add_symbol(shr_mem_region)
    load_op = shm_mem_loader_factory(vm=self._vm,
                                     dest=shr_mem_region,
                                     src=operand,
                                     shr_mem=self._shr_mem,
                                     num_threads=self._num_threads,
                                     load_and_transpose=do_transpose)

    self._instructions.append(load_op)
    self._load_instrs.append(load_op)
    return shr_mem_region

  def _make_allocator_and_symbol(self, name, operand):
    builder = ShrMemNewAllocBuilder(vm=self._vm, symbol_table=self._symbol_table)
    shr_mem_region = builder.build(name, operand.obj.get_volume())
    self._instructions.extend(builder.get_instructions())

    return shr_mem_region

  def get_srh_mem_loads(self):
    return self._load_instrs

  def _insert_sync_threads(self):
    self._instructions.append(SyncThreads(self._vm, self._num_threads))

  def _name_shr_reg(self):
    name = f'{GeneralLexicon.SHR_MEM_REGION_PREFIX}{self._counter}'
    self._counter += 1
    return name
