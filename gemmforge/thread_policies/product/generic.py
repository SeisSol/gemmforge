from typing import List

from gemmforge.tensor.dense import DenseTensor
from gemmforge.vm import VM
from ..abstract_thread_policy import AbstractTensorThreadPolicy


class GenericProductThreadPolicy(AbstractTensorThreadPolicy):
  def __init__(self,
               vm: VM,
               shr_mem_per_op: int,
               num_threads: int,
               ops: List[DenseTensor],
               res: DenseTensor):
    super().__init__(vm, num_threads, ops, res)
    self._shr_mem_per_op = shr_mem_per_op

  def _estimate_num_registers_per_mult(self, accumulator_length):
    # Note: derived experimentally
    print("WARNING: TODO: IMPROVE GENERIC PRODUCT THREAD POLICY 1")
    factor = self._vm.bytes_per_real() / 4
    return factor * (32 + accumulator_length)

  def get_num_ops_per_block(self):
    print("WARNING: TODO: IMPROVE GENERIC PRODUCT THREAD POLICY 2")
    accumulator_length = int(self._res.get_size() / self._res.get_dimensions()[0])
    max_num_regs_per_thread = self._estimate_num_registers_per_mult(accumulator_length)

    hw_descr = self._vm.get_hw_descr()
    shr_mem_bytes = self._shr_mem_per_op * self._vm.bytes_per_real()
    if shr_mem_bytes == 0:
      shr_mem_bytes = 1
      print("WARNING: THREAED POLICY SHR_MEM_BYTES IS 0")
    mults_wrt_shr_mem = hw_descr.max_local_mem_size_per_block / shr_mem_bytes
    mults_wrt_num_regs = hw_descr.max_reg_per_block / (self._num_threads * max_num_regs_per_thread)
    mults_per_sm = int(min(mults_wrt_shr_mem, mults_wrt_num_regs))

    return int(max(int(mults_per_sm / hw_descr.max_block_per_sm), 1))
