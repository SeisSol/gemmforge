from ..abstract_thread_policy import AbstractGemmLikeThreadPolicy, Architecture, DenseMatrix


class NvidiaGemmThreadPolicy(AbstractGemmLikeThreadPolicy):
  def __init__(self,
               arch: Architecture,
               reals_per_op: int,
               num_threads: int,
               bytes_per_real: int,
               op1: DenseMatrix,
               op2: DenseMatrix,
               res: DenseMatrix):
    super().__init__(arch, reals_per_op, num_threads, bytes_per_real, op1, op2, res)

  def _estimate_num_registers_per_mult(self, accumulator_length):
    # Note: derived experimentally
    factor = self._bytes_per_real / 4
    return factor * (32 + accumulator_length)

  def get_num_ops_per_block(self):

    accumulator_length = self._res.get_actual_num_cols()
    max_num_regs_per_thread = self._estimate_num_registers_per_mult(accumulator_length)

    shr_mem_bytes = self._reals_per_op * self._bytes_per_real
    mults_wrt_shr_mem = self._arch.max_local_mem_size_per_block / shr_mem_bytes
    mults_wrt_num_regs = self._arch.max_reg_per_block / (self._num_threads * max_num_regs_per_thread)
    mults_per_sm = int(min(mults_wrt_shr_mem, mults_wrt_num_regs))

    return max(int(mults_per_sm / self._arch.max_block_per_sm), 1)
