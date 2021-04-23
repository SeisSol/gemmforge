from ..abstract_thread_policy import AbstractUniOpThreadPolicy, Architecture, DenseMatrix


class NvidiaInitializerThreadPolicy(AbstractUniOpThreadPolicy):
  def __init__(self,
               arch: Architecture,
               num_threads: int,
               op1: DenseMatrix):
    super().__init__(arch, num_threads, op1)


  def get_num_ops_per_block(self):
   total_num_threas_per_op = self._num_threads * self._op1.get_actual_num_cols()
   max_num_regs_per_thread = 10 # Note: derived experimentally
   mults_wrt_num_regs = self._arch.max_reg_per_block / (total_num_threas_per_op * max_num_regs_per_thread)
   return max(int(mults_wrt_num_regs / self._arch.max_block_per_sm), 1)
