from gemmforge.symbol_table import DataView
from .abstract_loader import AbstractShrMemLoader


def _find_next_prime(number):
  factor = 2
  return _find_prime_in_range(number, factor * number)


def _find_prime_in_range(source, target):
  for number in range(source, target):
    for i in range(2, number):
      if number % i == 0:
        break
    else:
      return number


class ExtendedTransposePatchLoader(AbstractShrMemLoader):
  def __init__(self, **kwargs):
    super(ExtendedTransposePatchLoader, self).__init__(**kwargs)

    data_view = self._src.data_view
    optimal_num_cols = _find_next_prime(data_view.columns)
    self._shm_volume = data_view.lead_dim * optimal_num_cols

    self._dest.data_view = DataView(rows=data_view.columns,
                                    columns=data_view.rows,
                                    lead_dim=optimal_num_cols,
                                    is_transposed=True)

  def gen_code(self, writer):
    super(ExtendedTransposePatchLoader, self).gen_code(writer)
    writer("// using ExtendedTransposePatchLoader")

    src_data_view = self._src.data_view
    dest_data_view = self._dest.data_view
    thread_idx_x = self._lexic.thread_idx_x
    num_hops = int(self._shm_volume / self._num_threads)
    index_var = 'index'

    with writer.Scope():

      writer(f'int {index_var};')
      if num_hops > 0:
        if num_hops > self._manual_unroll_threshold:
          # load using a for-loop

          writer.Pragma('unroll')
          with writer.For(f'int i = 0; i < {num_hops}; ++i'):
            pass
            writer(f'{index_var} = {thread_idx_x} + i * {self._num_threads};')
            shr_mem_addr = f'({index_var} % {src_data_view.lead_dim}) * {dest_data_view.lead_dim}'
            shr_mem_addr += f' + {index_var} / {src_data_view.lead_dim}'

            glb_mem_addr = f'{thread_idx_x} + i * {self._num_threads}'
            self._assign(writer, shr_mem_addr, glb_mem_addr)

        else:
          # load using manual loop unrolling
          counter = 0

          while counter < num_hops:
            writer(f'{index_var} = {thread_idx_x} + {counter * self._num_threads};')
            shr_mem_addr = f'({index_var} % {src_data_view.lead_dim}) * {dest_data_view.lead_dim}'
            shr_mem_addr += f' + {index_var} / {src_data_view.lead_dim}'

            glb_mem_addr = f'{thread_idx_x} + {counter * self._num_threads}'

            self._assign(writer, shr_mem_addr, glb_mem_addr)
            counter += 1

      # the last hop to fill shared mem with data
      if (self._shm_volume % self._num_threads) != 0:
        residue = self._shm_volume - num_hops * self._num_threads

        with writer.If(f'{thread_idx_x} < {residue}'):
          writer(f'{index_var} = {thread_idx_x} + {num_hops * self._num_threads};')
          shr_mem_addr = f'({index_var} % {src_data_view.lead_dim}) * {dest_data_view.lead_dim}'
          shr_mem_addr += f' + {index_var} / {src_data_view.lead_dim}'

          glb_mem_addr = f'{thread_idx_x} + {num_hops * self._num_threads}'

          self._assign(writer, shr_mem_addr, glb_mem_addr)


class ExactTransposePatchLoader(AbstractShrMemLoader):
  """A strategy which loads only a necessary part of a matrix into shared memory.

  """

  def __init__(self, **kwargs):
    super(ExactTransposePatchLoader, self).__init__(**kwargs)

    data_view = self._src.data_view
    optimal_num_cols = _find_next_prime(data_view.columns)
    self._shm_volume = data_view.lead_dim * optimal_num_cols

    self._dest.data_view = DataView(rows=data_view.columns,
                                    columns=data_view.rows,
                                    lead_dim=optimal_num_cols,
                                    is_transposed=True)

  def gen_code(self, writer):
    super(ExactTransposePatchLoader, self).gen_code(writer)
    writer("// using ExactTransposePatchLoader")

    src_data_view = self._src.data_view
    dest_data_view = self._dest.data_view
    thread_idx_x = self._lexic.thread_idx_x
    num_hops = int(self._shm_volume / self._num_threads)
    index_var = 'index'
    counter_var = 'counter'

    with writer.Scope():

      writer.Pragma('unroll')
      # .format(self.matrix.get_actual_num_cols())
      with writer.For(f'int i = 0; i < {src_data_view.columns}; ++i'):
        num_hops = int(src_data_view.rows / self._num_threads)
        if num_hops > 0:
          # load using a for-loop
          writer.Pragma('unroll')
          with writer.For(f'int {counter_var} = 0; {counter_var} < {num_hops}; ++{counter_var}'):
            updated_thread_idx = f'{thread_idx_x} + {counter_var} * {self._num_threads}'

            writer(f'int {index_var} = {updated_thread_idx} + i * {src_data_view.rows};')
            shr_mem_index = f'({index_var} % {src_data_view.rows}) * {dest_data_view.lead_dim}'
            shr_mem_index += f' + {index_var} / {src_data_view.rows}'

            glb_mem_index = f'{updated_thread_idx} + i * {src_data_view.lead_dim}'

            self._assign(writer, shr_mem_index, glb_mem_index)

        # the last hop to fill shared mem with data
        if (src_data_view.rows % self._num_threads) != 0:
          residue = src_data_view.rows - num_hops * self._num_threads
          with writer.If(f'{thread_idx_x} < {residue}'):
            finial_offset = num_hops * self._num_threads
            updated_thread_idx = f'{thread_idx_x} + {finial_offset}'

            writer(f'int {index_var} = {updated_thread_idx} + i * {src_data_view.rows};')

            shr_mem_index = f'({index_var} % {src_data_view.rows}) * {dest_data_view.lead_dim}'
            shr_mem_index += f' + {index_var} / {src_data_view.rows}'

            glb_mem_index = f'{updated_thread_idx} + i * {src_data_view.lead_dim}'

            self._assign(writer, shr_mem_index, glb_mem_index)


# TODO: This method can be generalized from the exact transpose patch loader, 
# TODO: this method needs benchmarking to see the penalty compared to the previous methods,
# The transposed load might be resulting with many bank conflicts in the first load
# TODO: Specific tests for this strategy
class ArbitraryLeadingDimensionExactTransposePatchLoader(AbstractShrMemLoader):
  """A strategy which loads only a necessary part of a matrix into shared memory.

  """

  def __init__(self, **kwargs):
    super(ArbitraryLeadingDimensionExactTransposePatchLoader, self).__init__(**kwargs)

    data_view = self._src.data_view
    self._shm_volume = data_view.rows * data_view.columns

    self._dest.data_view = DataView(rows=data_view.columns,
                                    columns=data_view.rows,
                                    lead_dim=data_view.lead_dim,
                                    is_transposed=True)

  def gen_code(self, writer):
    super(ArbitraryLeadingDimensionExactTransposePatchLoader, self).gen_code(writer)
    writer("// using ArbitraryLeadingDimensionExactTransposePatchLoader")

    src_data_view = self._src.data_view
    dest_data_view = self._dest.data_view
    thread_idx_x = self._lexic.thread_idx_x
    num_hops = int(self._shm_volume / self._num_threads)
    index_var = 'index'
    counter_var = 'counter'

    assert (self._src.data_view.rows == self._dest.data_view.columns)
    assert (self._src.data_view.columns == self._dest.data_view.rows)

    with writer.Scope():

      writer.Pragma('unroll')
      # .format(self.matrix.get_actual_num_cols())
      with writer.For(f'int i = 0; i < {src_data_view.columns}; ++i'):
        num_hops = int(src_data_view.rows / self._num_threads)
        if num_hops > 0:
          # load using a for-loop
          writer.Pragma('unroll')
          with writer.For(f'int {counter_var} = 0; {counter_var} < {num_hops}; ++{counter_var}'):
            glb_mem_addr = f'{thread_idx_x} + {counter_var} * {self._num_threads} + i * {src_data_view.lead_dim}'

            # Change [i][j] -> [j][i]
            glb_mem_row = f'{thread_idx_x} + {counter_var} * {self._num_threads}'
            glb_mem_col = 'i'
            shr_mem_row = glb_mem_col
            shr_mem_col = glb_mem_row

            writer(f'const int shrMemRow = {shr_mem_row};')
            writer(f'const int shrMemCol = {shr_mem_col};')
            shr_mem_addr = f'shrMemRow + (shrMemCol * {dest_data_view.rows})'

            self._assign(writer, shr_mem_addr, glb_mem_addr)

          # the last hop to fill shared mem with data
          if (src_data_view.rows % self._num_threads) != 0:
            residue = src_data_view.rows - num_hops * self._num_threads
            with writer.If(f'{thread_idx_x} < {residue}'):
              # shr_mem_addr = f'{thread_idx_x} * {src_data_view.rows} + i'
              glb_mem_addr = f'{thread_idx_x} + {num_hops} * {self._num_threads} + i * {src_data_view.lead_dim}'

              # Change [i][j] -> [j][i]
              glb_mem_row = f'{thread_idx_x} + {num_hops} * {self._num_threads}'
              glb_mem_col = 'i'
              shr_mem_row = glb_mem_col
              shr_mem_col = glb_mem_row

              writer(f'const int shrMemRow = {shr_mem_row};')
              writer(f'const int shrMemCol = {shr_mem_col};')
              shr_mem_addr = f'shrMemRow + (shrMemCol * {dest_data_view.rows})'

              self._assign(writer, shr_mem_addr, glb_mem_addr)
        else:
          with writer.If(f'{thread_idx_x} < {src_data_view.rows}'):
            glb_mem_addr = f'{thread_idx_x} + i * {src_data_view.lead_dim}'

            # Change [i][j] -> [j][i]
            glb_mem_row = thread_idx_x
            glb_mem_col = 'i'
            shr_mem_row = glb_mem_col
            shr_mem_col = glb_mem_row

            writer(f'const int shrMemRow = {shr_mem_row};')
            writer(f'const int shrMemCol = {shr_mem_col};')
            shr_mem_addr = f'shrMemRow + (shrMemCol * {dest_data_view.rows})'

            self._assign(writer, shr_mem_addr, glb_mem_addr)

    # Since we changed the structure of the dataview after loading it to the shared memory
    # If lead_dim and num_rows are equal in a contigous matrix this is a no-op
    self._dest.data_view.rows = self._src.data_view.columns
    self._dest.data_view.columns = self._src.data_view.rows
    self._dest.data_view.lead_dim = self._dest.data_view.rows
