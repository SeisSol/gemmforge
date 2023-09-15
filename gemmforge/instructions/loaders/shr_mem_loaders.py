from copy import deepcopy

from .abstract_loader import AbstractShrMemLoader


class ExtendedPatchLoader(AbstractShrMemLoader):
  """A strategy which loads an entire matrix into shared memory
  including padding (caused by a difference between lead_dim and
  number of rows)
  """

  def __init__(self, **kwargs):
    super(ExtendedPatchLoader, self).__init__(**kwargs)

    data_view = self._src.data_view
    full_subvolume = (data_view.columns - 2) * data_view.lead_dim
    cropped_subvolume = data_view.rows + data_view.lead_dim
    self._shm_volume = cropped_subvolume + full_subvolume
    self._dest.data_view = deepcopy(self._src.data_view)

  def gen_code(self, writer):
    super(ExtendedPatchLoader, self).gen_code(writer)
    writer("// using ExtendedPatchLoader")

    with writer.Scope():

      thread_idx_x = self._lexic.thread_idx_x
      num_hops = int(self._shm_volume / self._num_threads)
      if num_hops > 0:
        if num_hops > self._manual_unroll_threshold:
          # load using a for-loop
          writer.Pragma("unroll")
          with writer.For(f'int i = 0; i < {num_hops}; ++i'):
            shr_mem_addr = f'{thread_idx_x} + i * {self._num_threads}'
            glb_mem_addr = f'{thread_idx_x} + i * {self._num_threads}'

            self._assign(writer, shr_mem_addr, glb_mem_addr)
        else:
          # load using manual loop unrolling
          for counter in range(num_hops):
            shr_mem_addr = f'{thread_idx_x} + {self._num_threads * counter}'
            glb_mem_addr = f'{thread_idx_x} + {self._num_threads * counter}'

            self._assign(writer, shr_mem_addr, glb_mem_addr)

      # the last hop to fill shared mem with data
      if (self._shm_volume % self._num_threads) != 0:
        residue = self._shm_volume - num_hops * self._num_threads
        with writer.If(f'{thread_idx_x} < {residue}'):
          shr_mem_addr = f'{thread_idx_x} + {num_hops * self._num_threads}'
          glb_mem_addr = f'{thread_idx_x} + {num_hops * self._num_threads}'

          self._assign(writer, shr_mem_addr, glb_mem_addr)


class ExtendedTensorLoader(AbstractShrMemLoader):
  """A strategy which loads an entire matrix into shared memory
  including padding (caused by a difference between lead_dim and
  number of rows)
  """

  def __init__(self, **kwargs):
    super(ExtendedTensorLoader, self).__init__(**kwargs)

    data_view = self._src.data_view
    self._shm_volume = self._src.obj.get_volume()
    self._dest.data_view = deepcopy(self._src.data_view)

  def gen_code(self, writer):
    super(ExtendedTensorLoader, self).gen_code(writer)
    writer("// using ExtendedTensorLoader")

    with writer.Scope():

      thread_idx_x = self._lexic.thread_idx_x
      num_hops = int(self._shm_volume / self._num_threads)
      if num_hops > 0:
        if num_hops > self._manual_unroll_threshold:
          # load using a for-loop
          writer.Pragma("unroll")
          with writer.For(f'int i = 0; i < {num_hops}; ++i'):
            shr_mem_addr = f'{thread_idx_x} + i * {self._num_threads}'
            glb_mem_addr = f'{thread_idx_x} + i * {self._num_threads}'

            self._assign(writer, shr_mem_addr, glb_mem_addr)
        else:
          # load using manual loop unrolling
          for counter in range(num_hops):
            shr_mem_addr = f'{thread_idx_x} + {self._num_threads * counter}'
            glb_mem_addr = f'{thread_idx_x} + {self._num_threads * counter}'

            self._assign(writer, shr_mem_addr, glb_mem_addr)

      # the last hop to fill shared mem with data
      if (self._shm_volume % self._num_threads) != 0:
        residue = self._shm_volume - num_hops * self._num_threads
        with writer.If(f'{thread_idx_x} < {residue}'):
          shr_mem_addr = f'{thread_idx_x} + {num_hops * self._num_threads}'
          glb_mem_addr = f'{thread_idx_x} + {num_hops * self._num_threads}'

          self._assign(writer, shr_mem_addr, glb_mem_addr)


class ExactPatchLoader(AbstractShrMemLoader):
  """A strategy which loads only a necessary part of a matrix into shared memory.

  """

  def __init__(self, **kwargs):
    super(ExactPatchLoader, self).__init__(**kwargs)
    data_view = self._src.data_view
    self._shm_volume = data_view.rows * data_view.columns

    self._dest.data_view = deepcopy(self._src.data_view)
    self._dest.data_view.lead_dim = data_view.rows

  def gen_code(self, writer):
    super(ExactPatchLoader, self).gen_code(writer)
    # If leading dim != num rows then we have a source where the matrix is stored
    # Non-contigously meaning space between columns. this would happen with tensor slices
    src_data_view = self._src.data_view
    dest_data_view = self._dest.data_view

    print(src_data_view)
    if src_data_view.lead_dim == src_data_view.rows:
      writer("// using ExactPatchLoader for leading dimension != number of rows")
    else:
      writer("// using ExactPatchLoader")

    with writer.Scope():
      thread_idx_x = self._lexic.thread_idx_x
      writer.Pragma("unroll")
      with writer.For(f'int i = 0; i < {src_data_view.columns}; ++i'):
        num_hops = int(dest_data_view.rows / self._num_threads)
        if num_hops > 0:
          if num_hops > self._manual_unroll_threshold:
            writer.Pragma("unroll")
            with writer.For(f'int counter = 0; counter < {num_hops}; ++counter'):
              shr_mem_addr = f'{thread_idx_x}'
              shr_mem_addr += f' + counter * {self._num_threads} + i * {dest_data_view.rows}'

              glb_mem_addr = f'{thread_idx_x}'
              glb_mem_addr += f' + counter * {self._num_threads} + i * {src_data_view.lead_dim}'

              self._assign(writer, shr_mem_addr, glb_mem_addr)
          else:
            for counter in range(num_hops):
              offset = counter * self._num_threads
              shr_mem_addr = f'{thread_idx_x} + {offset} + i * {dest_data_view.rows}'
              glb_mem_addr = f'{thread_idx_x} + {offset} + i * {src_data_view.lead_dim}'
              self._assign(writer, shr_mem_addr, glb_mem_addr)

        # the last hop to fill shared mem with data
        if (dest_data_view.rows % self._num_threads) != 0:
          residue = dest_data_view.rows - num_hops * self._num_threads
          with writer.If(f'{thread_idx_x} < {residue}'):
            finial_offset = num_hops * self._num_threads
            shr_mem_addr = f'{thread_idx_x} + {finial_offset} + i * {dest_data_view.rows}'
            glb_mem_addr = f'{thread_idx_x} + {finial_offset} + i * {src_data_view.lead_dim}'
            self._assign(writer, shr_mem_addr, glb_mem_addr)

    # Since we changed the structure of the dataview after loading it to the shared memory
    # If lead_dim and num_rows are equal in a contigous matrix this is a no-op
    self._dest.data_view.lead_dim = self._src.data_view.rows


class ExactTensorLoader(AbstractShrMemLoader):
  """A strategy which loads (currently) the whole tensor, todo subtensor

  """

  def __init__(self, **kwargs):
    super(ExactTensorLoader, self).__init__(**kwargs)
    data_view = self._src.data_view
    self._shm_volume = self._src.obj.get_size()
    self._dest.data_view = deepcopy(self._src.data_view)

  def gen_code(self, writer):
    super(ExactTensorLoader, self).gen_code(writer)

    src_data_view = self._src.data_view
    dest_data_view = self._dest.data_view
    src_data_view_padded_dimensions = self._src.obj.get_padded_dimensions()

    print(src_data_view)
    writer("// using ExactTensorLoader")

    with writer.Scope():
      thread_idx_x = self._lexic.thread_idx_x

      outerLoopCount = len(src_data_view.dimensions[1:])
      if outerLoopCount > 18:
        raise Exception(
          "Too many outer loops, need to change shr_mem_loader loop index generator for that to work")
      begin_char = 'i'
      current_char = begin_char
      for dim in src_data_view.dimensions[1:]:
        writer.Pragma("unroll")
        writer.For(f'int {current_char} = 0; {current_char} < {dim}; ++{current_char}').__enter__()
        current_char = chr(ord(current_char) + 1)

      offsetStr1 = ""
      offsetStr2 = ""
      current_char = begin_char
      for i in range(len(src_data_view.dimensions[1:])):
        offsetStr1 += f" + {dest_data_view.dimensions[i]} * {current_char}"
        offsetStr2 += f" + {src_data_view_padded_dimensions[i]} * {current_char}"
        current_char = chr(ord(current_char) + 1)

      num_hops = int(dest_data_view.dimensions[0] / self._num_threads)
      if num_hops > 0:
        writer.Pragma("unroll")
        with writer.For(f'int counter = 0; counter < {num_hops}; ++counter'):
          shr_mem_addr = f'{thread_idx_x}'
          shr_mem_addr += f' + counter * {self._num_threads}' + offsetStr1

          glb_mem_addr = f'{thread_idx_x}'
          glb_mem_addr += f' + counter * {self._num_threads}' + offsetStr2

          self._assign(writer, shr_mem_addr, glb_mem_addr)

      # the last hop to fill shared mem with data
      if (dest_data_view.dimensions[0] % self._num_threads) != 0:
        residue = dest_data_view.dimensions[0] - num_hops * self._num_threads
        with writer.If(f'{thread_idx_x} < {residue}'):
          finial_offset = num_hops * self._num_threads
          shr_mem_addr = f'{thread_idx_x} + {finial_offset}' + offsetStr1
          glb_mem_addr = f'{thread_idx_x} + {finial_offset}' + offsetStr2
          self._assign(writer, shr_mem_addr, glb_mem_addr)

      for dim in reversed(src_data_view.dimensions[1:]):
        writer.For(f'int i = 0; i < {dim}; ++i').__exit__(type=None, value=None, traceback=None)
