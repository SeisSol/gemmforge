from gemmforge.basic_types import GeneralLexicon
from .abstract_instruction import AbstractInstruction


class ShrMemBasedProduct(AbstractInstruction):
  """This is a gemm operation which is based on pre-loading data into
  the shared memory. This operation performs well on Nvidia
  and AMD GPUs"""

  def __init__(self, **kwargs):
    super(ShrMemBasedProduct, self).__init__(kwargs['vm'])
    self._op1 = kwargs['op1']
    self._op2 = kwargs['op2']
    self._dest = kwargs['dest']
    self._result_tensor = kwargs['result_tensor']
    self._operation_description = kwargs['operation_description']
    self._num_threads = kwargs['num_threads']

    self._is_ready = True

  def _find_operand_with_name(self, name):
    raise Exception("Method not fixed yet")
    for operand in [self._op1, self._op2]:
      if operand.name == name or \
          operand.name == GeneralLexicon.GLOBAL_MEM_PREFIX + name:
        return operand
    assert (False)

  def gen_code(self, writer):
    writer("/*")
    writer(f"This is the product kernel created from the following YaTeTo description:")
    writer(str(self._operation_description))
    writer("*/")
    # writer("/*")
    # writer("\n".join(str(x) for x in self._ops))
    # writer(str(self._ops))
    # writer("*/")
    
    thread_idx_x = self._vm.get_lexic().thread_idx_x
    operation = self._operation_description
    print(operation)
    op1 = self._op1
    threads_needed_for_operation = self._result_tensor.get_accumulated_dimensions()[-1] // self._result_tensor.get_dimensions()[1]
    writer.If(self.gen_mask_threads(int(threads_needed_for_operation))).__enter__()

    dims = self._result_tensor.get_dimensions()
    accumulated_dims = self._result_tensor.get_accumulated_dimensions()

    dest_strides = operation.result.memoryLayout._stride
    dest_indices = operation.result.indices
    loop_iterator_rows = None
    """
    for offset in range(len(dest_strides)):
      print(offset, dest_strides[offset], dest_indices[offset])
      print(dest_strides[offset], dims[1])
      if dest_strides[offset] == dims[1]:
        loop_iterator_rows = dest_indices[offset]
    print(dest_indices, dest_strides, loop_iterator_rows)
    assert (loop_iterator_rows != None)
    """
    loop_iterator_rows = operation.result.indices[1]
    #raise Exception(loop_iterator_rows)

    offests_strs = list()

    acc_dims = accumulated_dims
    writer(f"int rows_left = {thread_idx_x};")
    if (len(dims) >= 2):
      for i in range(len(dims)-1, -1, -1):
        if i == 1:
          #s1 = f"const int row_offset_{i-1} = rows_left;"
          #s2 = ""
          #s3 = f"const int dim_offset_{dest_indices[i-1]} = row_offset_{i-1};"
          s1 = ""
          s2 = ""
          s3 = ""
        elif i == 0:
          #s1 = f"const int row_offset_{i} = rows_left % {dims[1]};"
          s1 = f"const int row_offset_{i} = rows_left;"
          #s2 = f"rows_left -= row_offset_{i} * {acc_dims[i]//dims[1]}; // should be 0"
          s2 = ""
          s3 = f"const int dim_offset_{dest_indices[i]} = row_offset_{i};"
        else:
          s1 = f"const int row_offset_{i-1} = rows_left / {acc_dims[i]//dims[1]};"
          s2 = f"rows_left -= row_offset_{i-1} * {acc_dims[i]//dims[1]};"
          s3 = f"const int dim_offset_{dest_indices[i]} = row_offset_{i-1};"
        writer(s1)
        writer(s2)
        writer(s3)
    else:
      s1 = f"const int row_offset_{i-1} = rows_left;"
      s3 = f"const int dim_offset_{dest_indices[i-1]} = row_offset_{i-1};"
      writer(s1)
      writer(s3)

    # The dictionary should be ordered we need python 3.8
    it = 0
    item = operation.loopRanges[loop_iterator_rows]
    

    unit_stride_iterator = None
    for offset in range(len(dest_strides)):
      print(offset, dest_strides[offset], dest_indices[offset])
      if dest_strides[offset] == 1:
        unit_stride_iterator = dest_indices[offset]
    assert (unit_stride_iterator != None)

    #row_offset_str = f"const int row_offset = {thread_idx_x} % {self._result_tensor.get_dimensions()[0]};"
    #writer(row_offset_str)

    (loop_iterator, loop_range) = loop_iterator_rows, item
    writer.Pragma("unroll")
    writer.For(
      f"int {loop_iterator} = {loop_range.start}; {loop_iterator} < {loop_range.stop}; ++{loop_iterator}").__enter__()

    op1_strides = operation.leftTerm.memoryLayout._stride
    op1_indices = operation.leftTerm.indices
    op2 = self._op2
    op2_strides = operation.rightTerm.memoryLayout._stride
    op2_indices = operation.rightTerm.indices
    dest = self._dest
    dest_strides = operation.result.memoryLayout._stride
    dest_indices = operation.result.indices
    kernel_str = ""
    kernel_str += dest.name
    if loop_range.stop == 1:
      kernel_str += " = "
    else:
      kernel_str += f"["
      #for offset in range(len(dest_strides)):
      #  if loop_iterator_rows and dest_indices[offset] == loop_iterator_rows:
      #    kernel_str += thread_idx_x
      #  else:
      #    kernel_str += dest_indices[offset]
      #  kernel_str += " * " + str(dest_strides[offset])
      #  if offset != len(dest_strides) - 1:
      #    kernel_str += " + "
      kernel_str += loop_iterator_rows
      kernel_str += "] = "
    if operation.alpha != 1.0:
      kernel_str += str(operation.alpha) + " * "
    kernel_str += op1.name
    kernel_str += "["
    for offset in range(len(op1_strides)):
      if loop_iterator_rows and op1_indices[offset] == loop_iterator_rows:
        kernel_str += loop_iterator
      else:
        kernel_str += "dim_offset_" + op1_indices[offset]
      kernel_str += " * " + str(op1_strides[offset])
      if offset != len(op1_strides) - 1:
        kernel_str += " + "
    kernel_str += "] * "
    kernel_str += op2.name
    kernel_str += "["
    for offset in range(len(op2_strides)):
      if loop_iterator_rows and op2_indices[offset] == loop_iterator_rows:
        kernel_str += loop_iterator
      else:
        kernel_str += "dim_offset_" + op2_indices[offset]
      kernel_str += " * " + str(op2_strides[offset])
      if offset != len(op2_strides) - 1:
        kernel_str += " + "
    kernel_str += "];"
    writer(kernel_str)

    assert (loop_iterator_rows != None)
    writer.For("...").__exit__(type=None, value=None, traceback=None)
    writer.If("...").__exit__(type=None, value=None, traceback=None)

  def __str__(self) -> str:
    return f'{self._dest.name} = product(TODO...)'


class RegisterOnlyProduct(AbstractInstruction):
  def __init__(self, **kwargs):
    super(RegisterOnlyProduct, self).__init__(kwargs['vm'])
    raise Exception("Register Only Product Kernel is not yet supported")
