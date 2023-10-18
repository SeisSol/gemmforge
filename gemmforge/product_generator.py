import hashlib
import math
from copy import deepcopy
from io import StringIO

from gemmforge.instructions.allocate import ShrMemAlloc, ShrMemNewAlloc, ShrMemNewAssign
from gemmforge.instructions.builders.kernels.product.factory import ProductKernelsFactory
from gemmforge.instructions.ptr_manip import GetElementPtr
from . import constructs
from .abstract_gemmlike_generator import GemmLikeGenerator
from .abstract_generator import AbstractGenerator as Generator
from .basic_types import DataFlowDirection, GeneralLexicon
from .exceptions import GenerationError
from .instructions.builders.kernels import GemmKernelType
from .symbol_table import Symbol, SymbolType
from .thread_policies import TheadPolicyFactory
from .vm import VM


class ProductGenerator(GemmLikeGenerator):
  def __init__(self, vm: VM, kernel_type=GemmKernelType.AUTO):
    super(ProductGenerator, self).__init__(vm)
    self._kernel_type = kernel_type
    # Kernel Type is ignored right now prob.
    # Currently no transposed tensors are supported

    self._reg_array_obj = list()
    self._shr_mem_obj = list()
    self._shr_mem_loads = list()

    self._tensors = list()
    self._alphas = list()
    self._operation_descriptions = list()
    # No beta supported for this operation
    # self._betas = list()
    self._num_compute_threads = list()
    self._num_active_threads = list()

  # complete_operation_descriptions = List[self._descr, tensor_a, tensor_b, tensor_c, alpha, args]
  def set(self, complete_operation_descriptions):
    # if trans_a or trans_b:
    #    raise Exception("TODO: Tensor Product in gemmforge does not support transposed a or b currently")
    self._tensors = set()
    self._complete_operation_description = complete_operation_descriptions
    for offset, descr_item in enumerate(self._complete_operation_description):
      print(offset, descr_item)
      if descr_item[0] != "product":
        raise Exception("Fused product kernels should only exist of product descriptions")

      operation_description = descr_item[1]["descr"]
      # print(operation_description)
      # If a tensor appears both as sink and source we will change to sourcesink
      tensor_a = descr_item[1]["tensor_a"]
      tensor_a.set_name(operation_description.leftTerm.name)
      tensor_a.set_data_flow_direction(DataFlowDirection.SOURCE)
      tensor_a.temporary = operation_description.leftTerm.is_temporary
      tensor_a_as_sink = tensor_a.copy()
      tensor_a_as_sink.set_data_flow_direction(DataFlowDirection.SINK)
      if any(tensor.name == tensor_a.name for tensor in self._tensors):
        # Remove the tensor with the same name
        self._tensors = {tensor for tensor in self._tensors if tensor.name != tensor_a.name}
        tensor_a_as_sink.set_data_flow_direction(DataFlowDirection.SOURCESINK)
        self._tensors.add(tensor_a_as_sink)
      else:
        self._tensors.add(tensor_a)

      tensor_b = descr_item[1]["tensor_b"]
      tensor_b.set_name(operation_description.rightTerm.name)
      tensor_b.set_data_flow_direction(DataFlowDirection.SOURCE)
      tensor_b.temporary = operation_description.rightTerm.is_temporary
      tensor_b_as_sink = tensor_b.copy()
      tensor_b_as_sink.set_data_flow_direction(DataFlowDirection.SINK)
      if any(tensor.name == tensor_b.name for tensor in self._tensors):
        # Remove the tensor with the same name
        self._tensors = {tensor for tensor in self._tensors if tensor.name != tensor_b.name}
        tensor_b_as_sink.set_data_flow_direction(DataFlowDirection.SOURCESINK)
        self._tensors.add(tensor_b_as_sink)
      else:
        self._tensors.add(tensor_b)

      tensor_c = descr_item[1]["tensor_c"]
      tensor_c.set_name(operation_description.result.name)
      tensor_c.set_data_flow_direction(DataFlowDirection.SINK)
      tensor_c.temporary = operation_description.result.is_temporary
      tensor_c_as_source = tensor_c.copy()
      tensor_c_as_source.set_data_flow_direction(DataFlowDirection.SOURCE)
      if any(tensor.name == tensor_c.name for tensor in self._tensors):
        # Remove the tensor with the same name
        self._tensors = {tensor for tensor in self._tensors if tensor.name != tensor_c.name}
        tensor_c_as_source.set_data_flow_direction(DataFlowDirection.SOURCESINK)
        self._tensors.add(tensor_c_as_source)
      else:
        self._tensors.add(tensor_c)

      self._alphas.append(descr_item[1]["alpha"])
      self._operation_descriptions.append(operation_description)

    print(self._tensors)

    self._base_name = self._generate_base_name()
    self._is_set = True

  def generate(self):
    self._check_if_set()

    self._check()
    self._deduce_num_threads()
    self._populate_global_scope()
    self._emit_instructions()

    self._analyze()

    self._generate_kernel()
    self._generate_header()
    self._generate_launcher()

  def get_flops(self):
    print("WARNING GEMMFORGE: TODO: FLOPS")
    return 1

  def _generate_kernel(self):
    print("WARNING: TODO: A LOT OF STUFF HERE")
    src = StringIO()

    with constructs.Cpp(src) as file:

      max_num_threads_per_block = self._num_active_threads * self._num_ops_per_block
      kernel_bounds = max_num_threads_per_block
      team_index_str = self._lexic.batch_indexer_gemm()

      mem_sizes = [shr_mem_obj.get_total_size() for shr_mem_obj in self._shr_mem_obj]
      with self._lexic.kernel_definition(file,
                                         kernel_bounds,
                                         self._base_name,
                                         self._get_func_params(),
                                         self._precision,
                                         int(max(mem_sizes))):
        with file.If(f'{self.get_element_size_guard(file)}'):
          with file.If(f'{self.get_flag_guard(file)}'):
            names = set()
            assigned_names = set()
            for instrPerKernel in self._instructions:
            #The concept is, allocations are replicated on every kernel as every kernel gets these allocation requests
              for instr in instrPerKernel:
                if isinstance(instr, ShrMemNewAlloc):
                  instr.set_mults_per_block(self._num_ops_per_block) 
                  name = instr._dest.obj.name
                  if not name in names:
                    names.add(name)
                    instr.gen_code(file)
                  else:
                    instr.gen_code(None)
                if isinstance(instr, ShrMemNewAssign):
                  instr.set_mults_per_block(self._num_ops_per_block) 
                  name = instr._dest.obj.name
                  if not name in assigned_names:
                    instr.set_mults_per_block(1) 
                    instr.gen_code(file)
                    assigned_names.add(name)
                  else:
                    instr.gen_code(None)
            for instrPerKernel in self._instructions:
              with file.Scope():
                for instr in instrPerKernel:
                  if instr.is_ready() and \
                    not isinstance(instr, ShrMemNewAlloc) and \
                    not isinstance(instr, ShrMemNewAssign):
                    instr.gen_code(file)
                  else:
                    if not instr.is_ready():
                      pass
                      #raise GenerationError("product_generator: requested instr is not ready: ", instr)

      self._kernel = src.getvalue()

  def _generate_launcher(self):
    src = StringIO()
    with constructs.Cpp(src) as file:
      with file.Function(self._base_name, self._get_launcher_params()):
        file(f'{self._lexic.kernel_range_object()} {self._get_block_dim_spec()};')
        file(f'{self._lexic.kernel_range_object()} {self._get_grid_dim_spec()};')

        self._lexic.get_stream_via_pointer(file, 'stream', GeneralLexicon.STREAM_PTR_STR)
        file.Expression(self._lexic.get_launch_code(self._base_name,
                                                    'grid',
                                                    'block',
                                                    'stream',
                                                    self._get_func_args()))
        err = self._lexic.check_error()
        if err is not None:
          file.Expression(err)

      self._launcher = src.getvalue()

  def _generate_header(self):
    src = StringIO()
    with constructs.Cpp(src) as file:
      file.FunctionDeclaration(self._base_name, self._get_launcher_params(with_defaults=True))
      content = src.getvalue()
    self._header = content

  def _check(self):
    return

  def _deduce_num_threads(self):
    for i, operation_descr in enumerate(self._operation_descriptions):
      op1 = None
      op2 = None
      result = None
      for tensor in self._tensors:
        if tensor.name == operation_descr.leftTerm.name:
          op1 = tensor
        elif tensor.name == operation_descr.rightTerm.name:
          op2 = tensor
        elif tensor.name == operation_descr.result.name:
          result = tensor

      assert( result.direction == DataFlowDirection.SINK or result.direction == DataFlowDirection.SOURCESINK)
      total_cells = result.get_actual_volume()
      # Every thread gets a line, because only the first dimension is contiguously stored
      thread_count = total_cells / result.get_actual_num_dimensions()[0]

      num_vector_units_required = math.ceil(thread_count / self._hw_descr.vec_unit_length)
      self._num_compute_threads.append(thread_count)
      self._num_active_threads.append(num_vector_units_required * self._hw_descr.vec_unit_length)

  def _populate_global_scope(self):
    for tensor in self._tensors:
      if not tensor.temporary:
        self._symbol_table.add_symbol(Symbol(obj=tensor,
                                             name=tensor.name,
                                             stype=SymbolType.Batch))
    self._symbol_table.add_scope()

  def _emit_instructions(self):
    for i, operation_descr in enumerate(self._operation_descriptions):
      op1 = None
      op2 = None
      result = None
      for tensor in self._tensors:
        if tensor.name == operation_descr.leftTerm.name:
          op1 = tensor
        elif tensor.name == operation_descr.rightTerm.name:
          op2 = tensor
        elif tensor.name == operation_descr.result.name:
          result = tensor

      params = {'vm': self._vm,
                'product_kernel_type': self._kernel_type,
                'symbol_table': self._symbol_table,
                'op1': op1,
                'op2': op2,
                'result': result,
                'result_tensor': result,
                'alphas': self._alphas[i],
                'operation_description': operation_descr,
                'num_compute_threads': self._num_compute_threads[i],
                'num_active_threads': self._num_active_threads[i]}

      kernel_factory = ProductKernelsFactory(**params)
      self._kernel_type = kernel_factory.product_kernel_type()

      product_kernel_builder = kernel_factory.get_builder()
      product_kernel_builder.build()

      self._instructions.append(product_kernel_builder.get_instructions())
      self._symbol_table.pop_scope()
      self._symbol_table.add_scope()

      self._reg_array_obj.append(product_kernel_builder.get_reg_array_obj())
      self._shr_mem_obj.append(product_kernel_builder.get_shr_mem_obj())
      self._shr_mem_loads.append(product_kernel_builder.get_shr_mem_loads())


  def _analyze(self):
    for i, instrGroup in enumerate(self._instructions):
      shr_mem_counter = 0
      for instr in self._shr_mem_loads[i]:
        instr.set_shr_mem_offset(shr_mem_counter)
        shr_mem_counter += instr.compute_shared_mem_size()

      self._shr_mem_obj[i].set_size_per_mult(shr_mem_counter)

    # TODO: Support for multiple blocks
    self._num_ops_per_block = 1
    for i, instrGroup in enumerate(self._instructions):
      self._shr_mem_obj[i].set_mults_per_block(self._num_ops_per_block)
      for inst in instrGroup:
        if isinstance(inst, ShrMemNewAlloc):
          inst.set_mults_per_block(self._num_ops_per_block)


  def _generate_base_name(self):
    addresses = ""
    transpose = ""

    for tensor in self._tensors:
      if not tensor.temporary:
        addresses += f'{tensor.addressing[0]}_'
        transpose += "NT_"

    constants = f'{self._alpha}'

    tensorstrs = ""
    for tensor in self._tensors:
      if not tensor.temporary:
        tensorstrs += tensor.__str__()

    result = hashlib.md5(('{}_{}_{}'.format(
      constants,
      tensorstrs,
      self._kernel_type.value.__str__()).encode()))
    md5encoding = result.hexdigest()
    prefix = 's' if self._precision == "float" else "d"

    product_dims = ""
    for tensor in self._tensors:
      if not tensor.temporary:
        product_dims += "d"
        for d in tensor.get_dimensions():
          product_dims += f'{d}_'

    consts = "alpha_"
    consts += "alpha_".join([str(a).replace(".", "_") for a in self._alphas])
    return '{0}product_{1}_{2}_{3}_{4}_{5}'.format(prefix,
                                                   transpose,
                                                   product_dims,
                                                   consts,
                                                   addresses,
                                                   md5encoding[:Generator.ENCODING_LENGTH])

  def _get_func_params(self):
    base_params = super(ProductGenerator, self)._get_func_params(
      matrices=self._get_sorted_non_tmp_tensors()
    )
    for alpha in self._alphas:
      if not isinstance(alpha, float):
        raise Exception("Product supports only static alpha input (compile-time float not variable)")

    return f'{base_params}'

  def _get_launcher_params(self, with_defaults=False):
    base_params = super(ProductGenerator, self)._get_launcher_params(with_defaults,
                                                                     matrices=self._get_sorted_non_tmp_tensors())
    for alpha in self._alphas:
      if not isinstance(alpha, float):
        raise Exception("Product supports only static alpha input (compile-time float not variable)")

    return f'{base_params}'

  def _get_func_args(self):
    base_args = super(ProductGenerator, self)._get_func_args(
      matrices=self._get_sorted_non_tmp_tensors()
    )
    for alpha in self._alphas:
      if not isinstance(alpha, float):
        raise Exception("Product supports only static alpha input (compile-time float not variable)")

    return f'{base_args}'

  def _get_block_dim_spec(self):
    super(ProductGenerator, self)._get_block_dim_spec()
    return f'block({max(self._num_active_threads)}, {self._num_ops_per_block}, 1)'

  def _get_grid_dim_spec(self):
    super(ProductGenerator, self)._get_grid_dim_spec()
    num_blocks = "({0} + {1} - 1) / {1}".format(GeneralLexicon.NUM_ELEMENTS,
                                                self._num_ops_per_block)
    return f'grid({num_blocks}, 1, 1)'

  def _get_sorted_non_tmp_tensors(self):
    return sorted([m for m in self._tensors if not m.temporary], key=lambda x: x.name)
