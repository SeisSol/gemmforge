from copy import deepcopy
from gemmforge.instructions.allocate import ShrMemNewAlloc
from gemmforge.instructions.builders.kernels.product.factory import ProductKernelsFactory
from .abstract_gemmlike_generator import GemmLikeGenerator
from . import constructs
from io import StringIO
from .exceptions import GenerationError
from .abstract_gemmlike_generator import GemmLikeGenerator
from .basic_types import GeneralLexicon, DataFlowDirection
from .symbol_table import Symbol, SymbolType
from .abstract_generator import AbstractGenerator as Generator
from .instructions.builders.kernels import GemmKernelsFactory
from .instructions.builders.kernels import GemmKernelType
from .vm import VM
from .thread_policies import TheadPolicyFactory
import math
import hashlib

class ProductGenerator(GemmLikeGenerator):
  def __init__(self, vm: VM, kernel_type=GemmKernelType.AUTO):
    super(ProductGenerator, self).__init__(vm)
    self._kernel_type = kernel_type
    # Kernel Type is ignored right now prob.
    # Currently no transposed tensors are supported

    self._reg_array_obj = None
    self._shr_mem_obj = None
    self._shr_mem_loads = list()

    self._tensors = list()
    self._alpha = 1.0
    self._operation_descriptions = list()
    # No beta supported for this operation
    # self._betas = list()

  # complete_operation_descriptions = List[self._descr, tensor_a, tensor_b, tensor_c, alpha, args]
  def set(self, complete_operation_descriptions):
    #if trans_a or trans_b:
    #    raise Exception("TODO: Tensor Product in gemmforge does not support transposed a or b currently")
    self._tensors = set()
    for operation_description in complete_operation_descriptions:
      #print(operation_description)
      # If a tensor appears both as sink and source we will change to sourcesink
      tensor_x = operation_description[1]
      tensor_x.set_name(operation_description[0].leftTerm.name)
      tensor_x.set_data_flow_direction(DataFlowDirection.SOURCE)
      tensor_x_as_sink = deepcopy(tensor_x)
      tensor_x_as_sink.set_data_flow_direction(DataFlowDirection.SINK)
      if any(tensor.name == tensor_x.name for tensor in self._tensors):
        # Remove the tensor with the same name
        self._tensors = {tensor for tensor in self._tensors if tensor.name != tensor_x.name}
        tensor_x_as_sink.set_data_flow_direction(DataFlowDirection.SOURCESINK)
        self._tensors.add(tensor_x_as_sink)
      else:
        self._tensors.add(tensor_x)

      tensor_x = operation_description[2]
      tensor_x.set_name(operation_description[0].rightTerm.name)
      tensor_x.set_data_flow_direction(DataFlowDirection.SOURCE)
      tensor_x_as_sink = deepcopy(tensor_x)
      tensor_x_as_sink.set_data_flow_direction(DataFlowDirection.SINK)
      if any(tensor.name == tensor_x.name for tensor in self._tensors):
        # Remove the tensor with the same name
        self._tensors = {tensor for tensor in self._tensors if tensor.name != tensor_x.name}
        tensor_x_as_sink.set_data_flow_direction(DataFlowDirection.SOURCESINK)
        self._tensors.add(tensor_x_as_sink)
      else:
        self._tensors.add(tensor_x)

      tensor_x = operation_description[3]
      tensor_x.set_name(operation_description[0].result.name)
      tensor_x.set_data_flow_direction(DataFlowDirection.SINK)
      tensor_x_as_source = deepcopy(tensor_x)
      tensor_x_as_source.set_data_flow_direction(DataFlowDirection.SOURCE)
      if any(tensor.name == tensor_x.name for tensor in self._tensors):
        # Remove the tensor with the same name
        self._tensors = {tensor for tensor in self._tensors if tensor.name != tensor_x.name}
        tensor_x_as_source.set_data_flow_direction(DataFlowDirection.SOURCESINK)
        self._tensors.add(tensor_x_as_source)
      else:
        self._tensors.add(tensor_x)

      self._alpha = operation_description[4]
      self._operation_descriptions.append(operation_description[0])

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
      kernel_bounds = [max_num_threads_per_block]
      team_index_str = self._lexic.batch_indexer_gemm()

      with self._lexic.kernel_definition(file,
                                         kernel_bounds,
                                         self._base_name,
                                         self._get_func_params(),
                                         self._precision,
                                         self._shr_mem_obj.get_total_size()):
        with file.If(f'{self.get_element_size_guard(file)}'):
          with file.If(f'{self.get_flag_guard(file)}'):
            for instr in self._instructions:
              if instr.is_ready():
                instr.gen_code(file)
              else:
                raise GenerationError("product_generator: requested instr is not ready: ", instr)

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
    max_thread_count = 0

    for tensor in self._tensors:
      if tensor.direction == DataFlowDirection.SOURCE:
        total_cells = tensor.get_actual_volume()
        # Every thread gets a line, because only the first dimension is contiguously stored
        thread_count = total_cells / tensor.get_actual_num_dimensions()[0]

        if thread_count > max_thread_count:
          max_thread_count = thread_count

    num_vector_units_required = math.ceil(thread_count / self._hw_descr.vec_unit_length)
    self._num_compute_threads = thread_count
    self._num_active_threads = num_vector_units_required * self._hw_descr.vec_unit_length


  def _populate_global_scope(self):
    for tensor in self._tensors:
      if tensor.direction == DataFlowDirection.SOURCE or tensor.direction == DataFlowDirection.SINK:
        self._symbol_table.add_symbol(Symbol(obj=tensor,
                                              name=tensor.name,
                                              stype=SymbolType.Batch))
    self._symbol_table.add_scope()

  def _emit_instructions(self):
    params = {'vm': self._vm,
              'product_kernel_type': self._kernel_type,
              'symbol_table': self._symbol_table,
              'tensors': self._tensors,
              'alpha': self._alpha,
              'operation_descriptions': self._operation_descriptions,
              'num_compute_threads': self._num_compute_threads,
              'num_active_threads': self._num_active_threads}

    kernel_factory = ProductKernelsFactory(**params)
    self._kernel_type = kernel_factory.product_kernel_type()

    product_kernel_builder = kernel_factory.get_builder()
    product_kernel_builder.build()

    self._instructions = product_kernel_builder.get_instructions()
    self._reg_array_obj = product_kernel_builder.get_reg_array_obj()
    self._shr_mem_obj = product_kernel_builder.get_shr_mem_obj()
    self._shr_mem_loads = product_kernel_builder.get_shr_mem_loads()

  def _analyze(self):
    # compute total required shr. mem
    shr_mem_counter = 0
    for instr in self._shr_mem_loads:
      instr.set_shr_mem_offset(shr_mem_counter)
      shr_mem_counter += instr.compute_shared_mem_size()

    self._shr_mem_obj.set_size_per_mult(shr_mem_counter)

    result_tensor = None
    for tensor in self._tensors:
      if tensor.direction == DataFlowDirection.SINK:
        result_tensor = tensor
    assert(result_tensor != None)

    # compute num matrix multiplications per block
    thread_policy = TheadPolicyFactory.get_product_policy(vm=self._vm,
                                                          shr_mem_per_op=shr_mem_counter,
                                                          num_threads=self._num_active_threads,
                                                          ops=self._tensors,
                                                          res=result_tensor)

    self._num_ops_per_block = thread_policy.get_num_ops_per_block()
    self._shr_mem_obj.set_mults_per_block(self._num_ops_per_block)

    for inst in self._instructions:
      if isinstance(inst, ShrMemNewAlloc):
        inst.set_mults_per_block(self._num_ops_per_block)

  def _generate_base_name(self):
    addresses = ""
    transpose = ""

    for tensor in self._tensors:
      if tensor.direction == DataFlowDirection.SOURCE or \
         tensor.direction == DataFlowDirection.SINK:
        addresses += f'{tensor.addressing[0]}_'
        transpose += "NT_"

    constants = f'{self._alpha}'

    tensorstrs = ""
    for tensor in self._tensors:
      if tensor.direction == DataFlowDirection.SOURCE or \
         tensor.direction == DataFlowDirection.SINK:
        tensorstrs += tensor.__str__()

    result = hashlib.md5(('{}_{}_{}'.format(
      constants,
      tensorstrs,
      self._kernel_type.value.__str__()).encode()))
    md5encoding = result.hexdigest()
    prefix = 's' if self._precision == "float" else "d"

    product_dims = ""
    for tensor in self._tensors:
      if tensor.direction == DataFlowDirection.SOURCE or \
         tensor.direction == DataFlowDirection.SINK:
        product_dims += "d"
        for d in tensor.get_dimensions():
            product_dims += f'{d}_'

    consts = f'alpha_{int(self._alpha)}'
    return '{0}product_{1}_{2}_{3}_{4}_{5}'.format(prefix,
                                                    transpose,
                                                    product_dims,
                                                    consts,
                                                    addresses,
                                                    md5encoding[:Generator.ENCODING_LENGTH])

  def _get_func_params(self):
    base_params = super(ProductGenerator, self)._get_func_params(matrices=self._tensors)
    if isinstance(self._alpha, float):
      return base_params
    else:
      return f'{self._precision} {self._alpha}, {base_params}'

  def _get_launcher_params(self, with_defaults=False):
    base_params = super(ProductGenerator, self)._get_launcher_params(with_defaults, matrices=self._tensors)
    if isinstance(self._alpha, float):
      return base_params
    else:
      return f'{self._precision} {self._alpha}, {base_params}'

  def _get_func_args(self):
    base_args = super(ProductGenerator, self)._get_func_args(matrices=self._tensors)
    if isinstance(self._alpha, float):
      return base_args
    else:
      return f'{self._alpha}, {base_args}'

  def _get_block_dim_spec(self):
    super(ProductGenerator, self)._get_block_dim_spec()
    return f'block({self._num_active_threads}, {self._num_ops_per_block}, 1)'

  def _get_grid_dim_spec(self):
    super(ProductGenerator, self)._get_grid_dim_spec()
    num_blocks = "({0} + {1} - 1) / {1}".format(GeneralLexicon.NUM_ELEMENTS,
                                                self._num_ops_per_block)
    return f'grid({num_blocks}, 1, 1)'
