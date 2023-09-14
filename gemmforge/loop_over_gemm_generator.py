from copy import deepcopy

from gemmforge.gemm_generator import GemmGenerator
from gemmforge.matrix.dense import DenseMatrix
from .abstract_gemmlike_generator import GemmLikeGenerator
from . import constructs
from io import StringIO
from .exceptions import GenerationError, InternalError
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

class LoopOverGemmGenerator(GemmLikeGenerator):
  def __init__(self, vm: VM, kernel_type=GemmKernelType.AUTO):
    super(LoopOverGemmGenerator, self).__init__(vm)
    self._kernel_type = kernel_type

    self._complete_operation_description = list()
    self._gemm_generators = list()

    self._instructions = list()
    self._reg_array_objs = list()
    self._shr_mem_objs = list()
    self._shr_mem_loads = list()

    self._alphas = list()
    self._betas = list()
    
    self._tensors = set()
    self._matrices = set()

  def set(self, complete_operation_description, base_name=None):
    self._instructions = []
    self._complete_operation_description = complete_operation_description

    self._base_name = base_name if base_name is not None else self._generate_base_name()

    self._tensors = set()
    for offset, descr_item in enumerate(self._complete_operation_description):
      if descr_item[0] == "gemm":
        operation_description = descr_item[1]["descr"]
        #raise Exception(str(operation_description) +"\n"+ str(descr_item[1]))

        a = descr_item[1]["matrix_a"]
        a.set_name(operation_description.leftTerm.name)
        a.set_data_flow_direction(DataFlowDirection.SOURCE)
        a_as_sink = deepcopy(a)
        a_as_sink.set_data_flow_direction(DataFlowDirection.SINK)
        if any(tensor.name == a.name for tensor in self._tensors):
          # Remove the tensor with the same name
          self._tensors = {tensor for tensor in self._tensors if tensor.name != a.name}
          a_as_sink.set_data_flow_direction(DataFlowDirection.SOURCESINK)
          self._tensors.add(a_as_sink.as_tensor())
          self._matrices.add(a_as_sink)
        else:
          self._tensors.add(a.as_tensor())
          self._matrices.add(a)

        b = descr_item[1]["matrix_b"]
        b.set_name(operation_description.rightTerm.name)
        b.set_data_flow_direction(DataFlowDirection.SOURCE)
        b_as_sink = deepcopy(b)
        b_as_sink.set_data_flow_direction(DataFlowDirection.SINK)
        if any(tensor.name == b.name for tensor in self._tensors):
          # Remove the tensor with the same name
          self._tensors = {tensor for tensor in self._tensors if tensor.name != b.name}
          b_as_sink.set_data_flow_direction(DataFlowDirection.SOURCESINK)
          self._tensors.add(b_as_sink.as_tensor())
          self._matrices.add(b_as_sink)
        else:
          self._tensors.add(b.as_tensor())
          self._matrices.add(b)

        c = descr_item[1]["matrix_c"]
        c.set_name(operation_description.result.name)
        c.set_data_flow_direction(DataFlowDirection.SINK)
        c_as_source = deepcopy(c)
        c_as_source.set_data_flow_direction(DataFlowDirection.SOURCE)
        if any(tensor.name == c.name for tensor in self._tensors):
          # Remove the tensor with the same name
          self._tensors = {tensor for tensor in self._tensors if tensor.name != c.name}
          c_as_source.set_data_flow_direction(DataFlowDirection.SOURCESINK)
          self._tensors.add(c_as_source.as_tensor())
          self._matrices.add(c_as_source)
        else:
          self._tensors.add(c.as_tensor())
          self._matrices.add(c)

        gemm_generator = GemmGenerator(vm=self._vm, kernel_type=self._kernel_type)
        gemm_generator._symbol_table = self._symbol_table
        gemm_generator.set(trans_a = operation_description.transA, 
                           trans_b = operation_description.transB,
                           mat_a = a, mat_b=b, mat_c=c,
                           alpha = operation_description.alpha,
                           beta = operation_description.beta,
                           base_name=f"LOGGemmKernel{offset}")
        self._alphas.append(operation_description.alpha)
        self._betas.append(operation_description.beta)
        self._gemm_generators.append(gemm_generator)

    print(self._matrices)
    self._base_name = self._generate_base_name()
    self._is_set = True
    #print(self._complete_operation_description)
    #raise Exception(self._complete_operation_description)

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
    raise Exception("TODO")

  def _generate_kernel(self):
    self._kernel = "// Random Comment\n"
    src = StringIO()
    max_num_threads_per_block = self._num_active_threads * self._num_ops_per_block
    kernel_bounds = [max_num_threads_per_block]

    total_sizes = [obj.get_total_size() for obj in self._shr_mem_objs]
    gemm_kernels = list()
    current_gemm_generator = 0
    tab_count = 0

    with constructs.Cpp(src) as file:
      with self._lexic.kernel_definition(file,
                                          kernel_bounds,
                                          self._base_name,
                                          self._get_func_params(),
                                          self._precision,
                                          total_sizes):
        tab_count += 1
        with file.If(f'{self.get_flag_guard(file)}'):
          tab_count += 1
          with file.If(f'{self.get_element_size_guard(file)}'):
            tab_count += 1
            gemm_loop_offsets = {"lhs": ("", "+ 0"), "rhs": ("", "+ 0"), "result": ("", "+ 0")}
            for descr_item in self._complete_operation_description:
              if descr_item[0] == "forLoopBegin":
                descr = descr_item[1]
                file.Pragma("unroll")
                file.For(f"int {descr['index']} = {descr['start']}; {descr['index']} < {descr['stop']}; {descr['iter']}{descr['index']}").__enter__()
                tab_count += 1
              elif descr_item[0] == "forLoopEnd":
                file.For("").__exit__(None, None, None)
                tab_count -= 1
              elif descr_item[0] == "InnerLoopBody" or descr_item[0] == "OuterLoopBody":
                for key in ["lhs", "rhs", "result"]:
                  statement = descr_item[1][key]
                  s = f"//Original Loop: {statement['const_identifier']} {statement['float_type']}* {statement['lhs']} = "
                  s += f"{statement['rhs']} {statement['offset']}"
                  file(s.replace("  ", " "))
                  print(s.replace("  ", " "))
                  gemm_loop_offsets[key] = (statement['rhs'], statement["offset"])
              elif descr_item[0] == "gemm":
                gemm_generator = self._gemm_generators[current_gemm_generator]
                gemm_generator._generate_device_kernel(gemm_loop_offsets)
                gemm_kernels.append(gemm_generator._kernel)
                gemm_loop_offsets = {"lhs": ("", "+ 0"), "rhs": ("", "+ 0"), "result": ("", "+ 0")}
                for line in gemm_generator._kernel.split("\n"):
                  src.write("  "*tab_count + line + "\n")
                current_gemm_generator += 1
              else:
                raise InternalError("Unknown description item keyword found in LOG Generator")
          tab_count -= 1
        tab_count -= 1
      tab_count -= 1
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
    for gemm_generator in self._gemm_generators:
      gemm_generator._check()
    if self._complete_operation_description == None or \
      self._complete_operation_description == []:
        raise InternalError("Complete operation description for LOG Generator should not be empty at check stage")

  def _deduce_num_threads(self):
    self._num_compute_threads = 0
    self._num_active_threads = 0
    for gemm_generator in self._gemm_generators:
      gemm_generator._deduce_num_threads()
      gemm_compute_threads = gemm_generator._num_compute_threads
      gemm_active_threads = gemm_generator._num_active_threads

      if gemm_compute_threads > self._num_compute_threads:
        self._num_compute_threads = gemm_compute_threads
      if gemm_active_threads > self._num_active_threads :
        self._num_active_threads = gemm_active_threads

  def _populate_global_scope(self):
    for matrix in self._matrices:
      print(matrix)
      if matrix.direction == DataFlowDirection.SOURCE or \
          matrix.direction == DataFlowDirection.SINK:
        self._symbol_table.add_symbol(Symbol(obj=matrix,
                                              name=matrix.name,
                                              stype=SymbolType.Batch))
    self._symbol_table.print_scopes()
    #raise Exception("UWU")
    self._symbol_table.add_scope()

  def _emit_instructions(self):
    for gemm_generator in self._gemm_generators:
      gemm_generator._emit_instructions()
      self._instructions.append(gemm_generator._instructions)
      self._reg_array_objs.append(gemm_generator._reg_array_obj)
      self._shr_mem_objs.append(gemm_generator._shr_mem_obj)
      self._shr_mem_loads.append(gemm_generator._shr_mem_loads)

  def _analyze(self):
    for gemm_generator in self._gemm_generators:
      shr_mem_counter = 0
      for instr in gemm_generator._shr_mem_loads:
        instr.set_shr_mem_offset(shr_mem_counter)
        shr_mem_counter += instr.compute_shared_mem_size()

      gemm_generator._shr_mem_obj.set_size_per_mult(shr_mem_counter)

    # TODO: Support for multiple blocks
    self._num_ops_per_block = 1
    for gemm_generator in self._gemm_generators:
      gemm_generator._num_ops_per_block = self._num_ops_per_block
      gemm_generator._shr_mem_obj.set_mults_per_block(self._num_ops_per_block)

  def _generate_base_name(self):
    addresses = ""
    transpose = ""

    for tensor in self._tensors:
      if isinstance(tensor, DenseMatrix):
        tensor = tensor.as_tensor()
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

    consts = "alpha_"
    consts += "alpha_".join([str(a).replace(".", "_") for a in self._alphas])
    consts += "_beta_"
    consts += "beta_".join([str(b).replace(".", "_") for b in self._betas])
    return '{0}product_{1}_{2}_{3}_{4}_{5}'.format(prefix,
                                                    transpose,
                                                    product_dims,
                                                    consts,
                                                    addresses,
                                                    md5encoding[:Generator.ENCODING_LENGTH])


  def _get_func_params(self):
    base_params = super(LoopOverGemmGenerator, self)._get_func_params(matrices=self._tensors)
    if isinstance(self._alpha, float):
      return base_params
    else:
      return f'{self._precision} {self._alpha}, {base_params}'

  def _get_launcher_params(self, with_defaults=False):
    base_params = super(LoopOverGemmGenerator, self)._get_launcher_params(with_defaults, matrices=self._tensors)
    if isinstance(self._alpha, float):
      return base_params
    else:
      return f'{self._precision} {self._alpha}, {base_params}'

  def _get_func_args(self):
    base_args = super(LoopOverGemmGenerator, self)._get_func_args(matrices=self._tensors)
    if isinstance(self._alpha, float):
      return base_args
    else:
      return f'{self._alpha}, {base_args}'

  def _get_block_dim_spec(self):
    super(LoopOverGemmGenerator, self)._get_block_dim_spec()
    return f'block({self._num_active_threads}, {self._num_ops_per_block}, 1)'

  def _get_grid_dim_spec(self):
    super(LoopOverGemmGenerator, self)._get_grid_dim_spec()
    num_blocks = "({0} + {1} - 1) / {1}".format(GeneralLexicon.NUM_ELEMENTS,
                                                self._num_ops_per_block)
    return f'grid({num_blocks}, 1, 1)'
