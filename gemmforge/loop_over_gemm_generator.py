import hashlib
from copy import deepcopy
from io import StringIO

from gemmforge.gemm_generator import GemmGenerator
from gemmforge.instructions.allocate import ShrMemNewAlloc, ShrMemNewAssign
from gemmforge.instructions.builders import ptr_manip_builder
from gemmforge.instructions.builders.allocator_builder import ShrMemAllocBuilder, ShrMemNewAllocBuilder, ShrMemNewAssignBuilder
from gemmforge.instructions.builders.kernels.gemms.factory import GemmKernelsFactory
from gemmforge.instructions.builders.kernels.log.factory import LoopOverGemmKernelsFactory
from gemmforge.instructions.ptr_manip import GetElementPtr
from gemmforge.matrix.dense import DenseMatrix
from . import constructs
from .abstract_gemmlike_generator import GemmLikeGenerator
from .abstract_generator import AbstractGenerator as Generator
from .basic_types import DataFlowDirection, GeneralLexicon
from .exceptions import GenerationError, InternalError
from .instructions.builders.kernels import GemmKernelType
from .symbol_table import Symbol, SymbolType
from .vm import VM


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

    self._matrices = set()
    self._tmp_id = 0

    self._buffer_sizes = dict()
    self._buff_name_to_matrix_names = dict()
    self._buffer_matrices = list()

    self._apply_log_loop_heuristics = False

  def get_next_tmp_name(self):
    tmp_name = "tmp" + str(self._tmp_id)
    self._tmp_id += 1
    return tmp_name

  def set(self, complete_operation_description, base_name=None):
    if len(self._complete_operation_description) != 0:
      raise InternalError("Set should not be called twice")

    self._instructions = []
    self._complete_operation_description = complete_operation_description

    self._base_name = base_name if base_name is not None else self._generate_base_name()
    print(self._complete_operation_description)

    self._matrices = set()
    for offset, descr_item in enumerate(self._complete_operation_description):
      print(offset, descr_item)
      if descr_item[0] == "gemm":
        operation_description = descr_item[1]["descr"]
        # raise Exception(str(operation_description) +"\n"+ str(descr_item[1]))

        for offset, tensor_description in enumerate([operation_description.result, \
                                                      operation_description.leftTerm, \
                                                      operation_description.rightTerm]):
          if tensor_description.is_temporary:
            tmp_name = self.get_next_tmp_name()
            tensor_name = tensor_description.name
            tensor_description.name = tmp_name

            if tensor_name in self._buffer_sizes.keys():
              if tensor_description.eqspp.size > self._buffer_sizes[tensor_name]:
                self._buffer_sizes[tensor_name] = \
                  tensor_description.eqspp.size
            else:
              self._buffer_sizes[tensor_name] = \
                tensor_description.eqspp.size

            if tensor_name in self._buff_name_to_matrix_names:
              self._buff_name_to_matrix_names[tensor_name].append(tmp_name)
            else:
              self._buff_name_to_matrix_names[tensor_name] = [tmp_name]

            if offset == 0:
              descr_item[1]["matrix_c"].name = tmp_name
              print("Rename:", descr_item[1]["matrix_c"])
            elif offset == 1:
              descr_item[1]["matrix_a"].name = tmp_name
              print("Rename:", descr_item[1]["matrix_a"])
            else:
              descr_item[1]["matrix_b"].name = tmp_name
              print("Rename:", descr_item[1]["matrix_b"])

        a = descr_item[1]["matrix_a"]
        a.set_name(operation_description.leftTerm.name)
        a.set_data_flow_direction(DataFlowDirection.SOURCE)
        a.temporary = operation_description.leftTerm.is_temporary
        a_as_sink = a.copy()
        a_as_sink.set_data_flow_direction(DataFlowDirection.SINK)
        if any(matrix.name == a.name for matrix in self._matrices):
          # Remove the tensor with the same name
          self._matrices = {matrix for matrix in self._matrices if matrix.name != a.name}
          a_as_sink.set_data_flow_direction(DataFlowDirection.SOURCESINK)
          self._matrices.add(a_as_sink)
          print("Add: ", a_as_sink)
        else:
          self._matrices.add(a)
          print("Add: ", a)

        b = descr_item[1]["matrix_b"]
        b.set_name(operation_description.rightTerm.name)
        b.set_data_flow_direction(DataFlowDirection.SOURCE)
        b.temporary = operation_description.rightTerm.is_temporary
        b_as_sink = b.copy()
        b_as_sink.set_data_flow_direction(DataFlowDirection.SINK)
        if any(matrix.name == b.name for matrix in self._matrices):
          # Remove the tensor with the same name
          self._matrices = {matrix for matrix in self._matrices if matrix.name != b.name}
          b_as_sink.set_data_flow_direction(DataFlowDirection.SOURCESINK)
          self._matrices.add(b_as_sink)
          print("Add: ", b_as_sink)
        else:
          self._matrices.add(b)
          print("Add: ", b)

        c = descr_item[1]["matrix_c"]
        c.set_name(operation_description.result.name)
        c.set_data_flow_direction(DataFlowDirection.SINK)
        c.temporary = operation_description.result.is_temporary
        c_as_source = c.copy()
        c_as_source.set_data_flow_direction(DataFlowDirection.SOURCE)
        if any(matrix.name == c.name for matrix in self._matrices):
          # Remove the tensor with the same name
          self._matrices = {matrix for matrix in self._matrices if matrix.name != c.name}
          c_as_source.set_data_flow_direction(DataFlowDirection.SOURCESINK)
          self._matrices.add(c_as_source)
          print("Add: ", c_as_source)
        else:
          self._matrices.add(c)
          print("Add: ", c)

    loop_count = 0
    for offset, descr_item in enumerate(self._complete_operation_description):
      if descr_item[0] == "forLoopBegin":
        loop_count += 1
      if descr_item[0] == "forLoopEnd":
        loop_count -= 1
      if descr_item[0] == "gemm":
        operation_description = descr_item[1]["descr"]
        aname = operation_description.leftTerm.name
        bname = operation_description.rightTerm.name
        cname = operation_description.result.name

        am = None
        bm = None
        cm = None
        for matrix in self._matrices:
          if matrix.name == aname:
            am = matrix
          elif matrix.name == bname:
            bm = matrix
          elif matrix.name == cname:
            cm = matrix

        assert (am.direction != None)
        assert (am.name != None)
        assert (bm.direction != None)
        assert (bm.name != None)
        assert (cm.direction != None)
        assert (cm.name != None)
        gemm_generator = GemmGenerator(vm=self._vm, kernel_type=self._kernel_type)
        gemm_generator._symbol_table = self._symbol_table
        gemm_generator.set(trans_a=operation_description.transA,
                           trans_b=operation_description.transB,
                           mat_a=am, mat_b=bm, mat_c=cm,
                           alpha=operation_description.alpha,
                           beta=operation_description.beta,
                           base_name=f"LOGGemmKernel{offset}",
                           preserve_matrix_properties=True,
                           apply_log_loop_heuristics=self._apply_log_loop_heuristics and loop_count>=1,
                           load_bath_matrices=True)
        self._alphas.append(operation_description.alpha)
        self._betas.append(operation_description.beta)
        self._gemm_generators.append(gemm_generator)

    for gemm_generator in self._gemm_generators:
      gemm_generator._factory = LoopOverGemmKernelsFactory

    same_alpha = all(x == self._alphas[0] for x in self._alphas)
    #if not same_alpha:
    #  raise InternalError("TODO: multiple alphas in LOG")
    #self._alpha = self._alphas[0]

    print(self._matrices)
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
    raise Exception("TODO")

  def get_responsible_buffer(self, name):
    for buff_name, mat_names in self._buff_name_to_matrix_names.items():
      if name in mat_names:
        return buff_name
    return None

  def find_nearest_divisor(self, N, K):
      N = int(N)
      K = int(K)
      E = K

      while E > 0:
          if N % E == 0:
              return E  # E is a divisor of N

          E -= 1

      return 1 

  def _generate_kernel(self):
    self._kernel = ""
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
        with file.If(f'{self.get_element_size_guard(file)}'):
          with file.If(f'{self.get_flag_guard(file)}'):
            tab_count += 1
            for instr in self._instructions:
              if isinstance(instr, ShrMemNewAlloc) or \
                 isinstance(instr, GetElementPtr)  or \
                 isinstance(instr, ShrMemNewAssign):
                instr.set_mults_per_block(self._num_ops_per_block) 
              if instr.is_ready():
                instr.gen_code(file)
              else:
                raise GenerationError(f"gemm_generator: requested instr {instr} is not ready")
            tab_count += 1
            gemm_loop_offsets = {"lhs": ("", "+ 0"), "rhs": ("", "+ 0"), "result": ("", "+ 0")}
            loop_size = list()
            cc = 0
            for descr_item in self._complete_operation_description:
              if descr_item[0] == "gemm":
                gemm_generator = self._gemm_generators[cc]
                loop_size.append(gemm_generator._get_loop_size())
                cc+=1

            file("/*")
            file(f"This is the LoG created from the following YaTeTo description:")
            for descr in self._complete_operation_description:
              file(str(descr))
            file("*/")

            for it, descr_item in enumerate(self._complete_operation_description):
              loop_count = 0
              if descr_item[0] == "forLoopBegin":
                descr = descr_item[1]
                """
                inner_loop_size = loop_size[current_gemm_generator]
                unroll_count = 1
                elcount = inner_loop_size
                while elcount < 1024:
                  unroll_count += 1
                  elcount += inner_loop_size
                if int(descr['stop']) < 4 or unroll_count >= int(descr['stop']):
                  unroll_count = ""
                else:
                  unroll_count_old = int(unroll_count)
                  unroll_count = self.find_nearest_divisor(int(descr["stop"]), int(unroll_count))
                  if unroll_count == 1:
                    unroll_count = self.find_nearest_divisor(int(descr["stop"])+1, unroll_count_old )
                  if unroll_count == 1:
                    unroll_count = self.find_nearest_divisor(int(descr["stop"])+2, unroll_count_old )
                file.Pragma(f"unroll {unroll_count}")
                """
                if self._apply_log_loop_heuristics:
                  file.Pragma(f"unroll")
                #else:
                #  file.Pragma(f"unroll")
                file.For(
                  f"int {descr['index']} = {descr['start']}; {descr['index']} < {descr['stop']}; {descr['iter']}{descr['index']}").__enter__()
                tab_count += 1
                loop_count += 1
              elif descr_item[0] == "forLoopEnd":
                file.For("").__exit__(None, None, None)
                tab_count -= 1
                loop_count -= 1
              elif descr_item[0] == "InnerLoopBody":
                for key in ["lhs", "rhs", "result"]:
                  statement = descr_item[1][key]
                  s = f"//Original Loop: {statement['const_identifier']} {statement['float_type']}* {statement['lhs']} = "
                  s += f"{statement['rhs']} {statement['offset']}"
                  file(s.replace("  ", " "))
                  print(s.replace("  ", " "))
                  gemm_loop_offsets[key] = (statement['rhs'], statement["offset"])
              elif descr_item[0] == "OuterLoopBody":
                raise InternalError("OuterLoopBody in LOG not yet implemented")
              elif descr_item[0] == "gemm":
                with file.Scope():
                  s = str(self._complete_operation_description[it])
                  src.write(" " * (tab_count+1) + "//" + s + "\n")
                  gemm_generator = self._gemm_generators[current_gemm_generator]
                  gemm_generator._generate_device_kernel(gemm_loop_offsets, self.get_responsible_buffer)
                  gemm_kernels.append(gemm_generator._kernel)
                  gemm_loop_offsets = {"lhs": ("", "+ 0"), "rhs": ("", "+ 0"), "result": ("", "+ 0")}
                  for line in gemm_generator._kernel.split("\n"):
                    src.write("  " * (tab_count+1) + line + "\n")
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
      if gemm_active_threads > self._num_active_threads:
        self._num_active_threads = gemm_active_threads

  def _populate_global_scope(self):
    for matrix in self._matrices:
      #need_scope = False
      if not matrix.temporary:
        self._symbol_table.add_symbol(Symbol(obj=matrix,
                                             name=matrix.name,
                                             stype=SymbolType.Batch))

    print(self._buffer_sizes)
    for name, size in self._buffer_sizes.items():
      builder = ShrMemNewAllocBuilder(self._vm, self._symbol_table)
      buffer = DenseMatrix(size, 1, "none", [0, 0, size, 1], size)
      buffer.set_name(name + "_buffer")
      buffer.set_data_flow_direction(DataFlowDirection.SOURCESINK)
      self._buffer_matrices.append(buffer)
      symbol = builder.build(name + "_buffer", size, buffer)
      self._instructions.extend(builder.get_instructions())
    for buff_name, matrix_names in self._buff_name_to_matrix_names.items():
      builder = ShrMemNewAssignBuilder(self._vm, self._symbol_table)
      for matrix_name in matrix_names:
        did = False
        for matrix in self._matrices:
          print(matrix, matrix_name)
          if matrix.name == matrix_name:
            builder.build(matrix_name,buff_name + "_buffer",matrix)
            self._instructions.extend(builder.get_instructions())
            did = True
        if not did:
          raise Exception("HMM")
        #pass
    #if need_scope:
    #  self._symbol_table.add_scope()

    self._symbol_table.add_scope()

  def _emit_instructions(self):

    for gemm_generator in self._gemm_generators:
      gemm_generator._emit_instructions()
      self._symbol_table.pop_scope()
      self._symbol_table.add_scope()

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
      for inst in gemm_generator._instructions:
        if isinstance(inst, ShrMemNewAlloc):
          inst.set_mults_per_block(self._num_ops_per_block)

  def _generate_base_name(self):
    addresses = ""
    transpose = ""

    for tensor in self._matrices:
      if isinstance(tensor, DenseMatrix):
        tensor = tensor.as_tensor()
      if tensor.direction == DataFlowDirection.SOURCE or \
          tensor.direction == DataFlowDirection.SINK:
        addresses += f'{tensor.addressing[0]}_'
        transpose += "NT_"

    #constants = f'{self._alpha}'
    constants = ""
    tensorstrs = ""
    for tensor in self._matrices:
      if tensor.direction == DataFlowDirection.SOURCE or \
          tensor.direction == DataFlowDirection.SINK:
        tensorstrs += tensor.__str__()

    result = hashlib.md5(('{}_{}_{}'.format(
      constants,
      tensorstrs,
      self._kernel_type.value.__str__()).encode()))
    md5encoding = result.hexdigest()
    prefix = 's' if self._precision == "float" else "d"

    loopOverGEMM_dims = ""
    for tensor in self._matrices:
      if tensor.direction == DataFlowDirection.SOURCE or \
          tensor.direction == DataFlowDirection.SINK:
        loopOverGEMM_dims += "d"
        for d in [tensor.num_rows, tensor.num_cols]:
          loopOverGEMM_dims += f'{d}_'

    consts = "alpha_"
    consts += "alpha_".join([str(a).replace(".", "_") for a in self._alphas])
    consts += "_beta_"
    consts += "beta_".join([str(b).replace(".", "_") for b in self._betas])
    return '{0}loopOverGEMM_{1}_{2}_{3}_{4}_{5}'.format(prefix,
                                                        transpose,
                                                        loopOverGEMM_dims,
                                                        consts,
                                                        addresses,
                                                        md5encoding[:Generator.ENCODING_LENGTH])

  def _get_sorted_non_tmp_matrices(self):
    return sorted([m for m in self._matrices if not m.temporary], key=lambda x: x.name)
  
  def _get_func_params(self):
    base_params = super(LoopOverGemmGenerator, self)._get_func_params(
      matrices=self._get_sorted_non_tmp_matrices())
    for alpha in self._alphas:
      if not isinstance(alpha, float):
        raise Exception("LoopOverGemm supports only static alpha input (compile-time float not variable)")

    return base_params

  def _get_launcher_params(self, with_defaults=False):
    base_params = super(LoopOverGemmGenerator, self)._get_launcher_params(with_defaults, 
      matrices=self._get_sorted_non_tmp_matrices())
    for alpha in self._alphas:
      if not isinstance(alpha, float):
        raise Exception("LoopOverGemm supports only static alpha input (compile-time float not variable)")

    return base_params

  def _get_func_args(self):
    base_args = super(LoopOverGemmGenerator, self)._get_func_args(
      matrices=self._get_sorted_non_tmp_matrices())
    for alpha in self._alphas:
      if not isinstance(alpha, float):
        raise Exception("LoopOverGemm supports only static alpha input (compile-time float not variable)")

    return base_args

  def _get_block_dim_spec(self):
    super(LoopOverGemmGenerator, self)._get_block_dim_spec()
    return f'block({self._num_active_threads}, {self._num_ops_per_block}, 1)'

  def _get_grid_dim_spec(self):
    super(LoopOverGemmGenerator, self)._get_grid_dim_spec()
    num_blocks = "({0} + {1} - 1) / {1}".format(GeneralLexicon.NUM_ELEMENTS,
                                                self._num_ops_per_block)
    return f'grid({num_blocks}, 1, 1)'
