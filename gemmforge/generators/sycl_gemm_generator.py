from gemmforge import constructs
from io import StringIO
from .abstract_generator import AbstractGenerator as Generator
from .gemm_generator import GemmGenerator


class SyclGemmGenerator(GemmGenerator):
    """ Generates GEMM GPU kernels: C = alpha * A * B + beta * C
    """

    def _generate_kernel(self):
        glob_symbols = {}
        for matrix in [self.mat_a, self.mat_b, self.mat_c]:
            glob_symbols[matrix.name] = "GlobMat{}".format(matrix.name)

        current_symbols = {}

        src = StringIO()
        with constructs.Cpp(src) as file:

            max_num_threads_per_block = self.num_active_threads * self.num_mult_per_block
            kernel_bounds = [max_num_threads_per_block]
            local_mem = "cl::sycl::accessor<{}, 1, cl::sycl::access::mode::read_write, cl::sycl::access::target::local> Scratch ({}, cgh);".format(self.precision, self._get_total_shared_mem_size())
            with file.SyclKernel(self.base_name, self._get_func_params(), kernel_bounds, local_mem):
                with file.If("{} < {}".format(self.TEAM_INDEX_STR, Generator.NUM_ELEMENTS_STR)):

                    # declare ptrs for correct matrices
                    file.VariableDeclaration("const {}*".format(self.precision),
                                             glob_symbols[self.mat_a.name],
                                             self._get_global_matrix_ptr(self.mat_a))

                    file.VariableDeclaration("const {}*".format(self.precision),
                                             glob_symbols[self.mat_b.name],
                                             self._get_global_matrix_ptr(self.mat_b))

                    file.VariableDeclaration("{}*".format(self.precision),
                                             glob_symbols[self.mat_c.name],
                                             self._get_global_matrix_ptr(self.mat_c))

                    # find address of matrix B within block shared memory
                    shr_mem_address = "&Scratch[{} * {}]".format(self.name_threadIdx_y, self.shr_mem_size_per_mult)
                    file.VariableDeclaration("{}*".format(self.precision),
                                             self.mat_b_loader.get_output_symbol(),
                                             shr_mem_address)

                    if self.mat_a.transpose:
                        # find address of matrix A within block shared memory
                        shr_mem_offset = self.mat_b_loader.compute_shared_mem_size()
                        shr_mem_address = "&Scratch[{} * {} + {}]".format(self.name_threadIdx_y,
                                                                          self.shr_mem_size_per_mult,
                                                                          shr_mem_offset)
                        file.VariableDeclaration("{}*".format(self.precision),
                                                 self.mat_a_loader.get_output_symbol(),
                                                 shr_mem_address)

                    # load matrices into shared memory
                    self.mat_b_loader.generate_scr(file, glob_symbols[self.mat_b.name])
                    self.mat_a_loader.generate_scr(file, glob_symbols[self.mat_a.name])
                    file.Expression("item.barrier()")

                    # set up current compute symbols within the rest of the scope
                    current_symbols[self.mat_b.name] = self.mat_b_loader.get_output_symbol()
                    current_symbols[self.mat_a.name] = self.mat_a_loader.get_output_symbol()
                    file.Emptyline()

                    with file.If("{} < {}".format(self.name_threadIdx_x, self.num_compute_threads)):
                        # allocate a buffer for each cuda thread to hold computed results
                        file.Emptyline()
                        zero_fp_value = "0.0{}".format('f' if self.precision == "float" else '')
                        file.ArrayDeclaration(self.precision,
                                              "Results",
                                              [zero_fp_value] * self.mat_c.get_actual_num_cols())

                        file.VariableDeclaration(self.precision, "Value")

                        # perform matrix multiplication
                        # m, n, k - according to the BLAS documentation. Read BLAS spec.
                        if self.mat_a.transpose:
                            contraction_length = self.mat_a.get_actual_num_rows()
                        else:
                            contraction_length = self.mat_a.get_actual_num_cols()

                        file.Emptyline()
                        with file.For("int k = 0; k < {}; ++k".format(contraction_length)):
                            first_operand = "{}[{} + {} * k]".format(current_symbols[self.mat_a.name],
                                                                     self.name_threadIdx_x,
                                                                     self.mat_a_loader.get_lid_dim())
                            file.Assignment("Value", "{}".format(first_operand))
                            file.Emptyline()
                            file.Pragma("unroll")
                            with file.For("int n = 0; n < {}; ++n".format(self.mat_c.get_actual_num_cols())):
                                if self.mat_b.transpose:
                                    second_operand = "{}[n + {} * k]".format(current_symbols[self.mat_b.name],
                                                                             self.mat_b_loader.get_lid_dim())
                                else:
                                    second_operand = "{}[k + {} * n]".format(current_symbols[self.mat_b.name],
                                                                             self.mat_b_loader.get_lid_dim())

                                file.Accumulate("Results[n]",
                                                "Value * {}".format(second_operand))

                        # write results back to memory
                        file.Emptyline()
                        file.Pragma("unroll")
                        with file.For("int n = 0; n < {}; ++n".format(self.mat_c.get_actual_num_cols())):
                            rhs = "{}[{} + {} * n]".format(glob_symbols[self.mat_c.name],
                                                           self.name_threadIdx_x,
                                                           self.mat_c.num_rows)

                            if self.alpha == 1.0:
                                lhs = "Results[n]"
                            else:
                                if self.precision == "float" and isinstance(self.alpha, float):
                                    lhs = f'{self.alpha}f * Results[n]'
                                else:
                                    lhs = f'{self.alpha} * Results[n]'

                            if self.beta != 0.0:
                                if self.beta == 1.0:
                                    lhs += " + {}".format(rhs)
                                else:
                                    lhs += " + {} * {}".format(
                                        "{}{}".format(self.beta, 'f' if self.precision == "float" else ''),
                                        rhs)

                            file.Assignment(rhs, lhs)

            self._kernel = src.getvalue()

    def _generate_launcher(self):
        src = StringIO()
        with constructs.Cpp(src) as file:
            with file.Function(self.base_name, self._get_launcher_params()):
                file.VariableDeclaration("cl::sycl::range<3>", self._get_block_dim_spec())
                file.VariableDeclaration("cl::sycl::range<3>", self._get_grid_dim_spec())

                file.Expression("cl::sycl::queue defaultQueue {cl::sycl::host_selector{}}")

                if_stream_exists = f'({Generator.STREAM_PTR_STR} != nullptr)'
                stream_obj = f'static_cast<{self.arch_lexic.get_stream_name()} *>({Generator.STREAM_PTR_STR})'
                file(f'{self.arch_lexic.get_stream_name()} *stream = {if_stream_exists} ? {stream_obj} : &defaultQueue;')

                file.Expression(self.arch_lexic.get_launch_code(self.base_name,
                                                                "blocks",
                                                                "threads",
                                                                "stream",
                                                                self._get_func_args()))
                file.Expression("stream->wait()");
            self._launcher = src.getvalue()

    def _get_block_dim_spec(self):
        super(GemmGenerator, self)._get_block_dim_spec()
        return f'threads{{{self.num_active_threads}, {self.num_mult_per_block}, 1}}'

    def _get_grid_dim_spec(self):
        super(GemmGenerator, self)._get_grid_dim_spec()
        num_blocks = "({0} + {1} - 1) / {1}".format(Generator.NUM_ELEMENTS_STR,
                                                    self.num_mult_per_block)
        return f'blocks{{{num_blocks}, 1, 1}}'