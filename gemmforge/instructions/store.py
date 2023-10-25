from gemmforge.exceptions import InternalError
from gemmforge.symbol_table import Symbol, SymbolType
from gemmforge.vm import VM
from .abstract_instruction import AbstractInstruction


class StoreRegToGlb(AbstractInstruction):
    def __init__(self,
                 vm: VM,
                 dest: Symbol,
                 src: Symbol,
                 alpha: float,
                 beta: float,
                 num_threads: int):
        super(StoreRegToGlb, self).__init__(vm)

        # if dest.stype != SymbolType.Global:
        #  raise InternalError('store: operand `dest` is not in glb mem.')

        if src.stype != SymbolType.Register:
            raise InternalError('store: operand `src` is not a register obj')

        self._dest = dest
        self._src = src
        self._alpha = alpha
        self._beta = beta
        self._num_threads = num_threads
        self._is_ready = True

    def gen_code(self, writer):
        dest_matrix = self._dest.obj
        dest_name = self._dest.name
        precision = self._vm.fp_as_str()

        with writer.If(self.gen_mask_threads(self._num_threads)):
            if dest_matrix.get_actual_num_cols() > 1:
                writer.Pragma("unroll")
                writer.For(
                    f'int n = 0; n < {dest_matrix.get_actual_num_cols()}; ++n').__enter__()
                rhs = "{}[{} + {} * n]".format(dest_name,
                                               self._vm.get_lexic().thread_idx_x,
                                               dest_matrix.leading_dimension)
            else:
                rhs = "{}[{}]".format(dest_name,
                                      self._vm.get_lexic().thread_idx_x)

            real_suffix = 'f' if precision == "float" else ''

            src_access = '' if self._src.obj.size == 1 else '[n]'
            if not isinstance(self._alpha, float):
                lhs = f'{self._alpha} * {self._src.name}{src_access}'
            else:
                if self._alpha == 1.0:
                    lhs = f'{self._src.name}{src_access}'
                else:
                    lhs = f'{self._alpha}{real_suffix} * {self._src.name}{src_access}'

            if not isinstance(self._beta, float):
                lhs += f' + {self._beta} * {rhs}'
            else:
                if self._beta != 0.0:
                    if self._beta == 1.0:
                        lhs += f' + {rhs}'
                    else:
                        const = f'{self._beta}{real_suffix}'
                        lhs += f' + {const} * {rhs}'

            writer(f'{rhs} = {lhs};')

            if dest_matrix.get_actual_num_cols() > 1:
                writer.For("").__exit__(None, None, None)

    def __str__(self) -> str:
        return 'not implemented'


class StoreRegToGlbTensor(AbstractInstruction):
    def __init__(self,
                 vm: VM,
                 dest: Symbol,
                 src: Symbol,
                 alpha: float,
                 beta: float,
                 num_threads: int):
        super(StoreRegToGlbTensor, self).__init__(vm)

        # if dest.stype != SymbolType.Global:
        #  raise InternalError(f'store: operand `dest` is not in glb mem: {dest}')

        if src.stype != SymbolType.Register:
            raise InternalError(
                f'store: operand `src` is not a register obj: {src}')

        self._dest = dest
        self._src = src
        self._num_threads = num_threads
        self._is_ready = True
        self._alpha = alpha
        self._beta = beta

    def gen_code(self, writer):
        # Important this class assumes that we store the ith row of tensor in a register,
        # Register may be longer than than the row of matrix C if temporary results require
        # But it will never be shorter
        dest_tensor = self._dest.obj
        dest_name = self._dest.name
        precision = self._vm.fp_as_str()

        with writer.If(self.gen_mask_threads(int(self._num_threads))):
            real_suffix = 'f' if precision == "float" else ''
            thread_idx_x = self._vm.get_lexic().thread_idx_x,

            # Calculating the begin of the threadIdx.x'th row of tensor
            # Between 2 rows of a matrix, the distance is 1
            # But assume tensor of rank 3 (N1xN2xN3), assume it is matrices stored
            # back to back
            # First N rows have distance 1, but,
            # To the second N rows we have N1x(N2-1) elements (column major storage)
            # Assume (N1xN2xN3x2) then between the rows of rank 3 and rank 4,#
            # We have N1x(N2-1) and 2o on.

            # i'th row is we need to get where it is located
            # if i < N1, or N1 < i < N1*N2, or N1*N2 < i  < N1*N2*N3
            rank_offsets = list()
            example = 7
            i = len(self._dest.obj.get_elements_needed_for_dimensions_skip())
            rowsLeftIsConst = ""
            if i == 1:
                rowsLeftIsConst = "const "
            #writer(f"{rowsLeftIsConst}int rowsLeft = " + thread_idx_x[0] + ";")
            for cell_count in reversed(self._dest.obj.get_elements_needed_for_dimensions_skip()):
                rank_offset = "rowsLeft" + " / " + str(cell_count)
                print(f"rankOffset{i} = ", example // cell_count)
                #writer(f"const int rankOffset{i} = {rank_offset};")
                if i > 1:
                  #writer(f"rowsLeft -= rankOffset{i} * {cell_count};")
                  pass
                example -= (example // cell_count) * cell_count
                print(f"rowsLeft = ", example)
                rank_offsets.append((f"rankOffset{i}", cell_count))
                i -= 1

            #writer.Emptyline()
            s = ""
            rowOffsetStr = "const int rowOffsetForCurrentRow = "
            for rank_offset, multiplier in reversed(rank_offsets):
                s += rank_offset + " * " + str(multiplier) + " + "
                accumulated_dimensions = self._dest.obj.get_accumulated_dimensions()
                print(accumulated_dimensions)
                print(rank_offsets)
            rowOffsetStr += str(int(self._dest.obj.get_accumulated_dimensions()
                                [-1]/self._dest.obj.get_accumulated_dimensions()[1])) + ";"
            #writer(rowOffsetStr)
            s = s[:-2]
            #writer.Emptyline()

            dims = self._dest.obj.get_dimensions()
            acc_dims = self._dest.obj.get_accumulated_dimensions()
            cols = dims[1]
            writer(f"int rows_left = {thread_idx_x[0]};")
            if (len(dims) >= 2):
                for i in range(len(dims)-1, -1, -1):
                    if i == 1:
                        #s1 = f"const int row_offset_{i-1} = rows_left;"
                        #s2 = ""
                        #s3 = f"const int dim_offset_{dest_indices[i-1]} = row_offset_{i-1};"
                        s1 = ""
                        s2 = ""
                        #s3 = ""
                    elif i == 0:
                        #s1 = f"const int row_offset_{i} = rows_left % {dims[0]};"
                        s1 = f"const int row_offset_{i} = rows_left;"
                        #s2 = f"rows_left -= row_offset_{i} * {acc_dims[i]//dims[1]}; // should be 0"
                        #s3 = f"const int dim_offset_{dest_indices[i]} = row_offset_{i};"
                        s2 = ""
                    else:
                        s1 = f"const int row_offset_{i-1} = rows_left / {acc_dims[i]//dims[1]};"
                        s2 = f"rows_left -= row_offset_{i-1} * {acc_dims[i]//dims[1]};"
                        #s3 = f"const int dim_offset_{dest_indices[i]} = row_offset_{i-1};"
                    writer(s1)
                    writer(s2)
                    #writer(s3)
            else:
                s1 = f"const int row_offset_{i-1} = rows_left;"
                #s3 = f"const int dim_offset_{dest_indices[i-1]} = row_offset_{i-1};"
                writer(s1)
                #writer(s3)
            access_str = ""
            if (len(dims)>= 2):
              for i in range(len(dims)):
                if i > 1:
                  access_str += f"row_offset_{i-1} * {acc_dims[i]} + "
                elif i < 1:
                  access_str += f"row_offset_{i} * {acc_dims[i]} + "
                else:
                  continue
            else:
              access_str += f"row_offset_{i} + "

            writer.Pragma("unroll")
            with writer.For(f'int i = 0; i < {cols}; ++i'):
                betaadd = "" if self._beta == 0.0 else f" + {self._beta} * {self._dest.name}[{access_str}i * {dims[0]}]"
                alphamul = "" if self._alpha == 1.0 else f"{self._alpha} * "
                if int(cols) > 1:
                    writer(
                        f"{self._dest.name}[{access_str}i * {dims[0]}] = {alphamul}{self._src.name}[i]{betaadd};")
                else:
                    writer(
                        f"{self._dest.name}[{access_str}i * {dims[0]}] = {alphamul}{self._src.name}{betaadd};")
    def __str__(self) -> str:
        return 'not implemented'
