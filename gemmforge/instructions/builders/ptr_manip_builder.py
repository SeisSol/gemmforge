from gemmforge.basic_types import GeneralLexicon
from gemmforge.exceptions import InternalError
from gemmforge.instructions import GetElementPtr
from gemmforge.matrix.dense import DenseMatrix
from gemmforge.symbol_table import DataView, Symbol, SymbolType, TensorDataView
from gemmforge.tensor.dense import DenseTensor
from gemmforge.instructions.builders.abstract_builder import AbstractBuilder


class GetElementPtrBuilder(AbstractBuilder):
  def __init__(self, vm, symbol_table):
    super(GetElementPtrBuilder, self).__init__(vm, symbol_table)

  def build(self, src: Symbol, include_extra_offset: bool = True):
    self._reset()
    if src.stype == SymbolType.Batch:
      dest = Symbol(name=f'{GeneralLexicon.GLOBAL_MEM_PREFIX}{src.name}',
                    stype=SymbolType.Global,
                    obj=src.obj)

      obj = src.obj
      if isinstance(obj, DenseMatrix):
        dest.data_view = DataView(rows=obj.get_actual_num_rows(),
                                  columns=obj.get_actual_num_cols(),
                                  lead_dim=obj.leading_dimension,
                                  is_transposed=False)
      else:
        assert (isinstance(obj, DenseTensor))
        dest.data_view = TensorDataView(dimensions=obj.get_dimensions(), is_transposed=False)

      self._symbol_table.add_symbol(dest)
      self._instructions.append(GetElementPtr(self._vm, src, dest, include_extra_offset))
    elif src.stype == SymbolType.SharedMem:
      dest = Symbol(name=f'_{src.name}',
                    stype=SymbolType.SharedMem,
                    obj=src.obj)

      obj = src.obj
      if isinstance(obj, DenseMatrix):
        dest.data_view = DataView(rows=obj.num_rows,
                                  columns=obj.num_cols,
                                  lead_dim=obj.num_rows,
                                  is_transposed=False)
      elif isinstance(obj, DenseTensor):
        dest.data_view = TensorDataView(dimensions=obj.get_dimensions(), is_transposed=False)
      else:
        dest.data_view = DataView(rows=obj[1],
                                  columns=1,
                                  lead_dim=obj[1],
                                  is_transposed=False)

      self._symbol_table.add_symbol(dest)
      self._instructions.append(GetElementPtr(self._vm, src, dest, include_extra_offset))
    else:
      raise InternalError("EmelementPtrBuilder not Batch or ShrMem")

