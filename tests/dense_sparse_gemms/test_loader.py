import functools
import os
from copy import deepcopy
from itertools import product
from random import randint

import numpy as np

from gemmforge import DenseMatrix, SparseMatrix

first = True
writes = 0
random_coordinates = True
random_coo1 = list()
random_coo2 = list()


def gen_matrix_b(rowB, colB, transposed, btype):
  coo = {"name": "B", "rows": rowB, "cols": colB, "entries": [], "coordinates": []}

  if btype == "band_diagonal":
    if not transposed:
      coo["entries"].append([0, 0, 2.0])
      coo["coordinates"].append([0, 0])
      coo["entries"].append([1, 0, 1.0])
      coo["coordinates"].append([1, 0])
      for i in range(1, rowB - 1):
        coo["entries"].append([i - 1, i, 3.0])
        coo["coordinates"].append([i - 1, i])
        coo["entries"].append([i, i, 2.0])
        coo["coordinates"].append([i, i])
        coo["entries"].append([i + 1, i, 1.0])
        coo["coordinates"].append([i + 1, i])
      i = rowB - 1
      coo["entries"].append([i - 1, i, 3.0])
      coo["coordinates"].append([i - 1, i])
      coo["entries"].append([i, i, 2.0])
      coo["coordinates"].append([i, i])
    else:
      coo["entries"].append([0, 0, 2.0])
      coo["coordinates"].append([0, 0])
      coo["entries"].append([0, 1, 3.0])
      coo["coordinates"].append([0, 1])
      for i in range(1, rowB - 1):
        coo["entries"].append([i, i - 1, 1.0])
        coo["coordinates"].append([i, i - 1])
        coo["entries"].append([i, i, 2.0])
        coo["coordinates"].append([i, i])
        coo["entries"].append([i, i + 1, 3.0])
        coo["coordinates"].append([i, i + 1])
      i = rowB - 1
      coo["entries"].append([i, i - 1, 1.0])
      coo["coordinates"].append([i, i - 1])
      coo["entries"].append([i, i, 2.0])
      coo["coordinates"].append([i, i])
  elif btype == "single_column":
    at = 1
    for i in range(rowB):
      coo["entries"].append([i, at, 4.0])
      coo["coordinates"].append([i, at])
  elif btype == "single_row":
    at = 1
    for j in range(colB):
      coo["entries"].append([at, j, 4.0])
      coo["coordinates"].append([at, j])
  elif btype == "full":
    if transposed:
      for i in range(colB):
        for j in range(rowB):
          coo["entries"].append([i, j, 8.0])
          coo["coordinates"].append([i, j])
    else:
      for j in range(colB):
        for i in range(rowB):
          coo["entries"].append([i, j, 8.0])
          coo["coordinates"].append([i, j])
  elif btype == "chequered":
    npB = np.zeros((rowB, colB))
    if transposed:
      for i in range(rowB):
        offset = i % 2
        for j in range(offset, colB, 2):
          coo["entries"].append([i, j, 9.0])
          coo["coordinates"].append([i, j])
          npB[i, j] = i * 10 + j
    else:
      for j in range(colB):
        offset = j % 2
        for i in range(offset, rowB, 2):
          coo["entries"].append([i, j, 9.0])
          coo["coordinates"].append([i, j])
          npB[i, j] = i * 10 + j
  elif btype == "random":
    global random_coo1
    global random_coo2
    global first
    global writes
    entry_count = int(0.25 * rowB * colB)
    l = set()
    while len(l) < entry_count:
      i = randint(0, 8)
      j = randint(0, 8)
      l.add((i, j))
    llist = list(l)
    if transposed:
      llist.sort(key=lambda x: x[0] * colB + x[1])
    else:
      llist.sort(key=lambda x: x[0] + x[1] * rowB)

    if writes == 0:
      random_coo1 = (list(), list())
      for i, j in llist:
        random_coo1[0].append([i, j, 11.0])
        random_coo1[1].append([i, j])
      coo["entries"] = random_coo1[0]
      coo["coordinates"] = random_coo1[1]
    elif writes == 2:
      random_coo2 = (list(), list())
      for i, j in llist:
        random_coo2[0].append([i, j, 11.0])
        random_coo2[1].append([i, j])
      coo["entries"] = random_coo2[0]
      coo["coordinates"] = random_coo2[1]
    elif writes % 4 == 0 or writes % 4 == 1:
      coo["entries"] = random_coo1[0]
      coo["coordinates"] = random_coo1[1]
    else:
      coo["entries"] = random_coo2[0]
      coo["coordinates"] = random_coo2[1]

    if writes == 0:
      if transposed:
        raise Exception("The B-sparsity parameters should be exactly [False, True]")

      if not os.path.exists("gen_code"):
        os.mkdir("gen_code")

      with open("gen_code/coordinate_vector.cpp", "w") as f:
        f.write("#include <vector>\n")
        f.write("#include <tuple>\n")
        f.write("std::vector<std::tuple<int, int>> get_coordinates_B_core()\n")
        f.write("{\n")
        f.write("std::vector<std::tuple<int, int>> coordinates;\n")
        for (i, j) in llist:
          f.write(f"coordinates.push_back(std::make_tuple({i}, {j}));\n")
        f.write("return coordinates;\n")
        f.write("}\n")
        first = False
      writes += 1
    elif writes == 2:
      if not transposed:
        raise Exception("The B-sparsity parameters should be exactly [False, True]")
      with open("gen_code/coordinate_vector.cpp", "a") as f:
        f.write("std::vector<std::tuple<int, int>> get_coordinates_B_core_transposed()\n")
        f.write("{\n")
        f.write("std::vector<std::tuple<int, int>> coordinates;\n")
        for (i, j) in llist:
          f.write(f"coordinates.push_back(std::make_tuple({i}, {j}));\n")
        f.write("return coordinates;\n")
        f.write("}\n")
      writes += 1
    elif writes % 4 == 0 or writes % 4 == 1:
      writes += 1
    else:
      writes += 1
  else:
    raise Exception("NO")
  return coo


class LoaderError(Exception):
  pass


class TestLoader:
  def __init__(self, test_spec):
    self._test_spec = test_spec
    self._param_iterator = None
    self._analyze()

  def __iter__(self):
    return self

  def __next__(self):
    test_params = next(self._param_iterator)
    spec = deepcopy(self._test_spec)
    for param in test_params:
      _set_value(spec, param, test_params[param])

    if spec["matrix_b"]["sparse"]:
      dense, sparse = self._produce_matrix(spec["matrix_b"], spec)
      return (spec["trans_a"],
              spec["trans_b"],
              self._produce_matrix(spec["matrix_a"], spec),
              dense,
              sparse,
              self._produce_matrix(spec["matrix_c"], spec),
              spec["alpha"],
              spec["beta"],
              spec["num_elements"],
              spec["matrix_b"]["matrix_type"],
              self._gen_test_name(test_params),
              spec["kernel_type"])

    else:
      dense = self._produce_matrix(spec["matrix_b"], spec)

      return (spec["trans_a"],
              spec["trans_b"],
              self._produce_matrix(spec["matrix_a"], spec),
              dense,
              self._produce_matrix(spec["matrix_c"], spec),
              spec["alpha"],
              spec["beta"],
              spec["num_elements"],
              spec["matrix_b"]["matrix_type"],
              self._gen_test_name(test_params),
              spec["kernel_type"])

  def _produce_matrix(self, matrix_spec, spec):
    if matrix_spec["sparse"]:
      coo = gen_matrix_b(matrix_spec["rows"], matrix_spec["cols"], spec["trans_b"], matrix_spec["matrix_type"])
      sparse = SparseMatrix(num_rows=matrix_spec["rows"],
                            num_cols=matrix_spec["cols"],
                            addressing=matrix_spec["addressing"],
                            coordinates=coo["coordinates"],
                            values=None)
      dense = DenseMatrix(num_rows=matrix_spec["rows"],
                          num_cols=matrix_spec["cols"],
                          addressing=matrix_spec["addressing"],
                          bbox=[0, 0, matrix_spec["rows"], matrix_spec["cols"]])
      return (dense, sparse)
    else:
      dense = DenseMatrix(num_rows=matrix_spec["rows"],
                          num_cols=matrix_spec["cols"],
                          addressing=matrix_spec["addressing"],
                          bbox=matrix_spec["bbox"])
      return dense

  def is_param(self, param):
    if isinstance(param, str):
      if param.find('param') != -1:
        return True
      else:
        return False
    else:
      return False

  def _analyze(self):
    flatten_spec = {}
    _build_flatten_table(flatten_spec, self._test_spec)

    params = {}
    for item in flatten_spec:
      if (self.is_param(flatten_spec[item])):
        params[item] = self._test_spec[flatten_spec[item]]

    self._param_iterator = (dict(zip(params, x)) for x in product(*params.values()))

  def _gen_test_name(self, params):
    param_to_str = []
    for item in params:
      item_str = "_".join(item)
      value_str = params[item]
      if isinstance(params[item], float):
        value_str = str(params[item]).replace('.', '_')
      if isinstance(params[item], list):
        value_str = [str(item) for item in params[item]]
        value_str = "_".join(value_str)
      param_to_str.append("{}_{}".format(item_str, value_str))

    return "{}_{}".format(self._test_spec['test_base_name'], "_".join(param_to_str))


def _build_flatten_table(flatten_table, original_table, combo_key=()):
  if isinstance(original_table, dict):
    for key in original_table:
      _build_flatten_table(flatten_table, original_table[key], (*combo_key, key))
  else:
    flatten_table[combo_key] = original_table


def _set_value(table, combo_key, value):
  if len(combo_key) == 1:
    table[combo_key[0]] = value
  else:
    _set_value(table[combo_key[0]], combo_key[1:], value)


def _get_value(table, combo_key):
  return functools.reduce(dict.get, combo_key, table)
