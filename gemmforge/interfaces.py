from .matrix import DenseMatrix
from .tensor import DenseTensor


class YatetoInterface:
  def __init__(self):
    pass

  @classmethod
  def deduce_bbox(cls, yateto_ranges, mem_layout, transpose):
    """Converts yateto memory layout (bounding boxes) and ranges to GemmForge bounding boxes i.e.,
       a box is a list of rows and columns indices where the actual data is located within
       a memory patch and should be computed

    Args:
      yateto_ranges (set[loopRanges]): a range of rows and columns to operate on
      mem_layout (BoundingBox): memory layout given as yateto bounding box
      is_trans (bool): if true then a GemmForge bonding box needs to be transposed

    Returns:
      (list): bounding box in GemmForge format
    """
    if transpose:
      last, first = yateto_ranges
    else:
      first, last = yateto_ranges

    return [first.start - mem_layout[0].start,
            last.start - mem_layout[1].start,
            first.stop - mem_layout[0].start,
            last.stop - mem_layout[1].start]

  @classmethod
  def deduce_bbox_tensor(cls, yateto_ranges, mem_layout, transpose):
    """Converts yateto memory layout (bounding boxes) and ranges to GemmForge bounding boxes i.e.,
       a box is a list of rows and columns indices where the actual data is located within
       a memory patch and should be computed

    Args:
      yateto_ranges (set[loopRanges]): a range of rows and columns to operate on
      mem_layout (BoundingBox): memory layout given as yateto bounding box
      is_trans (bool): if true then a GemmForge bonding box needs to be transposed

    Returns:
      (list): bounding box in GemmForge format
    """
    if transpose:
      raise Exception("Support for transposed tensors not yet implemented")

    bbox = list()
    for i in range(len(yateto_ranges)):
      bbox.append(yateto_ranges[i].start - mem_layout[i].start)
    for i in range(len(yateto_ranges)):
      bbox.append(yateto_ranges[i].stop - mem_layout[i].start)

    return bbox

  @classmethod
  def produce_dense_matrix(cls,
                           yateto_ranges,
                           yateto_memory_layout_bbox,
                           addressing,
                           transpose,
                           leading_dimension=None):

    gemmforge_bbox = cls.deduce_bbox(yateto_ranges=yateto_ranges,
                                     mem_layout=yateto_memory_layout_bbox,
                                     transpose=transpose)

    return DenseMatrix(num_rows=yateto_memory_layout_bbox[0].stop,
                       num_cols=yateto_memory_layout_bbox[1].stop,
                       addressing=addressing,
                       bbox=gemmforge_bbox,
                       leading_dimension=leading_dimension)

  @classmethod
  def produce_dense_tensor(cls, yateto_ranges, yateto_memory_layout_bbox, addressing, transpose):
    print("i1: ", yateto_ranges)
    print("i2: ", yateto_memory_layout_bbox)
    if isinstance(yateto_ranges, int):
      raise Exception("uwu")
    gemmforge_bbox = cls.deduce_bbox_tensor(yateto_ranges=yateto_ranges,
                                            mem_layout=yateto_memory_layout_bbox,
                                            transpose=transpose)
    dimensions = [layout.stop for layout in yateto_memory_layout_bbox]
    return DenseTensor(dimensions=dimensions,
                       addressing=addressing,
                       bbox=gemmforge_bbox)
