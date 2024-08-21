from mindspore import Model, Tensor
from mindspore.train.callback import LossMonitor
from mindspore.train.serialization import load_checkpoint, save_checkpoint
from mindspore.common import dtype as mstype

import mindspore.dataset as ds
import mindspore.dataset.transforms.c_transforms as C
import mindspore.nn as nn
import mindspore.ops.operations as P



mnist_ds = ds.MnistDataset("/MNIST_Data", shuffle=True)
mnist_ds = mnist_ds.map(input_columns="label", operations=C.TypeCast(mstype.int32))
mnist_ds = mnist_ds.map(input_columns="image", operations=C.Normalize(mean=0.1307, std=0.3081))
mnist_ds = mnist_ds.map(input_columns="image", operations=C.HWC2CHW())
mnist_ds = mnist_ds.batch(32, drop_remainder=True)
print (mnist_ds.output_shapes)