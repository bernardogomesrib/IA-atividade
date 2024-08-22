import mindspore as ms
import mindspore.dataset as ds
import mindspore.nn as nn
from mindspore import Model, context
from mindspore.train.callback import Callback
import numpy as np

context.set_context(mode=context.GRAPH_MODE, device_target="CPU")

# Carregando os dados MNIST
images = np.load("mnist_images.npy")  # (60000, 28, 28)
labels = np.load("mnist_labels.npy")  # (60000,)

images = images.astype(np.float32) / 255.0
labels = labels.astype(np.int32)
images = np.expand_dims(images, axis=1)

def create_dataset(images, labels, batch_size=32, repeat_size=1):
    dataset = ds.NumpySlicesDataset({"image": images, "label": labels}, shuffle=True)
    dataset = dataset.batch(batch_size)
    dataset = dataset.repeat(repeat_size)
    return dataset

train_dataset = create_dataset(images, labels)

class Net(nn.Cell):
    def __init__(self):
        super(Net, self).__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Dense(28 * 28, 128)
        self.fc2 = nn.Dense(128, 64)
        self.fc3 = nn.Dense(64, 10)
        self.relu = nn.ReLU()
    
    def construct(self, x):
        x = self.flatten(x)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

network = Net()

loss_fn = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction='mean')
optimizer = nn.Adam(network.trainable_params(), learning_rate=0.001)
model = Model(network, loss_fn, optimizer, metrics={'accuracy'})

class AccuracyMonitor(Callback):
    def __init__(self):
        self.correct = 0
        self.total = 0

    def on_train_epoch_begin(self, run_context):
        self.correct = 0
        self.total = 0

    def on_train_step_end(self, run_context):
        cb_params = run_context.original_args()
        outputs = cb_params.net_outputs
        # Verifique se a saída tem a forma correta e se é um tensor
        if isinstance(outputs, ms.Tensor) and len(outputs.shape) > 1 and outputs.shape[-1] == 10:
            predictions = outputs.asnumpy().argmax(axis=1)
            labels = cb_params.batch_labels.asnumpy()
            self.correct += (predictions == labels).sum()
            self.total += labels.shape[0]
        else:
            print(f"Forma inesperada na saída do modelo: {outputs.shape}")

    def on_train_epoch_end(self, run_context):
        accuracy = self.correct / self.total if self.total > 0 else 0
        print(f'Epoch {run_context.original_args().cur_epoch_num}, Accuracy: {accuracy:.4f}')

class SilentLossMonitor(Callback):
    def on_train_step_end(self, run_context):
        pass

model.train(epoch=10, train_dataset=train_dataset, callbacks=[SilentLossMonitor(), AccuracyMonitor()])
