import mindspore as ms
import mindspore.dataset as ds
import mindspore.nn as nn
from mindspore import Model, context
from mindspore.train.callback import Callback
import numpy as np

# Definindo o contexto
context.set_context(mode=context.GRAPH_MODE, device_target="CPU")

# Carregar os dados MNIST
images = np.load("mnist_images.npy")  # (60000, 28, 28)
labels = np.load("mnist_labels.npy")  # (60000,)

# Normalizando as imagens para o intervalo [0, 1]
images = images.astype(np.float32) / 255.0
labels = labels.astype(np.int32)

# Expandindo as dimensões das imagens para (60000, 1, 28, 28) para adequar ao formato esperado
images = np.expand_dims(images, axis=1)

# Criar o dataset MindSpore
def create_dataset(images, labels, batch_size=32, repeat_size=1):
    dataset = ds.NumpySlicesDataset({"image": images, "label": labels}, shuffle=True)
    dataset = dataset.batch(batch_size)
    dataset = dataset.repeat(repeat_size)
    return dataset

# Criando o dataset
train_dataset = create_dataset(images, labels)

# Definindo a arquitetura da rede neural
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

# Instanciando o modelo
network = Net()

# Definir a função de perda e o otimizador
loss_fn = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction='mean')
optimizer = nn.Adam(network.trainable_params(), learning_rate=0.001)

# Preparando o modelo
model = Model(network, loss_fn, optimizer, metrics={'accuracy'})

# Callback personalizado para acumular e imprimir a acurácia no final de cada epoch
class AccuracyMonitor(Callback):
    def __init__(self):
        self.epoch_accuracy = 0.0
        self.num_batches = 0

    def epoch_begin(self, run_context):
        self.epoch_accuracy = 0.0
        self.num_batches = 0

    def step_end(self, run_context):
        cb_params = run_context.original_args()
        self.epoch_accuracy += cb_params.net_outputs.asnumpy()  # Acumulando a acurácia para cada batch
        self.num_batches += 1

    def epoch_end(self, run_context):
        final_accuracy = self.epoch_accuracy / self.num_batches  # Calculando a acurácia média
        print(f'Epoch {run_context.original_args().cur_epoch_num}, Accuracy: {1 - final_accuracy:.4f}')

# Removendo a impressão do LossMonitor para evitar mensagens indesejadas
class SilentLossMonitor(Callback):
    def step_end(self, run_context):
        pass

# Treinando o modelo
model.train(epoch=10, train_dataset=train_dataset, callbacks=[SilentLossMonitor(), AccuracyMonitor()])
