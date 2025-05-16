import numpy as np
import scipy

Parameter = None

def ParameterObj():
    class Parameter:
        layers = []
        calling = dict()

        def __init__(self, info):
            Parameter.layers.append(info[0])
            Parameter.calling[info[0]] = info[1:]

    return Parameter


class Module:
    def __init__(self):
        self._constructor_Parameter = ParameterObj()
        global Parameter
        Parameter = self._constructor_Parameter

    def forward(self):
        pass

    def __call__(self, x):
        return self.forward(x)

    def parameters(self):
        return self

class Linear:
    def __init__(self, input_channels: int, output_channels: int, bias=True):
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.bias = bias
        self.backward_list = []
        if bias:
            Parameter([self, np.random.uniform(- 0.5, 0.5, size=(self.input_channels, self.output_channels)),
                       np.random.uniform(- 0.5, 0.5, size=self.output_channels)])
        else:
            Parameter([self, np.random.uniform(- 0.5, 0.5, size=(self.input_channels, self.output_channels)),
                       np.zeros(self.output_channels)])

    def __call__(self, x):
        self.x = np.array(x, copy=True)
        result = x @ Parameter.calling[self][0] + Parameter.calling[self][1]
        return result

    def backward(self, input_matrix):
        x_gradient = input_matrix @ self.weight.T
        self.weight_gradient = self.x.T @ input_matrix
        self.bias_gradient = input_matrix.mean(axis=0)
        return x_gradient


class Conv2d:

    def __init__(self, input_channels: int, output_channels: int, kernel_size: int, bias = True):
        self.bias_flag = bias
        self.kernel_size = (output_channels, input_channels, kernel_size, kernel_size)
        self.filter_array = np.random.randn(1, *self.kernel_size) * 0.1
        self.bias = np.random.randn(1, output_channels, 1, 1) * 0.1
        Parameter([self, self.filter_array, self.bias * bias])


    def __call__(self, x):
        self.x = np.expand_dims(x, axis=1) # сохраняем на всякий случай значение
        # x: (batch, channels, height, width) - > (batch, 1, channels, height, width)
        return scipy.signal.fftconvolve(np.expand_dims(x, axis=1), self.filter_array, mode='valid').squeeze(axis=2) + self.bias


class Flatten:
    def __init__(self):
        pass

    def __call__(self, x):
        return x.reshape(1, -1)


class ReLU:
    def __init__(self):
        pass
        #Parameter([self, []])

    def __call__(self, x):
        self.x = x
        return np.maximum(0, x)

    def backward(self, input_matrix):
        return (self.x > 0) * input_matrix


class Softmax():
    def __init__(self):
        pass

    def __call__(self, z):
        return np.exp(z) / np.sum(np.exp(z), axis=1).reshape(-1, 1)


import skimage

class MaxPool2d:
    def __init__(self, kernel_size: tuple):
        self.kernel_size = kernel_size
        Parameter([self, []])

    def __call__(self, x):
	    self.x = np.array(x, copy=True)
	    return skimage.measure.block_reduce(a, (1, 1, *self.kernel_size), np.max)

class MinPool2d:
    def __init__(self, kernel_size: tuple):
        self.kernel_size = kernel_size
        Parameter([self, []])

    def __call__(self, x):
	    self.x = np.array(x, copy=True)
	    return skimage.measure.block_reduce(a, (1, 1, *self.kernel_size), np.min)


class CrossEntropyLoss:

    def __init__(self, l1_reg=0, l2_reg=0):
        self.l1_reg = l1_reg
        self.l2_reg = l2_reg
        self.predicted = None
        self.true = None

    def __call__(self, logits, true):
        predicted = np.exp(logits) / np.sum(np.exp(logits), axis=1).reshape(-1, 1)  # softmax
        self.predicted = np.array(predicted, copy=True)  # сделаем копию входных матрицы для дальнейших вычислений
        number_of_classes = predicted.shape[1]
        self.true = np.int_(np.arange(0, number_of_classes) == true)
        # вычисляем значение лосс-функции прямо по формуле
        self.loss = -1 * np.sum(self.true * np.log(self.predicted + 1e-5), axis=1)
        return self

    def backward(self):
        loss = self.predicted - self.true
        # Итерируем по каждому слою в обратном порәдке, благодаря тому, что мы всё сохранили в Parameter.layers
        for index, layer in enumerate(Parameter.layers[::-1]):
            if type(layer).__name__ == 'Linear':
                base = (layer.x.T @ loss) / loss.shape[0]
                l1_term = self.l1_reg * np.sign(Parameter.calling[layer][0])
                l2_term = self.l2_reg * Parameter.calling[layer][0]
                changes_w = base + l1_term + l2_term
                # нормировка на loss.shape[0] нужна, так как величина изменений зависит от размера батча
                if layer.bias:
                    changes_b = (np.sum(loss) / loss.shape[0])
                else:
                    changes_b = 0
                layer.backward_list = [changes_w, changes_b]
                # Cчитаем градиент для следующих слоев
                loss = loss @ Parameter.calling[layer][0].T

            elif type(layer).__name__ == 'Conv2d':
                expanded_loss = np.expand_dims(loss, axis=2)
                changes_w = scipy.signal.fftconvolve(layer.x, expanded_loss, mode='valid')
                changes_b = loss.sum(axis=(0, 2, 3)) * layer.bias_flag

                l1_term = self.l1_reg * np.sign(Parameter.calling[layer][0])
                l2_term = self.l2_reg * Parameter.calling[layer][0]

                changes_w += l1_term + l2_term

                layer.backward_list = [changes_w, changes_b]

                rotated_filters = np.transpose(layer.filter_array, (0, 1, 2, 4, 3))
                pad = layer.kernel_size[-1]
                padded_loss = np.pad(expanded_loss, ((0, 0), (0, 0), (0, 0), (pad - 1, pad - 1), (pad - 1, pad - 1)),
                                     'constant', constant_values=(0))

                loss = scipy.signal.fftconvolve(padded_loss, rotated_filters, mode='valid').squeeze(axis=1)

            elif type(layer).__name__ in ('ReLU', 'Flatten'):
                loss = layer.backward(loss)

            elif type(layer).__name__ == 'MaxPool2d':
                shapes = layer.x.shape
                new_shape = np.zeros(shapes)
                for k in range(shapes[0]):
                    for m in range(shapes[1]):
                        inx_ = 0
                        inx__ = 0
                        layer.i = 0
                        while layer.i < layer.x[k][m].shape[0] - layer.kernel_size[0] + 1:
                            layer.j = 0
                            inx__ = 0
                            while layer.j < layer.x[k][m].shape[1] - layer.kernel_size[1] + 1:
                                new_shape[k][m][layer.i:layer.i + layer.kernel_size[0],
                                layer.j:layer.j + layer.kernel_size[1]] = \
                                    loss[k][m][inx_][inx__]
                                inx__ += 1
                                layer.j += layer.kernel_size[1]

                            inx_ += 1
                            layer.i += layer.kernel_size[0]

                loss = np.squeeze([layer.x > 0] * new_shape, axis=0)

            elif type(layer).__name__ == 'MinPool2d':
                shapes = layer.x.shape
                new_shape = np.zeros(shapes)
                for k in range(shapes[0]):
                    for m in range(shapes[1]):
                        inx_ = 0
                        inx__ = 0
                        layer.i = 0
                        while layer.i < layer.x[k][m].shape[0] - layer.kernel_size[0] + 1:
                            layer.j = 0
                            inx__ = 0
                            while layer.j < layer.x[k][m].shape[1] - layer.kernel_size[1] + 1:
                                new_shape[k][m][layer.i:layer.i + layer.kernel_size[0],
                                layer.j:layer.j + layer.kernel_size[1]] = \
                                    loss[k][m][inx_][inx__]
                                inx__ += 1
                                layer.j += layer.kernel_size[1]

                            inx_ += 1
                            layer.i += layer.kernel_size[0]

                loss = np.squeeze([layer.x > 0] * new_shape, axis=0)

            elif type(layer).__name__ == 'BatchNorm2d':
                layer.backward_list = [np.sum(loss * layer.x, axis=0), np.sum(loss, axis=0)]
                dl_dx = loss * Parameter.calling[layer][0]
                dl_dstd = np.sum(dl_dx * (layer.x * layer.std) * (-1 / 2) / (layer.std ** 3), axis=0)
                dl_dmean = np.sum(dl_dx * (- 1 / layer.std), axis=0) + dl_dstd * (
                            np.sum(-2 * layer.x * layer.std, axis=0) / len(layer.x))
                loss = dl_dx / layer.std + dl_dstd * 2 * (layer.x * layer.std) / len(layer.x) + dl_dmean / len(layer.x)


class SGD:
    def __init__(self, model, learning_rate):
        self.model = model
        self.lr = learning_rate

    def step(self):
        for index, layer in enumerate(self.model._constructor_Parameter.layers[::-1]):
            if type(layer).__name__ == 'Linear':
                weight, bias = self.model._constructor_Parameter.calling[layer]
                weight_gradient, bias_gradient = layer.backward_list[0], layer.backward_list[1]
                self.model._constructor_Parameter.calling[layer] = [weight - self.lr * weight_gradient,
                                                                    bias - self.lr * bias_gradient]


class Adam:
    def __init__(self, model, learning_rate, momentum, ro):
        self.model = model
        self.lr = learning_rate
        self.momentum = momentum
        self.last_grad_w = None
        self.last_grad_b = None

        if ro < 0 or ro > 1:
            raise Exception("Incorrect ro value")

        self.ro = ro
        self.grad_velocity_w = None
        self.grad_velocity_b = None

    def step(self):

        if self.last_grad_w == None:
            self.last_grad_w = [0] * len(self.model._constructor_Parameter.layers)
            self.last_grad_b = [0] * len(self.model._constructor_Parameter.layers)
            self.grad_velocity_w = [0] * len(self.model._constructor_Parameter.layers)
            self.grad_velocity_b = [0] * len(self.model._constructor_Parameter.layers)

        for index, layer in enumerate(self.model._constructor_Parameter.layers[::-1]):
            if type(layer).__name__ in ('Linear', 'Conv2d', 'BatchNorm2d'):
                weight, bias = self.model._constructor_Parameter.calling[layer]
                weight_gradient, bias_gradient = layer.backward_list[0], layer.backward_list[1]

                self.grad_velocity_w[index] = self.ro * self.grad_velocity_w[index] + (
                            1 - self.ro) * weight_gradient ** 2
                self.grad_velocity_b[index] = self.ro * self.grad_velocity_b[index] + (1 - self.ro) * bias_gradient ** 2

                self.last_grad_w[index] = - self.lr * weight_gradient + self.momentum * self.last_grad_w[index]
                self.last_grad_b[index] = - self.lr * bias_gradient + self.momentum * self.last_grad_b[index]

                new_weight = weight + self.last_grad_w[index] / np.sqrt(self.grad_velocity_w[index] + 1e-5)
                new_bias = bias + self.last_grad_b[index] / np.sqrt(self.grad_velocity_b[index] + 1e-5)

                self.model._constructor_Parameter.calling[layer] = [new_weight, new_bias]


class SimpleNet(Module):
    def __init__(self):
        super().__init__()
        self.linear1 = Linear(input_channels=25, output_channels=10, bias=True)
        self.linear2 = Linear(input_channels=10, output_channels=2, bias=True)
        self.flatten = Flatten()
        self.relu = ReLU()
        self.softmax = Softmax()

    def forward(self, x):
        x_1 = self.flatten(x)
        x_2 = self.linear1(x_1)
        x_3 = self.relu(x_2)
        x_4 = self.linear2(x_3)
        return x_4

