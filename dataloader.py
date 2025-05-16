import numpy as np


class DataLoader():
    def __init__(self, data, batch_size=64, shuffle=True, flatten=True):
        self.data = data
        self.index = 0  # индекс нужен для корректной итерации
        self.items = []  # здесь будем хранить наборы батчей
        self.flatten = flatten

        # определяем сколько раз сможем проитерировать весь набор
        self.max = (data.shape[0] - 1) // batch_size + 1

        if shuffle == True:
            # мешаем набор если захотел пользователь
            self.data = np.random.permutation(self.data)
        # создаем список всех батчей
        for _ in range(self.max):
            self.items.append(self.data[batch_size * _: batch_size * (_ + 1)])

        # превращаем в итерируемый объект
        def __iter__(self):
            return self

    # для циклов for и while
    def __next__(self):
        if self.index < self.max:
            value = self.items[self.index]  # получаем батч
            self.index += 1
            if self.flatten:
                # возвращаем в нашем случае либо ((64, 784), (64, 1))
                return value[:, 1:], value[:, 0].reshape(-1, 1)
            else:
                # либо ((64, 28, 28), (64, 1)), то есть как изображение
                return value[:, 1:].reshape(value.shape[0], 28, 28), value[:, 0].reshape(-1, 1)
        else:
            self.index = 0
            raise StopIteration  # выходим из цикла

    # Define __iter__ to make the class iterable
    def __iter__(self):
        return self

    def __len__(self):
        return len(self.items)