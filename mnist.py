from nn import *
from dataloader import *
import pandas as pd
import kagglehub


def dump_mnist(dataset, index=None):
    # If index is provided, get the element at that index
    if index is not None:
        elem = dataset[index]
    else:
        # Otherwise, treat dataset as the element
        elem = dataset

    label = elem[0]
    image = elem[1:]

    # Convert image to a 28x28 pixel matrix
    pixels = list(image)
    pixels = [pixels[i:i + 28] for i in range(0, len(pixels), 28)]

    # Print label
    print(f"Label: {label}\n")

    # Pretty print the image using text symbols
    for row in pixels:
        row_str = ''.join(['#' if pixel > 128 else '.' for pixel in row])
        print(row_str)


def load_minst(suffix):
    # Download latest version
    data_path = kagglehub.dataset_download("oddrationale/mnist-in-csv")
    print("Path to dataset files:", data_path)

    csv_path = data_path + "/" + "mnist_" + suffix + ".csv"

    data_pd = pd.read_csv(csv_path)
    dataset = np.array(data_pd).astype(float)
    np.random.shuffle(dataset) # перемешаем датасет
    return dataset


dataset = load_minst("test")
dump_mnist(dataset, 0)
data_loader = DataLoader(dataset, batch_size=64, shuffle=True, flatten=True)


class SimpleNet(Module):
    def __init__(self):
        super().__init__()
        self.conv1 = Conv2d(input_channels = 1, output_channels = 4, kernel_size=5) #28 -> 24
        self.conv2 = Conv2d(input_channels = 4, output_channels = 8, kernel_size=5) #24 -> 20
        self.conv3 = Conv2d(input_channels = 8, output_channels = 16, kernel_size=5) #20 -> 16
        self.flatten = Flatten()
        self.linear1 = Linear(input_channels=16 * 16 * 16, output_channels=200, bias=True)
        self.linear2 = Linear(input_channels=200, output_channels=50, bias=True)
        self.linear3 = Linear(input_channels=50, output_channels=10, bias=True)
        self.relu = ReLU()

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.relu(x)
        x = self.flatten(x)
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        x = self.relu(x)
        x = self.linear3(x)
        return x


loss_fn = CrossEntropyLoss()
model = SimpleNet()
optim = Adam(model.parameters(), learning_rate = 0.001, momentum=0.9, ro=0.9)

for i in range(5):
    for index, batch in enumerate(data_loader):
        input_x, target = batch
        input_x = input_x / 255
        input_x = np.expand_dims(input_x, axis=1)  # (64, 28, 28) -> (64, 1, 28, 28)
        output = model(input_x)
        loss = loss_fn(output, target)
        loss.backward()
        optim.step()

        if index % 1 == 0:
            print(loss.loss.mean(), "index:", index)

    print(loss.loss.mean(), "epoch:", i)
