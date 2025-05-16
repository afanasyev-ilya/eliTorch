from nn import *

if __name__ == '__main__':
    np.random.seed(42)

    input_x = np.array([[0.99197708, -0.77980023, -0.8391331, -0.41970686, 0.72636492],
                        [0.85901409, -0.22374584, -1.95850625, -0.81685145, 0.96359871],
                        [-0.42707937, -0.50053309, 0.34049477, 0.62106931, -0.76039365],
                        [0.34206742, 2.15131285, 0.80851759, 0.28673013, 0.84706839],
                        [-1.70231094, 0.36473216, 0.33631525, -0.92515589, -2.57602677]])
    target_x = [1.0, 0.0]

    loss_fn = CrossEntropyLoss()
    model = SimpleNet()
    optim = SGD(model.parameters(), learning_rate=0.01)
    for i in range(100):
        output = model(input_x)
        loss = loss_fn(output, target_x)
        loss.backward()
        lr = 0.01
        optim.step()
        if (i % 20) == 0:
            print(loss.loss, i)

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
