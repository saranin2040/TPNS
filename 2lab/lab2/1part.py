import numpy as np

# Функция активации (сигмоид) и ее производная
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

# Инициализация параметров сети
input_size = 3  # 3 входных нейрона
hidden_size = 2  # 2 нейрона в скрытом слое
output_size = 1  # 1 выходной нейрон
expected=0
dWeights=0

np.random.seed(42)  # для воспроизводимости результатов

# Веса и смещения (биасы)
weights_input_hidden = np.random.rand(input_size, hidden_size)
bias_hidden = np.zeros((1, hidden_size))
weights_hidden_output = np.random.rand(hidden_size, output_size)
bias_output = np.zeros((1, output_size))
hidden_layer_input=[]
hidden_layer_output=[]
learnRate=0.1

weights=[]

# Прямое распространение (forward propagation)
def forward_pass(inputs):
    hidden_layer_input = np.dot(inputs, weights_input_hidden) 
    hidden_layer_output = sigmoid(hidden_layer_input)

    output_layer_input = np.dot(hidden_layer_output, weights_hidden_output) 
    output = sigmoid(output_layer_input)
    return output

def errorReversal(actual):
    error=actual-expected
    dWeights=error*actual*(1-actual)
    for i in range(len(weights)):
        for i in range(len(weights)):
        w=weights_hidden_output[0] - hidden_layer_output[0]*learnRate*dWeights
    weights_hidden_output[0]=w



target=forward_pass([0,1,0])

errorReversal(target)

print(target)

#print(weights_input_hidden)

# Это базовый каркас вашей нейронной сети. Для полноценного обучения необходимо также реализовать функцию потерь,
# обратное распространение и процесс обучения, который будет включать многократное выполнение прямого и обратного распространения.