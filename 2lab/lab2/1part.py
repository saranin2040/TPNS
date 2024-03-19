import numpy as np
import pandas as pd



class Neuron:

    def __init__(self,val=None,scolarVal=None,countWeight=None,countPrevLayer=None):
        if countWeight is not None and val is None and scolarVal is None:
            self.arrayWeight = self.fillArray(countWeight,countPrevLayer)
        elif val is not None and scolarVal is not None and countWeight is None:
            self.val=val
            self.scolarVal=scolarVal
    
    def fillArray(self, countWeight,countPrevLayer):
        limit = np.sqrt(6 / (countWeight + countPrevLayer))
        return np.random.uniform(-limit, limit, size=(countWeight,))
        #array = np.random.rand(countWeight)
        #return array

    val = None
    scolarVal=None
    arrayWeight = None
    weightsD=None


class Layers:

    def __init__(self,atrbs,countHidden,fun,funD):
        
        self.activeFun=fun
        self.activeFunD=funD

        layer = [Neuron(scolarVal=atrb,val=self.activeFun(atrb)) for atrb in atrbs]
        self.layers.append(layer)

        while countHidden >= 1:
            layer = [Neuron(countWeight=len(self.layers[-1]),countPrevLayer=len(self.layers)) for _ in range(countHidden)]
            self.layers.append(layer)
            countHidden -= 1
    
    def fillNeurons(self,values):

        for neuron, value in zip(self.layers[0], values):
            neuron.val = value

        for layerIndex in range(1, len(self.layers)):  # Начинаем с первого скрытого слоя
            for neuron in self.layers[layerIndex]:  # Для каждого нейрона в текущем слое
                prevLayer = [prevNeuron.val for prevNeuron in self.layers[layerIndex - 1]]  # Значения всех нейронов в предыдущем слое
                arrayWeight = neuron.arrayWeight  # Веса текущего нейрона
                # Скалярное произведение предыдущего слоя и весов текущего нейрона

                scolar=np.dot(prevLayer, arrayWeight)
                neuron.val = self.activeFun(scolar)
                neuron.scolarVal = scolar
    
    def errorBack(self,expOut, learnRate):
        for layerIndex in range(len(self.layers) - 1, 0, -1):
            # Проходимся по каждому нейрону в текущем слое
            for neuron_index, neuron in enumerate(self.layers[layerIndex]):                
                # Если текущий слой является выходным слоем
                if layerIndex == len(self.layers) - 1:
                    # Вычисляем ошибку для выходного слоя
                    error = neuron.val - expOut
                else:
                    # Суммируем ошибки из следующего слоя с учетом весов связей
                    error = sum(self.layers[layerIndex + 1][i].arrayWeight[neuron_index] * self.layers[layerIndex + 1][i].weightsD for i in range(len(self.layers[layerIndex + 1])))
                
                neuron.weightsD=error * self.activeFunD(neuron.scolarVal)

                # Обновляем веса нейрона
                for i in range(len(neuron.arrayWeight)):
                    # Вычисляем градиент ошибки по весу
                    neuron.arrayWeight[i]=neuron.arrayWeight[i]-neuron.weightsD * self.layers[layerIndex - 1][i].val*learnRate

    def getPrediction(self,input):
        for neuron, value in zip(self.layers[0], input):
            neuron.val = value

        for layerIndex in range(1, len(self.layers)):  # Начинаем с первого скрытого слоя
            for neuron in self.layers[layerIndex]:  # Для каждого нейрона в текущем слое
                prevLayer = [prevNeuron.val for prevNeuron in self.layers[layerIndex - 1]]  # Значения всех нейронов в предыдущем слое
                arrayWeight = neuron.arrayWeight  # Веса текущего нейрона
                # Скалярное произведение предыдущего слоя и весов текущего нейрона

                scolar=np.dot(prevLayer, arrayWeight)
                neuron.val = self.activeFun(scolar)
                neuron.scolarVal = scolar
        last_layer = self.layers[-1]
    # Находим индекс нейрона с наибольшим значением
        #max_index = max(range(len(last_layer)), key=lambda i: last_layer[i].val)
        #print(last_layer[0].val)

        if last_layer[0].val>=0.5:
            return 1
        else:
            return 0

        

    layers=[]
    activeFun=None
    activeFunD=None

# Функция активации (сигмоид) и ее производная
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoidD(x):
    return sigmoid(x)*(1-sigmoid(x))

def sigmoid_derivative(x):
    return x * (1 - x)

def selectRandomSample(data):
    # Выбираем случайную строку из датасета
    random_index = np.random.randint(len(data))
    random_sample = data.iloc[random_index]
    # Получаем входные данные (все столбцы, кроме "class")
    inputs = random_sample.drop('class').values
    # Получаем ожидаемый выход в виде массива размера два
    output_class = random_sample['class']
    #expected_output = [1, 0] if output_class == 0 else [0, 1]
    return inputs, output_class

def trainNetwork(data, N,countLayers,learnRate):
    layers=None
    for i in range(N):
        # Выбираем случайный пример из датасета
        inputs, expected_output = selectRandomSample(data)

        if layers is None:
            layers=Layers(inputs,countLayers,sigmoid,sigmoidD)

        # Заполняем нейроны входного слоя данными
        layers.fillNeurons(inputs)
        # Обратное распространение ошибки
        layers.errorBack(expected_output,learnRate)
        print(f"\r{round(i/N*100)}% training...", end="", flush=True)
    print(f"\r100% training done!", end="", flush=True)
    return layers

def testNetwork(data,N,layers):
    countTrue=0
    for _ in range(N):
        inputs, expected_output = selectRandomSample(data)
        #out = max(range(len(expected_output)), key=lambda i: expected_output[i])
        pred=layers.getPrediction(inputs)
        #print(f"{expected_output} == {pred}")
        if expected_output==pred:
            countTrue+=1
        
    return countTrue



filename = 'data/mushroomClear.data'  # Путь к вашему файлу
data = pd.read_csv(filename)
#print(data.iloc[0:10, :13])

#параметры
learnRate=0.2
itteration=9000
countHiddenLayes=3
#запуск обучения
layers=trainNetwork(data,itteration,countHiddenLayes,learnRate)

#проверка
N=1000
print(f"\ncount true: {testNetwork(data,N,layers)}/{N}")


#for sub_arr in layers.layers:
   # for neuron in sub_arr:
       # print(neuron.val)

#print(layers.layers)

"""
relu
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
"""