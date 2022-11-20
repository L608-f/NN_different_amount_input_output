import numpy.ma

# 3 входных нейрона, 1 выходной
weight = [0.1, 0.49, 0.9]
input_data = [8.4, 0.24, 1.1]
output_data = 1.5

def neural_network(input_data, weight):
    output = 0
    for i in range(len(input_data)):
        output += input_data[i] * weight[i]
    return output

speed = 0.01
for iteration in range(40):
    prediction = neural_network(input_data, weight)
    delta = prediction - output_data
    error = delta ** 2

    weight_delta = numpy.ma.zeros(len(weight))
    for i in range(len(weight_delta)):
        weight_delta[i] = input_data[i] * delta

    for i in range(len(weight)):
        weight[i] -= weight_delta[i] * speed

    print(iteration, error)

print()
print([0.1, 0.49, 0.9])
print(weight)