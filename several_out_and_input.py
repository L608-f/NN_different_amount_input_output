import numpy.ma

weight = numpy.ma.zeros((3, 3))
input_data = [8.4, 0.24, 1.1]
output_data = [10, 5, 1]
speed = 0.01

def neural_network(input_data, weight):
    output = [weight_summ(input_data, weight[i]) for i in range(len(weight))]
    return output

def weight_summ(input_data, weight_i):
    output = 0
    for i in range(len(input_data)):
        output += input_data[i] * weight_i[i]
    return output

for iteration in range(10):
    prediction = neural_network(input_data, weight)
    delta = [prediction[i] - output_data[i] for i in range(len(output_data))]
    error = [delta[i] ** 2 for i in range(len(delta))]
    weight_delta = numpy.ma.zeros((3, 3))

    for i in range(len(output_data)):
        for j in range(len(delta)):
            weight_delta[i][j] = input_data[j] * delta[i]

    for i in range(len(weight)):
        for j in range(len(weight[0])):
            weight[i][j] -= weight_delta[i][j] * speed

    print(iteration, error)

print()
print(weight)