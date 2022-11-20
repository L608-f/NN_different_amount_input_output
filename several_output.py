import numpy.ma

weight = [0.1, 0.49, 0.9]
output_data = [8.4, 0.24, 1.1]
input_data = 1.95

def neural_network(input_data, weight):
    output_data = numpy.ma.zeros(len(weight))
    for i in range(len(weight)):
        output_data[i] += input_data * weight[i]
    return output_data

speed = 0.1
for iteration in range(20):
    prediction = neural_network(input_data, weight)
    delta = [prediction[i] - output_data[i] for i in range(len(output_data))]
    error = [delta[i] ** 2 for i in range(len(output_data))]
    weight_delta = [input_data * delta[i] for i in range(len(delta))]

    for i in range(len(weight)):
        weight[i] -= weight_delta[i] * speed

    print(iteration, error)

print()
print(weight)