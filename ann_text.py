import numpy as np
import requests
import spacy

nlp = spacy.load("en_core_web_md")

class NeuralNetwork:
    def __init__(self):
        np.random.seed(1)
        self.synaptic_weights = 2 * np.random.random((300, 1)) - 1  # Use 300-dimensional word embeddings

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        return x * (1 - x)

    def train(self, training_inputs, training_outputs, training_iter):
        for i in range(training_iter):
            output = self.think(training_inputs)
            error = np.abs(training_outputs - output)  # Calculate element-wise absolute difference
            adjustments = np.dot(training_inputs.T, error * self.sigmoid_derivative(output))
            self.synaptic_weights += adjustments

    def think(self, inputs):
        inputs = np.array(inputs)
        output = self.sigmoid(np.dot(inputs, self.synaptic_weights))
        return output

if __name__ == "__main__":
    neural_network = NeuralNetwork()

    print("Random starting synaptic weights:\n", neural_network.synaptic_weights)

    training_inputs = np.array([["Lorem", "Ipsum", "Dolor"],
                                ["Sit", "Amet", "Consectetur"],
                                ["Adipiscing", "Elit", "Sed"],
                                ["Eiusmod", "Tempor", "Incididunt"]])

    training_outputs = np.array([[0, 1, 1, 0]]).T

    # Convert text inputs to numerical representations
    training_inputs = np.array([nlp(" ".join(text)).vector for text in training_inputs.tolist()])


    # Train the neural network
    neural_network.train(training_inputs, training_outputs, 10000)

    print("Synaptic weights after training:\n", neural_network.synaptic_weights)

    headers = {
        "X-RapidAPI-Key": "160e6acc34msh5c3ca8b91347bc6p1c67d5jsnaa26d0b8a1a1",
        "X-RapidAPI-Host": "wordsapiv1.p.rapidapi.com"
    }
    api_url = "https://wordsapiv1.p.rapidapi.com/words/hatchback/typeOf"

    # Fetch random text using API
    response = requests.get(api_url, headers=headers)
    print(response.json())
    if response.status_code == 200:
        text_data = response.text
        print(text_data)
        inputs = text_data.split()[:3]  # Select the first three words from the generated text

        # Convert text inputs to numerical representations
        inputs = np.array([nlp(" ".join(text)).vector for text in inputs])


        print("New situation: input data =", inputs)
        print("Output data:")
        print(neural_network.think(inputs))
    else:
        print("Failed to fetch text from the API.")
