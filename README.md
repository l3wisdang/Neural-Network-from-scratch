# 🧠 Neural Network from Scratch

This project explores the core mechanics of deep learning by building and training a neural network from scratch using Python and NumPy. Rather than relying on high-level libraries such as TensorFlow or PyTorch, I implemented the main mathematical ideas manually, including forward propagation, backpropagation, and gradient descent. I wanted to understand how neural networks *actually* learn, not just how to call a library that learns for me.

The repository includes two implementations: a single-layer network that solves a simple linearly separable problem, and a multi-layer network with one hidden layer that can handle a more complex, non-linear pattern similar to XOR.

## 📦 Technologies

- `Python`
- `NumPy`
- `Jupyter Notebook`

## ⚙️ What I Built

**Mathematical implementation**

I implemented the sigmoid activation function to introduce non-linearity into the network, along with its derivative so gradients could be calculated during backpropagation. I initialised the weight matrices, `syn0` and `syn1`, using a uniform random distribution centred around zero, which helps maintain stable gradient flow.

**Network architecture**

The first version maps an input matrix `X` of shape 4×3 directly to an output `y` through a single weight matrix. I then extended this into a three-layer model: input, hidden, and output. The hidden layer contains four neurons, which allows the model to learn *combinations* of features rather than just simple relationships between individual inputs.

**Training loop**

The training loop runs for thousands of iterations and follows a simple rhythm. The model makes predictions through a forward pass (matrix multiplication followed by sigmoid activation), calculates the error against the target output, passes that error backwards through the network using the sigmoid derivative to work out how each weight should adjust, and finally updates the weights with gradient descent. Watching the error shrink iteration by iteration is what made the whole process click for me.

## 📚 What I Learned

**Matrix mechanics**

One of the biggest lessons was how important matrix dimensions are. Multiplying a 4×3 input by a 3×4 weight matrix produces a 4×4 hidden layer. Once I could trace shapes through the network, the logic became much clearer, and dimension errors stopped feeling like mysteries.

**Why hidden layers matter**

I also saw first-hand why hidden layers are so important. A single layer works well for straightforward, linear relationships, but a hidden layer is needed to capture more complex patterns that are not obvious in the raw data.

**Backpropagation made concrete**

Backpropagation became far easier to understand once I could see it in code. In particular, calculating `l1_error` by projecting `l2_delta` backwards through the transpose of `syn1` made the chain rule feel concrete and intuitive rather than abstract maths.

**Real-world relevance**

These ideas connect directly to applications I care about as a finance student. The same matrix operations and error-correction process form the basis of models used in predictive analytics and risk assessment, such as credit scoring and fraud detection, and hidden layers are central to recommendation engines that find patterns in user behaviour. Computer vision and NLP use far larger models, but the underlying principle is identical: calculate an error, work out how it changes the weights, and keep adjusting until performance improves.

## 💬 How can it be improved?

- Add more hidden layers and experiment with their sizes
- Try alternative activation functions (ReLU, tanh) and compare convergence
- Introduce a learning rate and explore how it affects training stability
- Visualise the error curve over training iterations
- Apply the network to a real dataset, such as credit default classification

## 🔌 Running the Project

1. Clone the repository to your local machine
2. Install NumPy: `pip install numpy`
3. Open the notebooks in Jupyter:

```bash
jupyter notebook single_layer_network.ipynb
jupyter notebook multi_layer_network.ipynb
```

## 🙏 Acknowledgments

Core logic and architecture adapted from the tutorial [A Neural Network in 11 Lines of Python](https://iamtrask.github.io/2015/07/12/basic-python-network/) by [iamtrask](https://github.com/iamtrask). All implementation, experimentation, and write-up are my own.
