# Artificial Neural Networks

## Forward propagation
An artificial neural network (ANN) is a composition of layers of artificial neurons. Each neuron applies a weighted sum of its inputs, adds a bias term, and passes the result through a nonlinear activation function. Forward propagation is the process of passing an input vector through successive layers until the network produces an output vector used for prediction or classification. During training, the choice of activation function affects how gradients flow and whether deep stacks remain trainable, which interviewers often connect to vanishing or exploding gradients.

## Backpropagation
Backpropagation computes the gradient of a loss function with respect to each weight by applying the chain rule from the output layer back toward the input layer. Rumelhart, Hinton, and Williams (1986) popularized this efficient algorithm for training multi-layer networks. In practice, automatic differentiation frameworks implement reverse-mode differentiation so practitioners rarely hand-derive gradients. Interview answers should mention that backprop reuses intermediate activations stored during the forward pass and that batching stabilizes gradient estimates when data are shuffled each epoch.

## Universal approximation
A feedforward network with at least one hidden layer and a non-polynomial activation can approximate continuous functions on compact domains given sufficient width. This universal approximation property explains why ANNs are flexible function approximators, but it does not guarantee good generalization or efficient learning from finite data. Regularization, architecture design, and careful validation remain essential in applied deep learning projects.
