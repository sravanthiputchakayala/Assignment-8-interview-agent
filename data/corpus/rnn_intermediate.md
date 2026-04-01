# Recurrent Neural Networks

## Sequential modeling
Recurrent neural networks (RNNs) process variable-length sequences by maintaining a hidden state updated at each time step. The same transition weights are applied repeatedly, allowing the network to model temporal dependencies in speech, text, and time series. Elman's early work established the basic recurrent unit as a powerful but challenging-to-train architecture. Interview answers should contrast RNNs with feedforward models that require fixed-size inputs.

## Vanishing gradients in long sequences
Standard RNNs struggle to remember information over many steps because repeated application of the transition Jacobian can shrink or explode gradients during backpropagation through time. This vanishing gradient problem limits the effective memory of simple recurrent units when learning long-range dependencies. Interviewers often ask candidates to explain why gradient norms decay across many unfolded time steps and how gated architectures mitigate the issue.

## Relation to LSTMs
Long Short-Term Memory (LSTM) networks extend the RNN family with explicit memory cells and gating functions that regulate information flow. Although this section focuses on vanilla RNNs, understanding their limitations motivates why LSTMs and later architectures such as GRUs became default choices for many sequence tasks before the era of large transformers.
