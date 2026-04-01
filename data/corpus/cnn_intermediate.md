# Convolutional Neural Networks

## Local connectivity and weight sharing
Convolutional neural networks (CNNs) exploit spatial structure in inputs such as images. Convolutional layers use small learnable filters that slide across the input, sharing weights across spatial locations. This dramatically reduces parameter count compared with fully connected layers on flattened pixels while preserving translation equivariance: the same feature detector activates regardless of where a pattern appears. LeCun et al. demonstrated these ideas in LeNet for digit recognition, a frequent historical reference in interviews.

## Pooling and receptive fields
Pooling layers subsample feature maps by summarizing local neighborhoods with max or average operations. Pooling increases the effective receptive field of deeper layers, builds invariance to small spatial shifts, and reduces computational cost. Interview discussions often compare max pooling, which preserves sharp activations, with average pooling, which smooths responses. Modern architectures sometimes use strided convolutions instead of explicit pooling blocks while preserving similar benefits.

## Depth and hierarchical features
Stacking convolutional blocks yields hierarchical representations: early layers detect edges and textures; deeper layers capture object parts and semantic categories. Krizhevsky et al.'s AlexNet showed that deep CNNs trained on large labeled datasets could dominate image classification benchmarks. Candidates should articulate how depth, normalization, and data scale interact when explaining why CNNs outperform classical pipelines on vision tasks.
