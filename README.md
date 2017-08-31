# CNNGeometric MatConvNet implementation

![](http://www.di.ens.fr/willow/research/cnngeometric/images/teaser.png)

(![](http://www.di.ens.fr/willow/research/cnngeometric/images/new.gif) [PyTorch version released](https://github.com/ignacio-rocco/cnngeometric_pytorch))

This is the implementation of the paper: 

I. Rocco, R. Arandjelović and J. Sivic. Convolutional neural network architecture for geometric matching. CVPR 2017 [[website](http://www.di.ens.fr/willow/research/cnngeometric/)][[arXiv](https://arxiv.org/abs/1703.05593)]

using the MatConvNet toolbox for MATLAB.

If you use this code in your project, please cite use using:
````
@InProceedings{Rocco17,
  author       = "Rocco, I. and Arandjelovi\'c, R. and Sivic, J.",
  title        = "Convolutional neural network architecture for geometric matching",
  booktitle    = "Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition",
  year         = "2017",
}
````

### Getting started ###
  - demo.m downloads trained models and shows how to perform alignment
  - evaluate.m shows how to evaluate the trained models on the ProposalFlow and Caltech-101 datasets.
  - demo_train.m shows how to train a new model
