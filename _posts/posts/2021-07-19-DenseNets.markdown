---
layout: post
title:  "A Review: DenseNet"
date: 2021-07-19 00:00:00 +0800
# updated: 2021-06-19 20:14:51 +0800
category: posts
excerpt: Shorter connections for deeper, more accurate, and more efficient ConvNets 
---
**Title**: Densely Connected Convolutional Networks <br>
**Authors**: Gao Huang, Zhuang Liu, Laurens van der Maaten, Kilian Q. Weinberger <br>
**Year**: 2016 <br>
**Code Adapted From**: https://github.com/gpleiss/efficient_densenet_pytorch (PyTorch, but without memory efficient methods)

### **Background** 

In the 1990s, LeNet was developed as a system that uses backpropagation to detect digits for the purpose of identifying handwritten zipcode numbers provided by the US Postal Service (LeCun et al., 1998). While this problem may seem trivial now, identifying handwritten digits from the MNIST dataset can be done with a simple MLP with reasonably high accuracy. Yet what LeNet did was to provide a foundation for the way many of the modern deep neural nets do tasks today. In the past decade, state of the art accuracy for the past decade or so have been achieved through a variety of methods implemented on a type of deep neural network called the Convolutional Neural Network (ConvNet). Because of the ability of ConvNets to preserve the spatial nature of features in the image, they were able to provide better accuracies as compared to the standard feed forward network. With the advancements in greater computational powers, neural network operations can be greatly accelerated, which allows the development of more sophisticated models to perform more complex tasks.

It has been no secret that ConvNets are able to achieve lower classifcation errors by adding more layers (more deeper networks). The AlexNet architecture that first won the ImageNet Large Scale Visual Recognition Challenge (ILSVRC) in 2012 with only eight layers. The first five layers were convolutional layers followed by max pooling layers. Features from the last max pooling layer were flattened into a feed forward network, with the final dense layer performing classification. Errors were further reduced in 2014 with GoogleNet and VGG, with the former having 22 layers with skip connections, and the latter having a maximum of 19 layers. 

Why not just keep increasing the number of layers of the network? Using more layers do not proportionally translate to better performances due to problems such as vanishing gradients, where earlier layers of the network seem little to no change from backpropagation. ResNets were able to circumvent that in 2015 by introducing skip connections between blocks, producing a 152 layer deep neural network (ResNet-152) (He et al., 2015).

<p align="center">
    <img src="/assets/images/densenets/evolution.png" alt="Layers and layers" height="300" width="600"/>
    <br>
    Fig. 1: Increasing number of layers from winning entries of ILSVRC (Nguyen et al., 2017)
</p>

In a later paper 'Identity Mappings in Deep Residual Networks', the authors of ResNet improved on the original ResNet paper with a 1001-layer ResNet model (He et al., 2016). Interestingly, Kaiming He mentioned in a presentation on ResNet that the training and inference on a larger network would have provided lower errors, but they were bottlenecked by their hardware at that time.


#### **Do more layers really matter?**

Do we really need so many layers to achieve better performance? In 'Deep Networks with Stochastic Depth', the authors proposed a novel method known as **stochastic depth** for residual networks (Huang et al., 2016). During training, layers are randomly skipped, and the original model is then used during testing. This results in a reduction in training time while allowing models to be deeper (1200 layers for Resnet with stochastic depth). What stochastic depth also suggests is that there may be redundancy across layers in a deep network, where not all layers contribute equally in performance. In this paper, the authors propose a simple fix to ResNets: an architecture that ensures maximum information flow between layers, by connecting all layers **directly** to each other. Redundant layers are removed, and global average pooling is used to reduce the overall number of parameters. Because of the direct connections of layers and a shallower network, it results in a better flow of information across layers and faster training. The authors coined their approach of using densely connected convolutional neural network as *DenseNet* (Huang et al., 2016).

<br>

### **Architecture**

<p align="center">
    <img src="/assets/images/densenets/high_level_architecture.png" alt="DenseNet architecture" height="145" width="1000"/>
    <br>
    Fig. 2: High Level DenseNet Architecture (Huang et al., 2016)
</p>

Fig. 2 depicts the high level overview of the DenseNet architecture. Dense Blocks make up the main composition of the model. Transition layers made up of convolutional and pooling layers are placed between these blocks. After the final block, Global Average Pooling is used and features are passed into a Linear layer for classification.

A typical feed forward network consists of layers that each perform a non-linear transformation $H_l(·)$ of the previous layer. The output of the $(l-1)^{th}$ layer is fed as the input of the $l^{th}$ layer after performing this $H_l(·)$. 

This results in the following:

$$x_l = H_l(x_{l-1})$$

ResNets applies a shortcut connection between residual blocks, with the idea that gradients can flow directly through the shortcut connection during backpropagation from later to earlier layers without much impedance: 

$$x_l = H_l(x_{l-1}) + x_{l-1}$$

DenseNet layers connects previous layers directly to all subsequent layers. Within a Dense Block, feature maps from a layer are concatenated and passed on to the subsequent layers (without any non-linear transformation). At the end of the Dense Block, the non-linear transformation is applied to the concatenated layers:

$$x_l = H_l([x_0, ..., x_{l-1}])$$

$$\small{\text{where }} H_l(·) \small{\text{ for DenseNet consists of a Batch Normalization, ReLU activation, and a 3x3 convolution in that order}}$$

<br>

#### **Dense Block and Dense Layers**

{% highlight python %}
    class DenseBlock(nn.Module):
        def __init__(self, num_layers, input_features, bn_size, growth_rate, dropout_rate):
            super(DenseBlock, self).__init__()
            
            for i in range(num_layers):
                layer = DenseLayer(
                        input_features + growth_rate * i,
                        growth_rate = growth_rate,
                        bn_size = bn_size,
                        dropout_rate = dropout_rate)

                self.add_module('Dense Layer %d' %(i+1), layer)


        def forward(self, init_features):
            features = [init_features]
            for name, layer in self.named_children():
                new_features = layer(*features)
                features.append(new_features)

            return torch.cat(features, 1)
{% endhighlight %}

Each Dense Block consists of several Dense Layers below. Here, the authors introduce the **growth rate k** hyperparameter to maintain the number of feature maps in each Dense Layer. Each function $H_l(·)$ in DenseNet produces a fixed size *k* feature maps. This means that each Dense Layer can only produce an additional $k$ number of feature maps. The forward pass shown below starts off by concatenating feature maps from the previous Dense Layers $[x_0, ..., x_{l-1}]$ before operating on them. After the $l^{th}$ Dense Layer, the number of feature maps produced would be $k_0 + k * (l-1)$, where $k_0$ is the number of feature maps of the initial input layer.

{% highlight python %}
    class DenseLayer(nn.Module):
        def __init__(self, input_features, growth_rate, bn_size, dropout_rate=0.2):
            super(DenseLayer, self).__init__()
            self.add_module('norm1', nn.BatchNorm2d(input_features))
            self.add_module('relu1', nn.ReLU(inplace=True))
            self.add_module('conv1', nn.Conv2d(input_features, bn_size * growth_rate,
                            kernel_size=1, stride=1, bias=False))

            self.add_module('norm2', nn.BatchNorm2d(bn_size * growth_rate))
            self.add_module('relu2', nn.ReLU(inplace=True))
            self.add_module('conv1', nn.Conv2d(bn_size * growth_rate, growth_rate,
                            kernel_size=3, stride=1, padding=1, bias=False))

            self.dropout_rate = dropout_rate

        
        def forward(self, input):

            """ Concatenate inputs """
            concat_features = torch.cat(input, 1)

            """ Bottleneck layer """
            bottleneck_output = self.conv1(self.relu1(self.norm1(concat_features)))

            new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))

            """ If dropout is configured """
            if self.dropout_rate > 0:
                output = F.dropout(new_features, p=self.dropout_rate, inplace=True)

            return output
{% endhighlight %}

From the above, it can be seen that each Dense Layer in the Dense Block consists of the 1x1 convolutional layer (refered to as the bottleneck layer) and a 3x3 convolutional layer following that. The number of input feature maps is typically more than $k$ as it consists of the concatenation of the previous Dense Layer feature maps.

>Although each layer only produces k output feature-maps, it typically has many more inputs.

To reduce the number of feature maps to $k$, the bottleneck 1x1 convolution sets the output to be of $\text{(bn_size * growth_rate)}$ number of layers, where bn_size (bottleneck size) to be 4. Hence, the number of feature maps produced after the 1x1 bottleneck convolution is $4k$, with each feature map having the same height and width as the inputs (1x1 conv properties). Following that, the 3x3 convolution reduces the number of feature maps from $4k$ (output of bottleneck layer) to $k$. The value of the bottleneck size seemed to be arbitually chosen to be 4; the authors did not state their reason. Padding is set to 1 to to keep the size of the feature map the same. Dropout is configured for certain tasks where data augmentation was not used (see Full DenseNet below for more info).

The authors refer to the DenseNet model that utilises bottleneck layers as DenseNet-B. The authors claimed that maintaining a small growth rate $k$ is sufficient in obtaining good results; there is redundancy in models with large number of layers where not all layers are needed.

<br>

#### **Transition Layers**
Between each Dense Block contains a transition layer where convolution and pooling is done. Because concatenation within Dense Blocks can only be done with feature maps of the same size, down-sampling was done outside of the Dense Blocks.

Each Transition Layer consists of the following:
Batch Normalization ⟶ ReLU ⟶ 1x1 Convolution ⟶ 2x2 Average Pooling, stride 2

{% highlight python %}
    class TransitionLayer(nn.Module):
        def __init__(self, input_features, output_features):
            super(TransitionLayer, self).__init__()
            self.add_module('norm', nn.BatchNorm2d(input_features))
            self.add_module('relu', nn.ReLU(inplace=True))
            self.add_module('conv', nn.Conv2d(input_features, output_features,
                            kernel_size=1, stride=1, bias=False))
            self.add_module('pool', nn.AvgPool2d(kernel_size=2, stride=2))
{% endhighlight %}

<br>

### **Full DenseNet**

The authors trained and tested DenseNet on the ImageNet, CIFAR (10 and 100), and SVHN datasets. For different datasets, the authors do have different configurations set up in terms varying depths and hyperparameters, but they still maintain the densely packed layers.

Below is the main DenseNet-BC code for CIFAR-10:

{% highlight python %}
    class DenseNet(nn.Module):
        def __init__(self, growth_rate= 12, block_config = (16,16,16), compression=0.5,
                    num_init_features=24, bn_size=4, dropout_rate=0,
                    num_classes=10, small_inputs=True):

            super(DenseNet, self).__init__()

            # First Convolution
            if small_inputs:
                self.features = nn.Sequential(OrderedDict([
                    ('conv0', nn.Conv2d(3, num_init_features, kernel_size=3, stride=1, padding=1, bias=False))]))

            else:
                self.features = nn.Sequential(OrderedDict([
                    ('conv0', nn.Conv2d(3, num_init_features, kernel_size=7, stride=2, padding=3, bias=False))]))

                self.features.add_module('norm0', nn.BatchNorm2d(num_init_features))
                self.features.add_module('relu0', nn.ReLU(inplace=True))
                self.features.add_module('pool0', nn.MaxPool2d(kernel_size=3, stride=2, paddding=1, ceil_mode=False))


            # Each DenseBlock
            num_features = num_init_features
            for i, num_layers in enumerate(block_config):
                block = DenseBlock(
                        num_layers=num_layers,
                        num_input_features=num_features,
                        bn_size=bn_size,
                        growth_rate=growth_rate,
                        dropout_rate=dropout_rate)

                self.features.add_module('DenseBlock %d' %(i+1), block)
                num_features = num_features + num_layers * growth_rate
                if i != len(block_config) - 1:
                    trans = layer.TransitionLayer(num_input_features=num+features,
                                                num_output_features=int(num_features * compression))
                    
                    self.features.add_module('TransitionLayer %d' %(i+1), trans)
                    num_features = int(num_features * compression)


                # Final batch norm
                self.features.add_module('norm_final', nn.BatchNorm2d(num_features))

                # Linear Layer
                self.classifier = nn.Linear(num_features, num_classes)

                # Initialise
                for name, param in self.named_parameters():
                    if 'conv' in name and 'weight' in name:
                        n = param.size(0) * param.size(2) * param.size(3)
                        param.data.normal_().mul_(math.sqrt(2. / n))

                    elif 'norm' in name and 'weight' in name:
                        param.data.fill_(1)

                    elif 'norm' in name and 'bias' in name:
                        param.data.fill_(0)

                    elif 'classifier' in name and 'bias' in name:
                        param.data.fill_(0) 

        def forward(self, x):
            features = self.features(x)
            out = F.relu(features, inplace=True)
            out = F.adaptive_avg_pool2d(out, (1, 1))
            out = torch.flatten(out, 1)
            out = self.classifier(out)
            return out
{% endhighlight %}

- block_config refers to the number of Dense Blocks used, and the number of Dense Layers in each Dense Block. (16,16,16) here represents 3 Dense Blocks, each containing 16 Dense Layers. 

- The input image first goes through a convolution. For smaller images (CIFAR, SVHN), a 3x3 convolution (kernel size 3) is performed. For larger images (ImageNet), the following is performed:

    7x7 convolution with stride = 2, padding = 3, to produce $2*k$ number of inital feature maps ⟶ Batch Normalization ⟶ ReLU ⟶ max pooling of kernel size = 3, stride = 2. 

- Following the various Dense Blocks and Transition Layers, the feature maps were passed through: 

    ReLU ⟶ Global Average Pooling ⟶ Linear Layer for classification.


- To further reduce the number of feature maps after each Dense Block, **compression** was utilized **after Transition Layers**. A compression factor of $\theta$ was applied, where $0 \lt \theta \leq 1$.  If a Dense Block produces $m$ feature maps, following the Transition Layer, the subsequent Dense Block will receive $\left\lfloor\theta * m\right\rfloor$ feature maps as input. If $\theta$ is set to 1, then it as good as no compression.

    The choice as to whether or not to use compression resulted in 3 variants of DenseNet:
    1. DenseNet (no bottleneck layers, compression factor = 1)
    2. DenseNet-C (compression factor = 0.5)
    3. Densenet-BC (bottleneck layers, compression factor = 0.5)

<br>

- For the CIFAR and SVHN datasets, a batch size of 64 were used, and the model was trained on 300 and 40 epochs respectively. For ImageNet, models were trained for 90 epochs with a batch size of 256.

- Images were normalized with channel means and s.d as a preprocessing step, and models were trained with and without standard data augmentation.

    For training that did not utilised data augmentation had the following additions as quoted below from the paper
    > For the three datasets without data augmentation, i.e., C10, C100 and SVHN, we add a dropout layer after each convolutional layer (except the first one) and set the dropout rate to 0.2.

The initial learning rate was set at 0.1, and divided by 10 at 50% and 75% of the total training epochs. The weights were initialised with Kaiming initialization, with a weight decay of $10^{-4}$ and a Nestorov momentum of 0.9 without dampening.

<br>

### **Conclusion**

The authors were able to show that with a lower number of parameters, the DenseNet-BC architecture was able to produce a lower error than the other competing methods like ResNet. Overfitting was also less of an issue as it was observed that training without data augmentation yield a greater reduction in error over other methods. The authors attributed the architecture's success to the compactness of the model (shorter connections) that allows features from earlier layers to be accessed by later ones. They coined the process in which individual layers receiving more supervision because of the shorter connections as **deep supervision**. 


<br>
<br>

### **References**

He, K., Zhang, X., Ren, S., & Sun, J. (2015). Deep Residual Learning for Image Recognition. ArXiv:1512.03385 [Cs]. http://arxiv.org/abs/1512.03385

He, K., Zhang, X., Ren, S., & Sun, J. (2016). Identity Mappings in Deep Residual Networks. ArXiv:1603.05027 [Cs]. http://arxiv.org/abs/1603.05027

Huang, G., Sun, Y., Liu, Z., Sedra, D., & Weinberger, K. (2016). Deep Networks with Stochastic Depth. ArXiv:1603.09382 [Cs]. http://arxiv.org/abs/1603.09382

Huang, G., Liu, Z., & Weinberger, K. Q. (2016). Densely Connected Convolutional Networks. CoRR, abs/1608.06993. http://arxiv.org/abs/1608.06993

LeCun, Y., Bottou, L., Bengio, Y., & Ha, P. (1998). Gradient-Based Learning Applied to Document Recognition. 46.

Nguyen, K., Fookes, C., Ross, A., & Sridharan, S. (2017). Iris Recognition with Off-the-Shelf CNN Features: A Deep Learning Perspective. IEEE Access, PP, 1–1. https://doi.org/10.1109/ACCESS.2017.2784352