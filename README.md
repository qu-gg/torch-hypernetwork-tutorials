<h2 align='center'>torch-hypernetwork-tutorials</h2>
<h3 align='center'>Hypernetwork training considerations and implementation types in PyTorch. 
<br>Includes classification and time-series examples alongside 1D GroupConv Parallelization.</h3>


<a name="toc"></a>
## Table of Contents
- [Table of Contents](#toc)
- [Current Files](#currentFiles)
- [Information](#background)
  - [Common Types of Hypernetworks](#hypernetworkTypes)
  - [Hypernetwork Training is Tricky](#trickyHypernetworkTraining)
  - [How to Parallelize (Conditional) Hypernetworks](#parallelizeConditionalHypernetworks)
- [PyTorch Considerations](#pytorchConsiderations)
  - [torch.Tensor vs. torch.Parameter and why it matters](#tensorVSparameter)
  - [How to properly assign weights to preserve the computation graph](#weightAssignmentSection)
    - [How NOT to assign weights](#wrongWeightAssignment)
    - [How to CORRECTLY assign weights](properWeightAssignment)
  - [How to debug Hypernetworks](#hypernetworkDebugging)
    - [Prerequisites](#prerequisites)
    - [Specifics](#specifics)
      - [Before the backwards call](#beforeBackwardsCall)
      - [Before the optimizer step](#beforeOptimizerStep)
      - [Right after the optimizer step](#afterOptimizerStep)
- [References](#references)

<a name="currentFiles"></a>
## Current Files
Currently, we have simple examples on the MNIST dataset to highlight the implementation, even if it is a trivial task.
There is a regular full hypernetwork <code>example_MNIST_MLP_FullHypernetwork.py</code>, 
a chunked version <code>example_MNIST_MLP_ChunkedHypernetwork.py</code>, 
and the parallel MLP version where each input gets its own
individual MLP <code>example_MNIST_ParallelMLP_FullHypernetwork.py</code>.

This will be expanded with time-series (e.g. neural ODE) examples and more details on implementation considerations
throughout the repository. Feel free to raise Issues with questions or PRs for expanding!

<a name="generalInformation"></a>
## General Information

<a name="hypernetworkTypes"></a>
### Common Types of Hypernetwork
<b>Full Hypernetworks</b>: Uses a layer to fully map from the latent output of the hypernetwork to the target weights, 
often having large dimensionality and poor scaling issues.

<p align='center'><img width=500 src="https://user-images.githubusercontent.com/32918812/207256894-31bbfe98-03fc-4a45-97f0-e315c5417db6.png" alt="fullNetSchematic" /></p>
<p align='center'>Fig N. Schematic of the Full Hypernetwork, using embedding vectors over tasks. Sourced from [1].</p>

<b>Scaling Hypernetworks</b>: Rather than outputting the full weight tensors, one can instead output scaling coefficients 
that act on some portion of the weight tensor (e.g. rows-by-row, column-by-column). This reduces the degrees of freedom 
for the Hypernetwork, however it allows for linear scaling.

<p align='center'><img width=350 src="https://user-images.githubusercontent.com/32918812/207257345-38c31081-b810-4067-8903-7c07f5265dca.png" alt="scalingNetSchematic" /></p>
<p align='center'>Fig N. Schematic of the Scaling Hypernetwork. Sourced from [3].</p>

<b>Chunked Hypernets</b>: Generates ‘chunks’ of the target network at a time (e.g. layer-by-layer). As this sort of repeated 
application of the hypernetwork to the same input can cause undesired ‘weight sharing’ (representation space of the 
network is shared across layers), often additional trained latent vectors are used as additional inputs per layer. 
Note that this chunking formulation works perfectly well for Convolutional Kernels and the original paper [3] details 
how that may work across layers with differing kernel sizes.

<p align='center'><img width=300 src="https://user-images.githubusercontent.com/32918812/207257622-7b2bda49-bb71-498f-9ec4-a503245fe00b.png" alt="chunkedNetSchematic" /></p>
<p align='center'>Fig N. Schematic of the Chunked Hypernetwork. Sourced from [1].</p>

<a name="trickyHypernetworkTraining"></a>
### Hypernetwork Training is Tricky
Hypernetworks, despite their usefulness, can be incredibly finicky to train well. Firstly, they can take a long time
to start converging on complicated datasets and, in cases, fail to converge at a strong solution [4, 6]. The usage of Batch 
Normalization within the hypernetwork has empirically been shown to stabilize this a bit [5], however this isn't a
universal solution. Indeed, there is no theoretical guarantee that infinitely-wide hypernetworks converge to a global 
minima under gradient descent [6]. 

Due to this, several different works have tried to find solutions, both practical and theoretical, to this problem.
[4] and [5] highlight practical Hypernetwork initialization schemas that help stabilize training and kickstart convergence.
[6] as well propose an initialization schema, however also detail how convexity and convergence guarantees can emerge 
when the main-network becomes itself a wide MLP. In practice, I have found that these initialization tricks still aren't
the end-all-be-all for training, but they can help in certain tasks.

<a name="parallelizeConditionalHypernetworks"></a>
### How to Parallelize (Conditional) Hypernetworks
<b>Unconditional Hypernetworks</b> (ones that either have shared ‘task embeddings’ for multi-task learning or just 
parameterize one task) allow for the use of batching as the goal is learning networks over tasks. 
This allows them to either just split their batch samples into their tasks and run in parallel 
or do each batch over one task.

<p align='center'><img width=250 src="https://user-images.githubusercontent.com/32918812/206871773-99632e66-7329-4bb2-8996-261541d58041.png" alt="groupConvSchematic" /></p>
<p align='center'>Fig N. Schematic of the Unconditioned Hypernetwork, using a global embedding vector. Modified from [1].</p>


<b>Conditional Hypernetworks</b> (those using input, support sets, or control variables to influence the parameters) 
unfortunately do not have that ‘niceness’ in batch training. They often just get batches and apply their network 
sequentially, aggregating statistics and performing the batch updates that way. For some problems, the computation
cost incurred is manageable. However, for more complex problems (e.g. dynamic forecasting using ODEs), this 
computational burden eliminates its feasibility. Thus, if one is an MLP as their main-network, there is an 
alternate way to perform batching for conditional networks. 

A 1D Convolutional layer acts exactly as a Linear layer given the appropriate kernel (1x1) and stride sizes (1). 
You can repurpose the filters of the CNN to be the parameters of the MLP and use Grouped Convolution to achieve a 
forward pass over multiple different MLPs at one time. Groups in Convolution essentially routes which filters are
passed over which in the previous layer, such that the total number of filters is divided by the number of groups 
(num_filt / n_groups). So if you stack the neurons of each network as filters of a layer and group them by the batch 
size, you achieve parallelization.

<p align='center'><img src="https://user-images.githubusercontent.com/32918812/206871390-ba507236-1d9c-4d8f-a0a7-6fdc270b941f.png" alt="groupConvSchematic" /></p>
<p align='center'>Fig N. Schematic of the Group Convolution Operator, source from [2].</p>

This may seem like additional memory usage, but for Hypernetworks that you batch individual networks over anyway, 
it ends up being equivalent storage. In my experience when applying this to ODE dynamics affected by Hypernetworks, 
it sped training up from 1.5hr/epoch to 12min/epoch for my largest dataset!

<a name="pytorchConsiderations"></a>
## PyTorch Considerations
Here we detail PyTorch-specific implementation tricks and considerations when dealing with Hypernetworks, especially
tricky ones that new practitioners may find hard to come by. I hope this, combined with the implementations, have some
practical use and quickens the pace at which one may leverage this method!

<a name="tensorVSparameter"></a>
### torch.Tensor vs. torch.Parameter and why it matters
There is a subtle distinction between the Tensor and Parameter objects in PyTorch and the usage of Tensor in the wrong
place can cause frustration when implementing Hypernetworks with optimized embedding vectors (e.g. multi-task vectors, 
chunked Hypernetworks, etc). Specific information on the docs can be found 
<a href="https://pytorch.org/docs/stable/generated/torch.nn.parameter.Parameter.html">here</a>. 

The general idea is that Tensors may only hold temporary states (such as the hidden state of an RNN) and thus by design
Tensors are not registered as objects to be tracked within the backward pass of the computation graph. Specifically, it 
is not included within the Module's registered parameter list. This means that if one were to try and use a Tensor 
object as the embedding vectors, regardless of the <code>requires_grad</code> flag on the Tensor, it will never be given 
a gradient for itself in the optimization and never updated.

<p align='center'><img src="https://user-images.githubusercontent.com/32918812/207266188-0b5607eb-adfb-4748-b5c2-49e96f11978d.png" alt="tensorNoGradient" /></p>
<p align='center'>Fig N. Using a Tensor as the optimized embedding vector won't work, as no gradients are passed to it.</p>

There are 2 paths that can be taken to resolve this - either manually registering it in the parameter list or converting
the Tensor into a Parameter object. A PyTorch Parameter is a Tensor subclass and specifically designed for this use case
of being automatically added to the Module parameter list when assigned as a Module attribute (e.g. <code>self.embedding</code>). 
Here in this repository, we elect to use the Parameter object as it is cleaner and inherent for this purpose.

As can be seen, the simple change of wrapping the Tensor with the Parameter results in proper gradient tracking for the 
vectors.
<p align='center'><img src="https://user-images.githubusercontent.com/32918812/207268293-9c1ab609-94d7-4431-8dbd-1611b37c7b1d.png" alt="parameterGradient" /></p>
<p align='center'>Fig N. Using a Parameter as the optimized embedding vector works, as it is assigned to the Module's <code>.parameters()</code>.</p>

<a name="weightAssignmentSection"></a>
### How to properly assign weights to preserve the computation graph
Many things can break the computation graph and gradient flow between the main network and the Hypernetwork. For example,
something as innocent as the assignment of the Hypernet's output tensor to the main-net's weights can cause a break if
it is not done right. First we'll start with common mistakes used that cause a graph break despite it looking 'fine'.

<a name="wrongWeightAssignment"></a>
<b>How NOT to assign weights</b>:

<i>Example 1</i>: Setting the main-net weight Tensors to a Parameter made out of the output tensor creates a break as a .clone()
is applied to the weight Tensor <b>w</b>.
<p align='center'><img width=500 src="https://user-images.githubusercontent.com/32918812/207446316-423fb088-00c3-4685-a910-c8d742cf8a3f.png" alt="wrongWeightAssignment" /></p>

<i>Example 2</i>: Setting the Tensor directly to the .data attribute of the Parameter does not work as it breaks the <code>grad_fn</code>
backward connection.
<p align='center'><img width=400 src="https://user-images.githubusercontent.com/32918812/207446888-24aedeb7-e306-4209-89ed-f7657a462311.png" alt="wrongWeightAssignment2" /></p>

<a name="properWeightAssignment"></a>
<b>How to CORRECTLY assign weights</b>:

<i>Method 1</i> - Using the <code>torch.nn.functional</code> layers: These functional layers act as layers not pre-defined in
the nn.Module and allows you to pass in Tensors directly to use as their weights/biases. It seems that no 
<code>.clone()</code> is applied for this layer and thus the graph + grads are preserved. To use this method, it is as
simple as generating the weights from the Hypernetwork and passing in the output Tensors into the <code>F.linear()</code>
function alongside the input.

<p align='center'><img src="https://user-images.githubusercontent.com/32918812/207449907-ad9be26c-4a83-468b-b989-350ef51b6b0c.png" alt="method1Assignment" /></p>
<p align='center'>Fig N. Using <code>torch.nn.functional</code> layers with the direct Tensor output.</p>

<i>Method 2</i> - Directly accessing the defined <code>.weight</code> Tensor but not assigning to it directly: The problem 
with Examples 1 and 2 above is that they tried to modify the weights object itself, which induced <code>.clone()</code>
calls and broke the computation graph. If, instead, we fully delete the <code>.weight</code> object before assignment,
we can preserve the computation graph by straight setting the <code>.weight</code> attribute of the Parameter to that
output Tensor. For this method, we get the full weight output and split it by layer while assigning the splice to the main
network, deleting the old <code>.weight</code> first before assignment.

<p align='center'><img src="https://user-images.githubusercontent.com/32918812/207449563-60e8707c-ef77-494e-a26f-98bd395e1bdb.png" alt="method2Assignment" /></p>
<p align='center'>Fig N. Assigning the <code>.weight</code> Tensor directly by first deleting the old Tensor there.</p>

<a name="hypernetworkDebugging"></a>
### How to debug Hypernetwork issues
Debugging Hypernetworks is a fair bit of effort, however it is the best method in ensuring that the computation graph
is well-structured and your outputs are being properly assigned. This is best to do whenever there is an architectural
change, as we've shown that even 'simple' changes can break things. It is also advised to do this prior to any sort of
legitimate experiments that you wish to glean results from, as a broken computation graph results in defunct training 
and wasted time.

<a name="prerequisites"></a>
<b>Prerequisites</b>: In PyTorch, each layer (e.g. <code>Linear</code> or <code>1DConv</code>) has a weight and bias 
<code>Parameter</code> object that holds the Tensors for that layer, as well as any supporting attributes or state that 
aids in the computation graph updates. This includes the <code>grad</code> and <code>grad_fn</code> attributes, 
which respectively hold the gradients calculated from backprop and the function link to the computation graph next. 
The actual values are stored in the <code>.data</code> attributes of the Parameters.

<p align='center'><img src="https://user-images.githubusercontent.com/32918812/207706388-8854aa7d-a85b-465d-9d00-23447a1fcd51.png" alt="parameterAttributes" /></p>
<p align='center'>Fig N. Attributes of the weight <code>Parameter</code> for a <code>nn.Linear</code> layer.</p>

Broken computation graphs primarily break the <code>grad_fn</code> link and cause any higher-level Tensors to no longer 
receive the gradient flow during the <code>.backward</code> call. Thus, the relevant Tensors to view at various stages
of the update step <code>.data</code>, <code>.grad</code>, and <code>.grad_fn</code>.

<a name="specifics"></a>
<b>Specifics</b>: This method of debugging is best done in an IDE with easy access to breakpoints and viewing object state 
at different steps, such as PyCharm or VS Code. You can do this entirely with print statements; it is just messier. The
essential idea is to set a breakpoint at 3 stages - 1) right before the <code>loss.backward()</code> call, 2) right after
it but before the <code>optimizer.step()</code> call, and 3) right after the optimizer step.

<p align='center'><img width=400 src="https://user-images.githubusercontent.com/32918812/207711193-2e71fbec-5cf4-4210-8f26-2e74b3d55014.png" alt="breakpoints" /></p>
<p align='center'>Fig N. Where to assign debugging breakpoints for Hypernetworks.</p>

The reasons for this are as follows:

<a name="beforeBackwardsCall"></a>
<i>1) Before the <b>backwards</b> call</i>: This allows you to view whether the Tensor output of the Hypernetwork matches the
<code>.weight</code> Tensor of the main network. With this, you can confirm that at least the assignment to the main network
is working.

<p align='center'><img src="https://user-images.githubusercontent.com/32918812/207716367-d9f4ed65-a7d5-4811-ab05-0d98cf98679c.png" alt="debugWeights" /></p>
<p align='center'>Fig N. Viewing the Hypernetwork output and the main network weights.</p>

<a name="beforeOptimizerStep"></a>
<i>2) Before the <b>optimizer</b> step</i>: The <code>.backward()</code> call induces the gradient calculations of the generated
computation graph, meaning that you can view the <code>grad</code> and <code>grad_fn</code> of all Tensors. This is to 
ensure that the Hypernetwork Tensors have a gradient that will be used in their update. A gradient Tensor should be seen
here; otherwise, a graph breakage is occurring somewhere in the network.

<p align='center'><img src="https://user-images.githubusercontent.com/32918812/207720794-919edc43-6749-4875-bf93-c9a09723a1b7.png" alt="debugGradTensors" /></p>
<p align='center'>Fig N. Confirming the Hypernetwork Layers receive gradients after the <code>loss.backward()</code>call.</p>

In addition, you can manually check the <code>grad_fn</code> connections by walking up from the main network weight
Tensor back to the Hypernetwork output. In the figure below, you can confirm the connection if the memory address
of the Hypernetwork output is found in the <code>next_function</code> stack of the main network weight Tensor.

<p align='center'><img src="https://user-images.githubusercontent.com/32918812/207718980-830ef3ca-ade7-46dc-b3ee-6fdd1234c9da.png" alt="debugGradFN" /></p>
<p align='center'>Fig N. Checking the grad_fn stack of the main network weight Tensor to confirm the Hypernetwork connection.</p>

<a name="afterOptimizerStep"></a>
<i>3) Right after the <b>optimizer</b> step</i>: At this step you can ensure that the weight Parameters were 
indeed updated by the gradient and that the learning process is correct for the Hypernetwork layers.

<p align='center'><img src="https://user-images.githubusercontent.com/32918812/207722469-96df1849-6c5f-4e06-a0d9-222f2b128f28.png" alt="debugParameterUpdate" /></p>
<p align='center'>Fig N. Ensuring that the gradient update is being applied to the Hypernetwork weight Tensors after the <code>optimizer.step()</code>.</p>

### Conclusion + Recap
Writing shortly<sup>tm</sup>

<a name="references"></a>
### References
[1] Johannes von Oswald, Christian Henning, Benjamin F. Grewe, and João Sacramento. Continual learning with Hypernetworks, 2019.

[2] Shine Lee. Group Convolution. Depthwise Convolution. Global Depthwise Convolution. https://www.cnblogs.com/shine-lee/p/10243114.html, 2019.

[3] David Ha, Andrew M. Dai, and Quoc V. Le. HyperNetworks. In 5th International Conference on Learning Representations, ICLR 2017, Toulon, France, April 24-26, 2017, Conference Track Proceedings, 2017.

[4] Beck, Jacob, et al. "Hypernetworks in Meta-Reinforcement Learning." arXiv preprint arXiv:2210.11348 (2022).

[5] Chang, Oscar, Lampros Flokas, and Hod Lipson. "Principled weight initialization for hypernetworks." International 
Conference on Learning Representations. 2019.

[6] Littwin, Etai, et al. "On infinite-width hypernetworks." Advances in Neural Information Processing Systems 33 (2020): 13226-13237.

[7] https://discuss.pytorch.org/t/assign-parameters-to-nn-module-and-have-grad-fn-track-it/62677