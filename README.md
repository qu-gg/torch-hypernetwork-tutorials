# torch-hypernetwork-tutorials
Hypernetwork training considerations and implementation types in PyTorch. 
Includes classification and time-series examples.


### Current Files
Currently, we have simple examples on the MNIST dataset to highlight the implementation, even if it is a trivial task.
There is a regular full hypernetwork <code>example_MNIST_MLP_FullHypernetwork.py</code>, 
a chunked version <code>example_MNIST_MLP_ChunkedHypernetwork.py</code>, 
and the parallel MLP version where each input gets its own
individual MLP <code>example_MNIST_ParallelMLP_FullHypernetwork.py</code>.

This will be expanded with time-series (e.g. neural ODE) examples and more details on implementation considerations
throughout the repository. Feel free to raise Issues with questions or PRs for expanding!


### How to Parallelize (Conditional) Hypernetworks
Unconditional Hypernetworks (ones that either have shared ‘task embeddings’ for multi-task learning or just 
parameterize one task) allow for the use of batching as the goal is learning networks over tasks. 
This allows them to either just split their batch samples into their tasks and run in parallel 
or do each batch over one task.

<p align='center'><img width=250 src="https://user-images.githubusercontent.com/32918812/206871773-99632e66-7329-4bb2-8996-261541d58041.png" alt="groupConvSchematic" /></p>
<p align='center'>Fig 1. Schematic of the Unconditioned Hypernetwork, using a global embedding vector. Modified from [1].</p>


Conditional Hypernetworks (those using input, support sets, or control variables to influence the parameters) 
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
<p align='center'>Fig 2. Schematic of the Group Convolution Operator, source from [2].</p>

This may seem like additional memory usage, but for Hypernetworks that you batch individual networks over anyway, 
it ends up being equivalent storage. In my experience when applying this to ODE dynamics affected by Hypernetworks, 
it sped training up from 1.5hr/epoch to 12min/epoch for my largest dataset!

### References
[1] Johannes von Oswald, Christian Henning, Benjamin F. Grewe, and João Sacramento. Continual learning with Hypernetworks, 2019.

[2] Shine Lee. Group Convolution. Depthwise Convolution. Global Depthwise Convolution. https://www.cnblogs.com/shine-lee/p/10243114.html, 2019.