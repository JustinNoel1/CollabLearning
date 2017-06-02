# CollabLearning
In this project, we introduce several techniques for training collaborative ensembles of deep neural networks. In addition to learning to solve an image classification problem, these collaborative ensembles also learn to reach a consensus on their predictions. As a consequence, each of the models in these ensembles learns to mimic a traditional ensemble, with corresponding improvements to performance.

A written summary of this project is located in the `latex/report.pdf` file, including a detailed analysis of the performance of these models. The upshot of this summary is that neural networks can be trained in collaborative ensembles where they learn to reach a consensus on their functions. We observe:
* The individual models in these collaborative ensembles (for most reasonable choices of hyperparameters) obtain better performance on new data then their traditionally trained counterparts. 
* Some of the individual collaboratively trained models even outperform the entire traditional ensemble yielding a faster and smaller approximation of an ensemble.
* Even a very small amount of collaboration can improve both the individual models and the entire ensemble.

The details of our build environment are contained in the `requirements.txt` file or, for those using conda, are in the `environment.yaml` file. Probably the most important requirements are Python 3.6 and Tensorflow 1.1.

To train the models:
1. Execute 
`python my_capstone.py`
 from the source directory. This will download the datasets as necessary and train all of the models according to the hardcoded hyperparameters in `my_capstone.py`. Note that at various points this will train some of the ensembles on a severely truncated dataset. This are just tests. If you see some of the models have terrible behavior, but are training extremely fast, this is what is happening.
1. Take a vacation. (If you are using a CPU, make it a long vacation).

Statistical data, graphs, and checkpoints will be dumped into the `collab_logs` directory. These can be combined into more meaningful graphs by running:
`python analysis.py`
The output will be pushed into the `latex/images` directory. 