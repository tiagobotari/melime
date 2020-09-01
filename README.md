# Meaningful Local Explanation for MachineLearning Models

In this project, we propose a better sampling strategy for generating explanations.
In particular, we propose to sample data that are more likely to live on the
manifold of the real data distribution.








### Running the Innvestigate experiment

The experiment in the file `m-lime/experiments/ex_mnist_cnn_innvestigate.py` needs the [`innvestigate`]() framework.
To install it, run the following command: 

```bash
> git submodule update --init
```
This will download the module to the directory `submodules/innvestigate`.

Afterwards, the script can be run from the root directory: 

```bash
> python m-lime/experiments/ex_mnist_cnn_innvestigate.py
```

This will plot the result using `matplotlib`.
If you want to output a pdf, then prepend the command as follows:

```bash
> PLOTFILENAME="output.pdf" python m-lime/experiments/ex_mnist_cnn_innvestigate.py
```


