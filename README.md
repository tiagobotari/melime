# MeLIME - Meaningful Local Explanation for Machine Learning Models

In this project,  we introduce strategies to improve local explanations taking into account the  distribution  of  the  data (on the manifold) used  to  train  the  black-box models. MeLIME, produces more meaningful explanations compared to other techniques over different ML models,  operating on various types of data.  MeLIME generalizes the LIME (https://github.com/marcotcr/lime) method, allowing more flexible perturbation sampling and the use of  different  local  interpretable  models.   Additionally,  we  introduce modifications  to standard training algorithms of local interpretable models fostering more robust explanations,  even  allowing  the  production  of counterfactual  examples.

The preprint of the paper is available on https://arxiv.org/abs/2009.05818


## How to install using python virtual venv

```
git clone https://github.com/tiagobotari/melime.git
cd melime
python3.7 -m venv venv
python setup.py install 
```

## Examples

The examples are on the folder /melime/experiments.

### Running the Innvestigate experiment

The experiment in the file `m-lime/experiments/ex_mnist_cnn_innvestigate.py` needs the [`innvestigate`]() framework.
To install it, run the following command: 

```bash
> git submodule update --init
```
This will download the module to the directory `submodules/innvestigate`.

Afterwards, the script can be run from the root directory: 

```bash
> python melime/experiments/ex_mnist_cnn_innvestigate.py
```

This will plot the result using `matplotlib`.
If you want to output a pdf, then prepend the command as follows:

```bash
> PLOTFILENAME="output.pdf" python m-lime/experiments/ex_mnist_cnn_innvestigate.py
```


