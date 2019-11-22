# Density LIME

In this project, we propose a better sampling strategy for generating explanations.
In particular, we propose to sample images that are more likely to live on the
manifold of the real data distribution.


## Installation
In order to be able to do inpainting stuff, the git submodules needs to be initialized:

```
git submodule update --init --recursive
```

This will clone the dependent repositories into to folder `src/external_modules`.


## Project structure

```
src:
	lime: 				# Copy of original lime implementation
	density-lime:		# Extensions and modifications of lime implementations
	playground:			# Various example files.
	external_modules: 	# Directory for git submodules.
```

## Running Examples

In the `src/playground/` directory there is a couple of examples. One of these examples
is the `image_lime_test.py`, which will generate a plot like the one found in the 
`src/playground/plots` directory. To run the script, run the following command when
being located in the `src/playground` directory:

```bash
> python image_lime_test -i car train surf toilet
```

The `car`, `train`, `surf`, and `toilet` argument refers to png images located in the
`src/playground/data` directory.

