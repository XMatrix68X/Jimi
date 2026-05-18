# ML_ChemTable
You can refer to the "Guildline.pdf" for detailed introduction and environment preparation for the machine learning model. This model is used to replace the FGM table.
Here, I will introduce my parameter setting and training strategy for this model.
## The parameter setting
### The basic networking setting
The number of hidden layers: **5**

The number of neurons: **10, 20, 40, 20, 10**

For the loss function and optimization function, you can find them in the guildline as well as in the "main.py".
### The input and output
The **inputs** of the network are : Z(mixture fraction) and C(normalized progress variable)

The **outputs**: The values of RHO, Diff, VISC, T, SRC_PROG 
## The training process.
The ANN prediction accuracy depends strongly on the training complexity.

First, 1000(Z) * 1000(C) pairs of data are used to train the model. In this progress, the epoch is set to be 3000;

Second, 1000(Z) * 500(C) pairs of data are used to train the model. In this progress, the epoch is set to be 30000 (You can find them in the 01.orgData_LB);

Finally, 100(Z) * 100(C) pairs of data are used for testing the accuracy of the model. (You can find them in The_Test_Data file)

### The validation
In this project, Pearson product-moment correlation coefficient is used to test the validation of the model based on the test set of data as mentioned before. You can find the code in "Pearson.py". In short, the closer Pearson value is to 1, the more accurate the prediction is. 

### The result
The trained model is "Target_network".

The Pearson values:

RHO: **0.99933**

DIFF: **0.99962**

VISC: **0.99944**

T: **0.99927**

SRC_PROG: **0.99312**


# Joint Institute Medium Intellectual

## Project Overview

This project explores how machine learning can be used to support flamelet-based combustion modeling by replacing a traditional lookup-table step with an artificial neural network (ANN). The workflow is centered on training a PyTorch model from precomputed combustion data and then preparing that trained model for later use in an OpenFOAM-based solver workflow.

The original project guideline is preserved in `The_Guildline.pdf`. This Markdown file serves as a practical text description of the project and the contents of this folder.

## Main Idea

In a flamelet-style workflow, quantities such as density, temperature, diffusion terms, source terms, and viscosity are often retrieved from tabulated data using a pair of inputs. In this project, those inputs are:

- `Z`: mixture fraction
- `C`: progress variable

Instead of reading output values directly from a lookup table, the project trains an ANN that learns the mapping:

`(Z, C) -> target thermochemical quantities`

This approach is intended to provide a machine-learning-based surrogate for the tabulated chemistry procedure.

## What The Model Predicts

The available training and test data in this folder indicate that the model is designed to predict several combustion-related output fields, including:

- `RHO`
- `T`
- `DIFF`
- `SRC_PROG`
- `VISC`

These outputs are represented as structured data grids defined over combinations of `Z` and `C`.

## Folder Contents

### Core Python Files

- `main.py`
  Main project script for constructing, training, and saving the ANN model.

- `netTest.py`
  Defines the neural-network architecture used by the training and inference scripts.

- `New_TestforANN.py`
  Test or inference-oriented script for evaluating a trained model on sample data.

- `Pearson.py`
  Utility script related to correlation analysis or statistical inspection of data quality.

### Model Artifacts

- `Target_network.pt`
  Serialized PyTorch model containing the trained network in a directly loadable form.

- `Target_network.pkl`
  Serialized network parameters or model state for a lighter-weight loading path.

### Reference Document

- `The_Guildline.pdf`
  Original project write-up describing the ANN workflow, expected data layout, and intended OpenFOAM integration.

### Data Directories

- `01.orgData_LB`
  Contains organized raw or baseline target-field data such as `RHO.txt`, `T.txt`, `DIFF.txt`, `SRC_PROG.txt`, and `VISC.txt`.

- `The_Test_Data`
  Contains sample input and output files used for testing or validation, including `theZ.txt`, `theC.txt`, and target files prefixed with `Y-`.

## Data Organization

The model uses two scalar inputs:

- `Z` values stored in text files
- `C` values stored in text files

The target outputs are stored as field values arranged over the `Z-C` grid. Conceptually:

- rows correspond to different `Z` values
- columns correspond to different `C` values

Each target file therefore represents one physical quantity sampled over the same structured input space.

This makes the project suitable for supervised learning, where each `(Z, C)` pair is matched to known target values.

## Model Workflow

The intended workflow of the project is:

1. Prepare input data for `Z` and `C`.
2. Prepare target data for the physical quantities to be predicted.
3. Build the ANN in PyTorch.
4. Train the ANN using normalized target values.
5. Save the trained model.
6. Use the trained model for prediction in Python.
7. Integrate the trained model into an OpenFOAM solver through the PyTorch C++ interface (`libtorch`).

## Neural Network Design

According to the guideline document, the network is a feedforward ANN that uses:

- configurable layer sizes
- configurable input and output dimensions
- ReLU activation functions
- Adam optimizer
- mean squared error (MSE) loss

The project is focused more on demonstrating the workflow than on presenting a highly generalized machine-learning framework. The scripts are therefore best understood as project-specific research code.

## Training Notes

The original guideline emphasizes that the output tags are normalized before training. This means:

- the trained model predicts normalized quantities
- postprocessing may be required to recover physical values in original units

The document also notes an important failure mode: if a target field is constant or nearly constant, training may produce `nan` losses because of optimization instability or vanishing-gradient-like behavior.

## Intended OpenFOAM Integration

A major goal of the project is not only to train an ANN in Python, but also to connect that model to OpenFOAM. The proposed path is:

- train the model in Python using PyTorch
- export the trained network
- load the model in C++ with `libtorch`
- modify an existing OpenFOAM solver so that ANN inference replaces or supplements lookup-table evaluation

The PDF specifically frames this as a modification of an existing solver rather than the creation of a completely new one.

## Scope Of This Repository Section

This folder appears to be a compact research prototype or project handoff package. It includes:

- the training and network-definition scripts
- example data
- trained model files
- the original project explanation

It should be treated as a focused experimental codebase for ANN-assisted combustion modeling, rather than a fully polished production package.

## Summary

This project demonstrates a machine-learning surrogate for combustion table lookup. Using `Z` and `C` as inputs, it trains a PyTorch ANN to predict multiple thermochemical outputs and prepares that trained model for later use in an OpenFOAM environment. The folder contains the training code, the network definition, sample data, trained model artifacts, and the original PDF guideline that describes the overall workflow.
