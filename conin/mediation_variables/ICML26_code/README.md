# Installation Instructions

1. Create the conda environment: run ```conda env create -f conin_tmlr.yaml```
2. Activate it: ```conda activate conin_tmlr```.
3. Pytorch will need to be installed separately. You can refer to the [website instructions](https://pytorch.org/) for which version is right for your machine. Most likely, you only need to specify which OS and leave everything else on default.
   1. If your machine doesn't have a Nvidia gpu, you should select "Compute Platform: CPU". Be sure to set the device argument in the code to 'cpu' as well.
5. Install it as a Jupyter kernel: ```python -m ipykernel install --user --name conin_tmlr --display-name "conin_tmlr"```
6. Once you open the ```examples.ipynb``` notebook, you should be able to select the kernel by clicking the circle in the upper-right corner. If it's not appearing, trying refreshing your browser/IDE.

# File Descriptions

1. "examples" is an example notebook that should be sufficient for introducing you to the code and what functions to run. Primary notebook you should reference.
2. "ICML_comparisonStudies" contains the comparisons to ILP and beam search. There are some bespoke functions that are tailored to subsequence constraints.
3. "ICML_dna" is a mess. Lots of locally defined functions and tailored functions for DNA. Not advisable.

