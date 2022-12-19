# Neural_transition_based_dependency_parser


This is a repository for neural transition-based dependency parser. The parser is implemented in PyTorch. 

To convert jupyter notebook to python script, run the following command:
```zsh
 jupyter nbconvert --to python hwk4.ipynb 
```

```zsh
conda create --name nlp_cuda116_python3_9 python=3.9   # for cuda 11.6, pytorch 1.12.1
conda activate nlp_cuda116_python3_9 
conda install astunparse numpy ninja pyyaml mkl mkl-include setuptools cmake cffi typing_extensions future six requests dataclasses
conda install -c pytorch magma-cuda116
conda install pytorch torchvision torchaudio cudatoolkit=11.6 -c pytorch -c conda-forge
conda install matplotlib pandas -y
pip3 install torchtext tabulate
pip3 install torchdata cloudpickle
conda install spacy -y
pip3 install ipywidgets widgetsnbextension pandas-profiling

conda install -c conda-forge notebook -y
conda install -c conda-forge nb_conda_kernels -y
conda install -c conda-forge jupyterlab -y
conda install -c conda-forge jupyter_contrib_nbextensions -y
```




