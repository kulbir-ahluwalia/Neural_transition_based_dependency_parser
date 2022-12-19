# Neural_transition_based_dependency_parser


This is a repository for neural transition-based dependency parser. The parser is implemented in PyTorch. 

To convert jupyter notebook to python script, run the following command:
```zsh
 jupyter nbconvert --to python hwk4.ipynb 
```

For RTX 3060 laptop:
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

For RTX3090Ti:
# NVIDIA-SMI 525.60.11    Driver Version: 525.60.11    CUDA Version: 12.0 
```angular2html
conda create --name nlp_cuda117_python3_9 python=3.9 -y  # for cuda 11.6, pytorch 1.12.1
conda activate nlp_cuda117_python3_9 
conda install astunparse numpy ninja pyyaml mkl mkl-include setuptools cmake cffi typing_extensions future six requests dataclasses
# conda install -c pytorch magma-cuda117 -y
conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia
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
### Renaming conda environments
```angular2html
conda deactivate
conda rename -n old_name  new_name
conda rename -n nlp_cuda116_python3_9  nlp_cuda117_python3_9 

```




