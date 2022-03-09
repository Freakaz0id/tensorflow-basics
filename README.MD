# Requirements #

Anaconda: https://www.anaconda.com/downloads   

TensorFlow: https://www.tensorflow.org/ 

TensorFlow GPU: https://www.tensorflow.org/install/gpu

VS Code: https://code.visualstudio.com/

## MSVC Compiler ##
Required compiler on Windows to build some of the required packages.

* Microsoft Visual C++ Redistributable: https://support.microsoft.com/en-us/help/2977003/the-latest-supported-visual-c-downloads
* Microsoft Build Tools 2015: https://www.microsoft.com/en-US/download/details.aspx?id=48159

Restart afterwards.

## CUDA SDK ##
CUDA can improve performance of machine learning using the graphics card. 
This only makes sense, if you have a powerful Nvidia graphics card available, otherwise you can work with the CPU version instead. 

Follow these steps to install CUDA: https://www.tensorflow.org/install/gpu
* Latest GPU driver
* CUDA Toolkit 11.2 https://developer.nvidia.com/cuda-toolkit-archive
* cuDNN SDK 8.1.0: https://developer.nvidia.com/cudnn

## Conda Environment ##
Create environment called 'tensorflow'
```
conda create --name tensorflow python=3.8
```

Activate environment
```
conda activate tensorflow
```

Install requirements
```
cd tensorflow-basics
cd my-project

pip install --upgrade pip
pip install -r requirements.txt
```
This includes tensorflow 2.6 as well as other helpful packages like scikit-learn.



## Alternativ: Docker ##

You can run tensorflow using Jupyter Notebook with a preconfigured docker image:

```
docker pull tensorflow/tensorflow:latest  # Download latest stable image
docker run -it -p 8888:8888 tensorflow/tensorflow:latest-jupyter  # Start Jupyter server 
```

