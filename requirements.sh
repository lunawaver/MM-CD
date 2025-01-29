#!/bin/bash

conda create --name MM-CD python==3.10.8

conda activate MM-CD
  
pip install torch==2.1.2 --index-url https://download.pytorch.org/whl/cu118

pip install transformers==4.40.0
pip install numpy==1.26.4
pip install pandas==2.2.2
pip install pillow==10.3.0
pip install scikit-learn==1.4.2
pip install tqdm==4.66.2
