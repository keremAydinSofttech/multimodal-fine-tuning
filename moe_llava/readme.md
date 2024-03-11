git clone https://github.com/PKU-YuanGroup/MoE-LLaVA
cd MoE-LLaVA
conda create -n moellava python=3.10 -y
conda activate moellava
pip install --upgrade pip  # enable PEP 660 support
pip install -e .
pip install -e ".[train]"
pip install flash-attn --no-build-isolation
pip install llama-index
pip install replicate
conda install mpi4py
conda install -c conda-forge cudnn
conda install -c anaconda cudatoolkit
conda install -c conda-forge cudatoolkit-dev