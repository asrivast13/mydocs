conda create -n nemo3 pip python=3.8
conda activate nemo3
pip install Cython
git clone https://github.com/NVIDIA/NeMo.git ~/Source/NeMo
cd ~/Source/NeMo/
./reinstall.sh
conda uninstall llvmlite
pip install llvmlite
pip install numpy numba
