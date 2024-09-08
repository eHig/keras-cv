git checkout eric/efficient-det
cd keras-cv
export XLA_FLAGS=--xla_gpu_cuda_data_dir=/usr/lib/cuda
pip install -r requirements.txt
pip install tensorflow==2.16.1

pip install wandb