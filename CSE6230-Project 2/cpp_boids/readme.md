module purge
module load cmake
module load cuda/12.1.1-6oacj6
module load gcc


depend on folder:
in build/, i do nvcc -arch=sm_70 ../main.cu ../gpu.cu -o ./gpu
