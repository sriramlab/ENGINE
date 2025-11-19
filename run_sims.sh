#!/bin/bash

geno=data/example
env_file=data/example.env

num_indv=5000
num_snps=5000
sigma_g=0.5
sigma_gxe=0.05
sigma_nxe=0.05

save_folder=output/
mkdir -p ${save_folder}


seed=42

python3 engine.py --bed-prefix ${geno} --env-file ${env_file} \
    --num-samples ${num_indv} --col-stop ${num_snps} --seed ${seed} --B 100 \
    --force-positive-feature "Age" \
    --lifestyle-envs \
    --env-cols "TDI,Sleep duration,Age,Smoking status,Alcohol frequency" \
    --sigma-g ${sigma_g} --sigma-gxe ${sigma_gxe} --sigma-nxe ${sigma_nxe} \
    --iters 2000 --save-files ${save_folder}/test.${seed} --step 1 \
    --cv-variants --l1-tau 5e-3 --l1-anneal linear