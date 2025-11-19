# ENGINE
**E**fficient multi-e**N**vironmental **G**ene-environment **I**nteraction i**N**ference **E**stimator
---

## Overview

ENGINE is a variance-component framework for **multi-environment gene–environment (G×E) interaction inference** at biobank scale.

Given:

- a standardized genotype matrix (PLINK `.bed/.bim/.fam`),
- an environment matrix (e.g. lifestyle exposures),
- and either a **real** or **simulated** phenotype,

ENGINE:

- learns a **single environmental embedding** (a weighted combination of environments),
- estimates **variance components** \(\sigma_g^2, \sigma_{g\times e}^2, \sigma_{n\times e}^2, \sigma_e^2\),

This repository contains the main script `engine.py` and an example script `run_sims.sh`.

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/sriramlab/ENGINE.git
cd ENGINE
```

### 2. Create and activate a Python environment

```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install dependencies

We recommend installing from `requirements.txt`:

```bash
pip install -r requirements.txt
```

Minimal `requirements.txt`:

```txt
numpy
pandas
tqdm
bed-reader
scipy
torch
```

---

## Input Data

### 1. Genotypes

ENGINE expects genotypes in **PLINK binary format**:

- `${bed_prefix}.bed`
- `${bed_prefix}.bim`
- `${bed_prefix}.fam`

The `--bed-prefix` argument points to `${bed_prefix}`.

### 2. Environment file

ENGINE reads environment data from a tab-delimited file:

- Must contain columns: `FID`, `IID`, plus environment columns.
- Example (header):

```text
FID    IID    Age    Sleep duration    TDI    Smoking status    Alcohol frequency    ...
```

### 3. Phenotype file

- If `--pheno-file` is **not** provided, ENGINE **simulates** a phenotype using:
  - `--sigma-g`, `--sigma-gxe`, `--sigma-nxe`,
  - and the supplied `--bed-prefix`, `--env-file`, SNP range, etc.
- If you have a real phenotype, provide:
  - `--pheno-file` with space-separated columns including `FID`, `IID`, and the phenotype column.
  - `--pheno-name` to indicate which column is the phenotype (default: `PHENO`).

---

## Quick Start: Example Simulation

The included `run_sims.sh` script runs a simple synthetic example:

```bash
bash run_sims.sh
```

This script:

- Uses `data/example` as the PLINK prefix (expects `example.bed/bim/fam`).
- Uses `data/example.env` as the environment file.
- Simulates a phenotype with:
  - `N = 10,000` individuals (`--num-samples 5000`)
  - `M = 10,000` SNPs (`--col-stop 5000`)
  - variance components `σ_g = 0.5`, `σ_g×e = 0.05`, `σ_n×e = 0.05`.
- Restricts to a subset of lifestyle environments:
  ```bash
  --env-cols "TDI,Sleep duration,Age,Smoking status,Alcohol frequency"
  ```
- Forces the learned embedding to have a **non-negative weight on Age**:
  ```bash
  --force-positive-feature "Age"
  ```
- Enables **variant split-half cross-validation** and **L1 spherical soft-thresholding**:
  ```bash
  --cv-variants --l1-tau 5e-3 --l1-anneal linear
  ```
- Writes output files into `output/` with prefix `test.${seed}`.

After the run, you should see outputs like:

- `output/test.42.E.tsv`
- `output/test.42.y.tsv`
- `output/test.42.E_ids.tsv`
- `output/test.42.y_ids.tsv`
- `output/test.42.keep.txt`
- `output/test.42.extract.txt`
- `output/test.42.alpha_summary.csv`
- `output/test.42.sigma_summary.csv`
- `output/test.42.E_full.tsv`
- `output/test.42.e_hat_full.tsv`
- `output/test.42.e_hat.tsv`
- `output/test.42.e_hat_ids.tsv`

---

## Running ENGINE Manually

Basic simulated-phenotype run:

```bash
python3 engine.py \
    --bed-prefix data/example \
    --env-file data/example.env \
    --num-samples 5000 \
    --col-stop 5000 \
    --sigma-g 0.5 \
    --sigma-gxe 0.05 \
    --sigma-nxe 0.05 \
    --iters 2000 \
    --B 100 \
    --step 0.2 \
    --save-files output/test.sim \
    --cv-variants \
    --l1-tau 5e-3 --l1-anneal linear
```

Real-phenotype example:

```bash
python3 engine.py \
    --bed-prefix data/example \
    --env-file data/example.env \
    --pheno-file data/example.pheno \
    --pheno-name BMI \
    --num-samples 10000 \
    --col-stop -1 \
    --B 100 --iters 2000 --step 0.2 \
    --save-files output/real.BMI
```

---

## Key Command-Line Arguments

Only a subset of the most important flags are listed here. Run:

```bash
python3 engine.py -h
```

for the full help message.

### Core inputs

- `--bed-prefix`: PLINK prefix (required).
- `--env-file`: environment file (required).
- `--pheno-file`: phenotype file (optional).
- `--pheno-name`: phenotype column name in `--pheno-file` (default: `PHENO`).
- `--num-samples`: number of individuals to use (rows subset from PLINK).
- `--col-start`, `--col-stop`: SNP index range \([start, stop]\); `--col-stop -1` → use all SNPs.

### Simulation parameters

(used **only when `--pheno-file` is not given**)

- `--sigma-g`: additive genetic variance (default: 0.5).
- `--sigma-gxe`: G×E variance (default: 0.05).
- `--sigma-nxe`: environment-dependent noise variance (default: 0.05).

### Environment handling

- `--env-cols`: comma-separated list of environment column names to use.
- `--lifestyle-envs`: interpret the env file as containing predefined lifestyle variables in a fixed order.
- `--force-positive-feature`: force the final embedding to have positive weight on a named feature (e.g. `"Age"`).

### Optimization & regularization

- `--iters`: max iterations for learning the embedding \(\alpha\).
- `--step`: base step size for geodesic updates on the unit sphere.
- `--B`: number of Hutchinson probes for GRM trace estimation.
- `--block-snps`: SNP block size for streaming over the genotype matrix.
- `--kgxew-fp32`: store certain large tensors in float32 to reduce memory.
- `--l1-tau`: base strength of spherical soft-thresholding on \(\alpha\).
- `--l1-anneal`: annealing schedule for `--l1-tau` (`none`, `linear`, `cosine`).
- `--l1-tau-min`: final \(\tau\) when using annealing.
- `--nonneg-sigma`: enforce \(\sigma_g, \sigma_{g×e}, \sigma_{n×e}, \sigma_e \ge 0\).

### Variant split-half CV

- `--cv-variants`: enable variant split-half cross-validation:
  - fit \(\alpha\) on SNP half A, estimate \(\sigma\) on B; swap; average.
- `--cv-z`: gate for zeroing out \(\sigma_{g×e}\) and \(\sigma_{n×e}\) if both held-out halves do not exceed \(|\sigma| \le z \cdot SE\).

### Output control & logging

- `--save-files`: prefix for saving intermediate and summary files.
- `--log-level`: logging level (e.g. `INFO`, `DEBUG`).
- `--debug`: additional internal debugging (`off`, `light`, `med`, `heavy`).

---

## Outputs (summary)

Given `--save-files PREFIX`, ENGINE can produce:

- `PREFIX.E.tsv` – standardized/whitened environment matrix used for α-stage (no IDs).
- `PREFIX.y.tsv` – phenotype vector used in the run (no IDs).
- `PREFIX.E_ids.tsv` – same as above, with `FID`/`IID`.
- `PREFIX.y_ids.tsv` – phenotype with `FID`/`IID`.
- `PREFIX.E_full.tsv` – full transformed environment matrix (all rows/cols, FID/IID).
- `PREFIX.e_hat_full.tsv` – full learned environmental score (FID/IID, `e_hat`).
- `PREFIX.e_hat.tsv` / `PREFIX.e_hat_ids.tsv` – learned environmental score, with/without IDs.
- `PREFIX.keep.txt` – PLINK `--keep` file listing selected individuals.
- `PREFIX.extract.txt` – PLINK `--extract` file listing selected SNPs.
- `PREFIX.alpha_summary.csv` – summary of learned embedding coefficients and selection info.
- `PREFIX.sigma_summary.csv` – summary of variance components and SEs.

---

## Reproducibility

- Use `--seed` to set the base RNG seed for:
  - simulated phenotypes,
  - Hutchinson probes,
  - optimization initialization.
- To reproduce a run exactly, record:
  - all command-line arguments,
  - ENGINE commit hash,
  - Python and package versions.

---

## Citing ENGINE

If you use ENGINE in your work, please cite:

