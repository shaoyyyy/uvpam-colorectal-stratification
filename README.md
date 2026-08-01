# uvpam-colorectal-stratification

This repository contains the analysis code and configuration files supporting the manuscript:

"266-nm ultraviolet photoacoustic microscopy for label-free stratification of colorectal lesions"

## Contents

- `run_reproduction.py`: main reproduction workflow
- `configuration/`: analysis parameters and feature definitions
- `feature_selection/`: feature selection procedures and records
- `reference_results/`: reference outputs for verification
- `requirements.txt`: software dependencies

## Data availability

The original UV-PAM images, H&E images, and derived feature tables are not included in this public repository.

Because the dataset is generated from human tissue specimens and is subject to institutional data-sharing procedures, qualified researchers may request access from the corresponding author.

The code structure is provided to facilitate methodological transparency and reproducibility.

## Usage

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the reproduction workflow:

```bash
python run_reproduction.py
```
