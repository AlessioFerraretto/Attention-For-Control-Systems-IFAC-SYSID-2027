# CONDA

## 1. Turn on your Conda environment (which has Python 3.10)

source $HOME/miniconda3/bin/activate

## Only the first time: conda create --name attention-for-control-systems python=3.10

conda activate attention-for-control-systems

## 2. Install your project

python -m pip install --upgrade pip
pip install -r requirements.txt

## 3. Run your local scripts

python -m training_script experiment=dynamic-future-prediction-v3

# UV + VENV

## 1. Create and activate a virtual environment

```bash
python3.10 -m venv .venv
```

Activate the environment:

**Linux/macOS**

```bash
source .venv/bin/activate
```

**Windows**

```powershell
.venv\Scripts\activate
```

## 2. Install dependencies

Install `uv` if you don't already have it:

```bash
pip install uv
```

Then synchronize the environment:

```bash
uv sync
```

## 3. Run experiments

```bash
python -m training_script experiment=future-prediction
```
