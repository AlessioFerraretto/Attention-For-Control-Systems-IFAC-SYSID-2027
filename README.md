# 1. Turn on your Conda environment (which has Python 3.10)

source $HOME/miniconda3/bin/activate

# Only the first time

conda create --name attention-for-control-systems python=3.10
conda activate attention-for-control-systems

# 2. Install your project

python -m pip install --upgrade pip
pip install -r requirements.txt

# 3. Run your local scripts

python -m training_script experiment=stable_spline
