# FewShotFairness


## Software Setup 
Tested using Python 3.11.0, miniconda 23.1.0, git 2.25.1

1. Set up conda environment called `fairness` with 
```
conda env create -f environment.yml 
```
2. Set environment variables `HF_ACCESS_TOKEN` to your huggingface API token and `OPENAI_API_KEY` to your openai API key

## Data Setup
We use 3 datasets. 
- Bias in Bios
- HateXplain
- TwitterAAE

### Bias in Bios 
To setup bias in bios run these commands
```bash
wget https://storage.googleapis.com/ai2i/nullspace/biasbios/train.pickle -P path/to/data/folder/
wget https://storage.googleapis.com/ai2i/nullspace/biasbios/dev.pickle -P path/to/data/folder/
wget https://storage.googleapis.com/ai2i/nullspace/biasbios/test.pickle -P path/to/data/folder/
```

### HateExplain
To setup HateExplain run this command
```bash
git clone https://github.com/hate-alert/HateXplain.git
```

### Twitter AAE
There are a couple steps to setup twitter aae. We follow the steps found [here](https://github.com/HanXudong/fairlib/tree/main/data).

We reproduce them here for your convenience

1. Download TwitterAAE
```bash
wget http://slanglab.cs.umass.edu/TwitterAAE/TwitterAAE-full-v1.zip
```
2. Clone `demog-text-removal` to prepare the data
```
https://github.com/yanaiela/demog-text-removal.git
```
3. Setup environment for `demog-text-removal` (**Requires python 2.7**)
```bash
conda create -n adv-demog-text python==2.7 anaconda
source activate adv-demog-text
pip install -r requirements.txt
```
4. Run `make_data.py` found in `demog-text-removal/src/data` with `adv-demog-text` environment activated

```bash
python make_data.py /path/to/downloaded/twitteraae_all /path/to/project/data/processed/sentiment_race sentiment race
```

## Running
We use a toml config (WIP) to run the main function. You can take a look at the one provided to get a feel for how to use it

To run this program, activate `fairness` conda environment and run 

```bash
python -m src --config /path/to/config.toml
```

## Tests
We include tests for sanity checking to run these 

```bash
python -m pytest
```