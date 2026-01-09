import json
import os

notebook_content = {
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# MagNet Deep Learning Pipeline\n",
    "\n",
    "This notebook serves as the entry point for running the MagNet project locally.\n",
    "It executes the `main.py` script while maintaining the modular folder structure."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 1. Setup Environment\n",
    "\n",
    "Install the required dependencies if you haven't already."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "!pip install -r requirements.txt"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 2. Train Models\n",
    "\n",
    "Run the training pipeline using `main.py`.\n",
    "Available models: `scaler`, `sequence`, `seq2seq`, `cnn`, `transformer`."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Example: Train CNN Model\n",
    "# Ensure the data file path is correct relative to this notebook\n",
    "!python main.py --data \"../3C90_TX-25-15-10_Data1_Cycle.mat\" --model cnn --epochs 50"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Example: Train Transformer Model\n",
    "!python main.py --data \"../3C90_TX-25-15-10_Data1_Cycle.mat\" --model transformer --epochs 50"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 3. Help\n",
    "Run the help command to see all options."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "!python main.py --help"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.8.10"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}

script_dir = os.path.dirname(os.path.abspath(__file__))
notebook_path = os.path.join(script_dir, 'MagNet_Run.ipynb')

with open(notebook_path, 'w') as f:
    json.dump(notebook_content, f, indent=1)

print(f"Created notebook at {notebook_path}")
