{
 "cells": [
  {
   "cell_type": "raw",
   "metadata": {
    "tags": [
     "remove_cell"
    ]
   },
   "source": [
    "---\n",
    "title: \"Experimenting with TFT\"\n",
    "author: Douglas Araujo\n",
    "format: \n",
    "    html:\n",
    "        toc: true\n",
    "        toc-location: right\n",
    "        toc-depth: 4\n",
    "        number-sections: true\n",
    "        code-fold: true\n",
    "        code-tools: true\n",
    "        embed-resources: true\n",
    "bibliography: ref.bib\n",
    "---"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "This notebook describes the temporal fusion transformers [@lim2021temporal] architecture, and ports it over to keras 3 while making some punctual improvements, including bringing the notation closer to the one in the paper.\n",
    "\n",
    "The original repository is [here](https://github.com/google-research/google-research/tree/master/tft)."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 1,
   "metadata": {},
   "outputs": [],
   "source": [
    "#| output: false\n",
    "\n",
    "from __future__ import annotations\n",
    "\n",
    "import os\n",
    "import torch\n",
    "os.environ[\"KERAS_BACKEND\"] = \"torch\"\n",
    "#os.environ[\"CUDA_VISIBLE_DEVICES\"] = \"1\"\n",
    "import keras\n",
    "from keras import layers\n",
    "\n",
    "import matplotlib.pyplot as plt\n",
    "import numpy as np\n",
    "import pandas as pd\n",
    "import random\n",
    "\n",
    "from datetime import timedelta\n",
    "from dateutil.relativedelta import relativedelta\n",
    "from fastcore import docments\n",
    "from nbdev.showdoc import show_doc\n",
    "from sklearn.preprocessing import StandardScaler\n",
    "from torch.utils.data import Dataset, DataLoader\n",
    "from tqdm import tqdm"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Introduction"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "The main characteristics of TFT that make it interesting for nowcasting or forecasting purposes are:\n",
    "\n",
    "- **multi-horizon forecasting**: the ability to output, at each point in time $t$, a sequence of forecasts for $t+h, h > 1$\n",
    "- **quantile prediction**: each forecast is accompanied by a quantile band that communicates the amount of uncertainty around a prediction\n",
    "- **flexible use of different types of inputs**: static inputs (akin to fixed effects), historical input and known future input (eg, important holidays, years that are known to have major sports events such as Olympic games, etc)\n",
    "- **interpretability**: the model learns to select variables from the space of all input variables to retain only those that are globally meaningful, to assign attention to different parts of the time series, and to identify events of significance\n",
    "\n",
    "## Main innovations\n",
    "\n",
    "The present model includes the following innovations:\n",
    "\n",
    "- **Multi-frequency input**\n",
    "\n",
    "- **Context enhancement from lagged target**: the last known values of the target variable are embedded (bag of observations), and this embedding is used similar to the static context enhancement as a starting point for the cell value in the *decoder* LSTM."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Preparing the data\n",
    "\n",
    "The functions below will be tested with simulated and real data. The former helps to illustrate issues like dimensions and overall behaviour of a layer, whereas the latter will demonstrate on a real setting in economics how the input and output data relate to one another.\n",
    "\n",
    "More specifically, the real data used will be a daily nowcasting exercise of monthly inflation. Note that the models will not necessarily perform well, since their use here is for illustration purposes and thus they are not optimised. Also, the dataset is not an ideal one for nowcasting: other variables could also be considered."
   ]
  },
