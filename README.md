# SciWiki Dataset Construction

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Dataset on Hugging Face](https://img.shields.io/badge/Dataset-HuggingFace-blue)](https://huggingface.co/datasets/danilotpnta/SciWiki)
<!-- [![Build Status](https://img.shields.io/github/actions/workflow/status/yourusername/yourrepo/ci.yml)](https://github.com/yourusername/yourrepo/actions) -->

A curated collection of 2k high-quality Wikipedia articles aligned with ScienceDirect topics. This repository houses scripts to fetch URLs, filter articles using the ORES API, and extract content in multiple formats—laying the foundation for generating scientific articles with LLMs.

## Project Overview

The SciWiki dataset pipeline performs the following tasks:
- **URL Retrieval:** Gathers Wikipedia URLs corresponding to ScienceDirect topics.
- **Article Filtering:** Uses the ORES API to evaluate and filter articles for quality.
- **Content Extraction:** Extracts and stores article content in multiple formats (HTML, plain text, JSON, Markdown).



<!-- ## Data Directory Structure

After running the dataset construction process, your `data/` folder will be organized as follows:

```
data/
├── urls/
│   └── topics_urls.json
├── ores/
│   └── topics_ores_scores.json
└── articles/
    ├── txt/
    └── json/
``` -->


## Installation

### Prerequisites

Ensure you have [Conda](https://docs.conda.io/en/latest/) installed or other virtual environment managers like `venv`.

### Steps

1. **Create and Activate the Environment**

```sh
conda create -n sciwiki python=3.11 -y
conda activate sciwiki
```

2. **Install Required Packages**

Make sure you have the correct torch version installed for your system. For CUDA 11.1, use the following command:

```sh
pip install torch==2.5.1+cu121 --extra-index-url https://download.pytorch.org/whl/cu121
```

Then, install the remaining dependencies:

```sh
pip install -r requirements.txt
```


## Dataset Construction and Usage

### Pre-requisites
It is recommended to set up your pythonpath to the root of the project directory. This will allow you to run the scripts from any location.

If you are using **Zsh**, you can use the following commands:
```bash
echo 'export PYTHONPATH="$HOME/SciWiki-Maker"' >> ~/.zshrc
source ~/.zshrc
```

### Running the Dataset Builder

For running the dataset builder as a script you can use the following command:


```bash
python src/build_dataset.py \
    --topics_file input/topics.json \
    --output_dir data \
    --max_workers 1 \
    --file_types txt json
```


### Running the Individual Modules
#### 1. URL Fetcher

```bash
python src/modules/get_wikipedia_urls.py \
    --topics_json input/topics.json \
    --output_dir data/urls/
```

#### 2. ORES Score Fetcher

```bash     
python src/modules/get_ores_scores.py \
    --urls_file data/urls/topics_urls.json \
    --output_dir data/ores/
```

#### 3. Article Extractor

```bash
python src/modules/extract_articles.py \
    --topics_file input/topics.json \
    --ores_scores_file data/ores/topics_ores_scores.json \
    --output_dir data/articles/ \
    --max_workers 1 \
    --file_types txt json
``` 


### Command Line Arguments

- **`--topics_file`**:  
  Path to the JSON file containing the initial topics. *(Required)*

- **`--output_dir`**:  
  Directory where all output files will be saved.  
  *(Default: `data` folder at project root)*

- **`--max_workers`**:  
  Maximum number of workers for both URL fetching and ORES score fetching.  
  *(Default: 1 for non-parallelism)*

- **`--file_types`**:  
  Space-separated list of file types to save for each article.  
  Choose from: `html`, `txt`, `json`, `md`.  
  *(Default: `txt` and `json`)*


## Additional Information

- **Data Sources:** Wikipedia and ScienceDirect.
- **APIs Used:** ORES API for quality assessment.
- **Output Formats:** Articles are saved in HTML, plain text, JSON, and Markdown.


## License and Attribution

- **Dataset License:**  
  - The dataset is derived from Wikipedia content and is subject to the [Creative Commons Attribution-ShareAlike (CC BY-SA)](https://creativecommons.org/licenses/by-sa/3.0/) license.
  
- **Code License:**  
  - The repository's code is licensed under the MIT License.

- **Inspiration and Acknowledgments:**  
  - This dataset was inspired by the **FreshWiki Dataset**, which focuses on the most-edited Wikipedia pages between February 2022 and September 2023.
  - We acknowledge the FreshWiki authors for their contributions and for providing a data construction pipeline that informed several of our processing scripts.

  - For more details on the original dataset and pipeline, please refer to:
    > Yijia Shao et al. (2024). "Assisting in Writing Wikipedia-like Articles From Scratch with Large Language Models." [NAACL HLT 2024](https://arxiv.org/abs/2402.14207).

- **Additional Attribution:**  
  - Please ensure proper attribution to Wikipedia, ScienceDirect, and any other contributors when using or referencing this dataset.