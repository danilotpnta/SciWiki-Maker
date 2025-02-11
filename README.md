# SciWiki Dataset Construction

A curated collection of 2k high-quality Wikipedia articles aligned with ScienceDirect topics. This repository houses scripts to fetch URLs, filter articles using the ORES API, and extract content in multiple formats—laying the foundation for generating scientific articles with LLMs.


## Project Overview

The SciWiki dataset pipeline performs the following tasks:
- **URL Retrieval:** Gathers Wikipedia URLs corresponding to ScienceDirect topics.
- **Article Filtering:** Uses the ORES API to evaluate and filter articles for quality.
- **Content Extraction:** Extracts and stores article content in multiple formats (HTML, plain text, JSON, Markdown).

---

## Data Directory Structure

After running the dataset construction process, your `data/` folder will be organized as follows:

```
data/
├── urls/
│   └── topics_urls.json
├── ores/
│   └── topics_ores_scores.json
└── articles/
    ├── html/
    ├── txt/
    ├── json/
    └── md/
```


## Installation

### Prerequisites

Ensure you have [Conda](https://docs.conda.io/en/latest/) installed or other virtual environment managers like `venv`.

### Steps

1. **Create and Activate the Environment**

   ```sh
   conda create -n myenv python=3.11
   conda activate myenv
   ```

2. **Install Required Packages**

   ```sh
   pip install -r requirements.txt
   ```


## Dataset Construction and Usage

To construct the SciWiki dataset, run the following command:

```bash
python src/build_dataset.py \
  --topics_file data/topics_all.json \
  --output_dir data \
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