import os
from pathlib import Path

# Get the project root directory (2 levels up from constants.py)
project_root = Path(__file__).parent.parent.parent
data_dir = os.path.join(project_root, "data")

# Input paths
topics_json = os.path.join(data_dir, "topics.json")
topics_urls_json = os.path.join(data_dir, "topics_urls.json")
topic_ores_scores_json = os.path.join(data_dir, "topics_ores_scores.json")

# Output paths
urls_dir = os.path.join(data_dir, "urls")
ores_dir = os.path.join(data_dir, "ores")
articles_dir = os.path.join(data_dir, "articles")

for directory in [data_dir, urls_dir, ores_dir, articles_dir]:
    os.makedirs(directory, exist_ok=True)