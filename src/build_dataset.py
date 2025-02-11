import sys
import os

# # Determine the project root directory (one level up from the src folder)
# project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
# if project_root not in sys.path:
#     sys.path.insert(0, project_root)

import argparse
import logging
from typing import List

from src.modules import get_wikipedia_urls, get_ores_scores, get_wikipedia_articles

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler("dataset_build.log")],
)
logger = logging.getLogger(__name__)


class DatasetBuilder:
    def __init__(self, output_dir: str):
        """
        Initialize the DatasetBuilder with the output directory.

        Args:
            output_dir: Directory where all output files will be saved.
                        (By default, this will be the top-level 'data' folder.)
        """
        self.output_dir = os.path.abspath(output_dir)
        os.makedirs(self.output_dir, exist_ok=True)

        self.files = {
            "topics_urls": os.path.join(self.output_dir, "topics_urls.json"),
            "ores_scores": os.path.join(self.output_dir, "topics_ores_scores.json"),
            "ores_scores_csv": os.path.join(self.output_dir, "topics_ores_scores.csv"),
        }

    def get_wikipedia_urls(self, topics_file: str, max_workers: int = 1) -> None:

        logger.info("Step 1: Getting Wikipedia URLs...")
        args = argparse.Namespace(
            topics_json=topics_file, output_dir=self.output_dir, max_workers=max_workers
        )
        get_wikipedia_urls.main(args)
        logger.info("Completed getting Wikipedia URLs")

    def get_ores_scores(self, max_workers: int = 1, batch_size: int = 20) -> None:

        logger.info("Step 2: Getting ORES scores...")
        args = argparse.Namespace(
            topics_urls_json=self.files["topics_urls"],
            extract_HQ_articles=True,
            output_dir=self.output_dir,
            max_workers=max_workers,
            batch_size=batch_size,
        )
        get_ores_scores.main(args)
        logger.info("Completed getting ORES scores")

    def get_wikipedia_articles(self, file_types: List[str] = None) -> None:

        logger.info("Step 3: Getting Wikipedia articles...")
        if file_types is None:
            file_types = ["html", "txt", "json", "md"]

        articles_dir = os.path.join(self.output_dir, "articles")
        os.makedirs(articles_dir, exist_ok=True)

        args = argparse.Namespace(
            batch_path=self.files["ores_scores_csv"],
            output_dir=articles_dir,
            files_types_to_save=file_types,
            url=None,
        )
        get_wikipedia_articles.main(args)
        logger.info("Completed getting Wikipedia articles")

    def build(
        self, topics_file: str, max_workers: int = 1, file_types: List[str] = None
    ) -> None:
        """
        Execute the full dataset building pipeline.
        """
        try:
            # Step 1: Get Wikipedia URLs
            self.get_wikipedia_urls(topics_file, max_workers)

            # Step 2: Get ORES scores
            self.get_ores_scores(max_workers)

            # Step 3: Get Wikipedia articles
            self.get_wikipedia_articles(file_types)
            logger.info("Dataset build completed successfully!")

        except Exception as e:
            logger.error(f"Error building dataset: {str(e)}")
            raise


def main():
    builder = DatasetBuilder(args.output_dir)
    builder.build(args.topics_file, args.max_workers, args.file_types)


if __name__ == "__main__":

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(project_root, "data")

    parser = argparse.ArgumentParser(description="Build the SciWiki dataset")
    parser.add_argument(
        "--topics_file",
        type=str,
        required=True,
        help="Path to JSON file containing initial topics",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=data_dir,
        help="Directory where all output files will be saved (default: 'data' folder at project root)",
    )
    parser.add_argument(
        "--max_workers",
        type=int,
        default=1,
        help="Maximum number of workers for both URL fetching and ORES score fetching (default: 1 for non-parallelism)",
    )
    parser.add_argument(
        "--file_types",
        nargs="+",
        default=["txt", "json"],
        choices=["html", "txt", "json", "md"],
        help="Types of files to save for each article",
    )
    """
    python /home/toapantabarahonad/SciWiki-Maker/src/build_dataset.py --topics_file  data/topics.json
    """
    args = parser.parse_args()
    main()
