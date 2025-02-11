import re
import os
import time
import logging
from tqdm import tqdm

import pandas as pd

import requests
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry
from concurrent.futures import ThreadPoolExecutor, as_completed

from config.constants import topics_urls_json, data_dir
from .utils import load_json, dump_json, format_args
import argparse

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)-8s : %(message)s")


def create_session():
    session = requests.Session()
    retry_strategy = Retry(
        total=5, backoff_factor=2, status_forcelist=[429, 500, 502, 503, 504]
    )
    adapter = HTTPAdapter(
        max_retries=retry_strategy, pool_connections=25, pool_maxsize=25
    )
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    return session


def print_stats(df):
    num_high_quality = df["predicted_class"].isin(["B", "GA", "FA"]).sum()
    percentage_high_quality = (num_high_quality / len(df)) * 100

    print(
        f"\n=============== SUMMARY ===============\n"
        f"- {num_high_quality} out of {len(df)} Topics "
        f"({percentage_high_quality:.2f}%) are of high-quality class B, GA, or FA."
    )


def print_warning(topics_urls_json):
    if not os.path.exists(topics_urls_json):
        raise FileNotFoundError(
            f"""
            Error: The required file '{topics_urls_json}' does not exist.
            Please ensure that the file is generated by running:
            get_wikipedia_urls.py
            Once the file is created, re-run this script.
            """
        )


def get_html_content(session, url):
    try:
        time.sleep(0.5)  # Add delay between requests
        response = session.get(url, timeout=10)
        response.raise_for_status()
        return response.text
    except requests.RequestException as e:
        logger.error(f"Error fetching the URL {url}: {e}")
        return None


def call_ORES_api(session, revids, verbose=False):
    base_url = "https://ores.wikimedia.org/v3/scores/enwiki"
    params = {"models": "articlequality", "revids": revids}

    try:
        time.sleep(0.5)  # Add delay between API calls
        response = session.get(base_url, params=params, timeout=10)
        response.raise_for_status()
        json_response = response.json()
        if verbose:
            logger.debug(f"Response for revids {revids}: {json_response}")
        return json_response["enwiki"]["scores"][f"{revids}"]["articlequality"]["score"]
    except (requests.RequestException, KeyError) as e:
        logger.error(f"Error processing revids {revids}: {e}")
        return None


def process_topic(session, topic_url_pair):
    topic, url = topic_url_pair
    if not url:
        return None

    try:
        url_topic = url.split("/wiki/")[-1]

        # Get revision ID
        history_url = (
            f"https://en.wikipedia.org/w/index.php?title={url_topic}&action=history"
        )
        html_content = get_html_content(session, history_url)
        if html_content is None:
            return None

        match = re.search(r'"wgCurRevisionId":(\d+)', html_content)
        if not match:
            logger.error(f"Cannot find revids for {topic}.")
            return None

        revids = match.group(1)
        predicted_quality = call_ORES_api(session, revids)

        if predicted_quality is None:
            return None

        return {
            "topic": topic,
            "url": url,
            "predicted_class": predicted_quality["prediction"],
            "predicted_scores": predicted_quality["probability"],
        }
    except Exception as e:
        logger.error(f"Error processing topic {topic}: {e}")
        return None


def batch_process_topics(topic_urls_dict, results_path, max_workers=8, batch_size=20):
    session = create_session()

    # Initialize or load existing results
    if os.path.exists(results_path):
        results = load_json(results_path)
        processed_topics = {item["topic"] for item in results}
    else:
        results = []
        processed_topics = set()

    # Filter topics to process
    topics_to_process = {
        topic: url
        for topic, url in topic_urls_dict.items()
        if topic not in processed_topics
    }

    logger.info(f"Found {len(results)} existing results")
    logger.info(f"Processing {len(topics_to_process)} remaining topics")

    # Process in batches
    items = list(topics_to_process.items())
    for i in range(0, len(items), batch_size):
        batch = items[i : i + batch_size]

        # Add delay between batches
        if i > 0:
            time.sleep(3)

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(process_topic, session, item): item for item in batch
            }

            for future in tqdm(
                as_completed(futures),
                total=len(batch),
                desc=f"Processing batch {i//batch_size + 1}/{(len(items) + batch_size - 1)//batch_size}",
            ):
                result = future.result()
                if result:
                    results.append(result)
                    # Save progress after each successful result
                    dump_json(results, results_path)

    return results


def main(args):
    print_warning(args.topics_urls_json)

    # Define paths for intermediate and final results
    suffix = "_all" if "all" in os.path.basename(args.topics_urls_json) else ""
    filename = os.path.splitext(os.path.basename(args.topics_urls_json))[0]
    results_path = os.path.join(args.output_dir, f"{filename}_ores_scores{suffix}.json")
    csv_path = os.path.join(args.output_dir, f"{filename}_ores_scores{suffix}.csv")

    # Load topics
    topics_with_urls_dict = load_json(args.topics_urls_json)

    # Process topics
    results = batch_process_topics(
        topics_with_urls_dict,
        results_path,
        max_workers=args.max_workers,
        batch_size=args.batch_size,
    )

    # Convert to DataFrame
    df = pd.DataFrame(results)

    # Print statistics
    print_stats(df)

    # Filter high-quality articles if requested
    if args.extract_HQ_articles:
        high_quality_classes = {"B", "GA", "FA"}
        df = df[df["predicted_class"].isin(high_quality_classes)]

    # Save final results
    df.to_csv(csv_path, index=False)
    dump_json(results, results_path, ensure_ascii=False)

    print(f"Files saved to:\n- CSV: {csv_path}\n- JSON: {results_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Get ORES scores for Wikipedia articles."
    )
    parser.add_argument(
        "--topics_urls_json",
        type=str,
        default=topics_urls_json,
        help="Path to the JSON file containing topics.",
    )
    parser.add_argument(
        "--extract_HQ_articles",
        action="store_true",
        default=True,
        help="Extract high-quality Wikipedia articles based on ORES scores.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=data_dir,
        help="Directory where to save the ORES scores JSON file.",
    )
    parser.add_argument(
        "--max_workers",
        type=int,
        default=1,
        help="Maximum number of worker threads (default: 1 no parallelism)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=20,
        help="Number of topics to process in each batch (default: 20)",
    )

    args = parser.parse_args()
    print(format_args(args))

    main(args)
