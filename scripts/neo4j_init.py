import asyncio
import logging
import os
import re

import dotenv
import pandas as pd

# Setup the logging configuration
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

# Load all environment variables from the .env file
dotenv.load_dotenv()

from shuiyuan_auto_reply.constants import settings
from shuiyuan_auto_reply.database.neo4j_mgr import create_global_async_neo4j_manager


async def init_database(username: str):
    """Initialize the Neo4j database"""
    try:
        logging.info("Initializing Neo4j database...")
        neo4j_manager = await create_global_async_neo4j_manager(strict=True)
        if neo4j_manager is None:
            raise RuntimeError("NEO4J_DB_URL is not configured")
        # Initialize the Neo4j database
        global_async_neo4j_manager.userid = username
        await global_async_neo4j_manager.initialize()

        # The signature has to be removed before storing the sentences
        sig_re = r"<div data-signature>.*?</div>"

        # Try to open the CSV file and import data
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        file_path = os.path.join(project_root, "user_archive", username, "user_archive.csv")
            
        if os.path.exists(file_path):
            # Load the CSV data
            logging.info(f"Importing data from {file_path}...")
            df = pd.read_csv(file_path)
            # Some data should not be imported, filter them out
            data_to_import = []
            for idx, raw in enumerate(df["post_raw"]):
                # If NaN, skip
                if pd.isna(raw):
                    continue
                # Auto-reply posts should not be imported
                if settings.contains_auto_reply_tag(str(raw)):
                    continue
                # For other posts, import them into the database
                # But signature needs to be removed from the post
                data_to_import.append(
                    re.sub(sig_re, "", str(raw), flags=re.DOTALL).strip()
                )
            # Make every record unique
            data_to_import = list(set(data_to_import))
            # Log the number of records to be imported
            logging.info(f"Number of records to import: {len(data_to_import)}")
            # Wait for all import routines to complete
            await neo4j_manager.store_sentences(data_to_import)
            logging.info("Data imported successfully!")
        else:
            logging.warning(f"CSV file {file_path} not found. Skipping data import.")
        logging.info("Neo4j database initialized successfully!")
    except Exception as e:
        logging.error(f"Error initializing Neo4j database: {e}")
        raise


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    username = input("请输入要初始化的用户名/角色名: ").strip()
    if not username:
        logging.error("用户名不能为空")
        sys.exit(1)
    asyncio.run(init_database(username))
