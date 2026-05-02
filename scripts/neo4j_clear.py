import asyncio
import dotenv
import os
import sys
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

# Set the correct path to allow imports from src and load env vars
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
dotenv.load_dotenv()

from src.database.neo4j_mgr import global_async_neo4j_manager

async def clear_database():
    logging.info("开始清空 Neo4j 数据库...")
    try:
        # Connect to the correct database and delete all nodes and relationships
        async with global_async_neo4j_manager.driver.session(database="neo4j") as session:
            await session.run("MATCH (n) DETACH DELETE n")
        logging.info("数据库已全部清空！")
    except Exception as e:
        logging.error(f"清空失败: {e}")
    finally:
        await global_async_neo4j_manager.close()

if __name__ == "__main__":
    asyncio.run(clear_database())
