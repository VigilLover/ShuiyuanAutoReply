import argparse
import asyncio
import logging
import sys

import dotenv

from shuiyuan_auto_reply.database.neo4j_mgr import create_global_async_neo4j_manager


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

dotenv.load_dotenv()


async def clear_user_sentences(username: str) -> int:
    logging.info("开始清空 Neo4j 中用户/角色 %s 的历史语料...", username)
    neo4j_manager = await create_global_async_neo4j_manager(strict=True)
    if neo4j_manager is None:
        raise RuntimeError("NEO4J_DB_URL is not configured")

    deleted_count = await neo4j_manager.clear_sentences(username)
    logging.info("已删除 %s 条 Sentence 节点。", deleted_count)
    return deleted_count


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Clear Neo4j sentence archive for a specific Shuiyuan user/persona."
    )
    parser.add_argument("username", nargs="?", help="要清空的用户名/角色名")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    username = args.username or input("请输入要清空的用户名/角色名: ").strip()
    if not username:
        logging.error("用户名不能为空")
        sys.exit(1)

    try:
        asyncio.run(clear_user_sentences(username))
    except Exception as exc:
        logging.error("清空失败: %s", exc)
        raise


if __name__ == "__main__":
    main()
