import argparse
import asyncio
import dotenv
import logging

# Setup the logging configuration
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

# Load all environment variables from the .env file
dotenv.load_dotenv()

# Add the parent directory to the system path for module resolution
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


from src.shuiyuan.objects import TimeInADay
from src.shuiyuan.shuiyuan_model import ShuiyuanModel
from models.mention_model import MentionModel
from models.tarot_topic_model import TarotTopicModel
from models.stock_topic_model import StockTopicModel
from models.record_topic_model import RecordTopicModel


async def main(persona: str):
    """
    Main function to run the ShuiyuanModel.
    This function initializes the model and retrieves the session.
    """

    async with await ShuiyuanModel.create() as model:
        # Let's try to get the post streams
        # Here bot_username is fixed to "wolf_lumine" so it listens to mentions for wolf_lumine,
        # but the persona argument controls the trigger word, nickname, prompt and neo4j filter.
        mention_model = MentionModel(model, bot_username="wolf_lumine", persona=persona)
        # tarot_topic_model = TarotTopicModel(model, 393697)
        # stock_topic_model = StockTopicModel(model, 392286)
        # record_topic_model = RecordTopicModel(model, 441566)

        # stock_topic_model.add_time_routine(TimeInADay(hour=9, minute=30), True)
        # stock_topic_model.add_time_routine(TimeInADay(hour=11, minute=30), True)
        # stock_topic_model.add_time_routine(TimeInADay(hour=15, minute=0), True)
        # record_topic_model.add_time_routine(TimeInADay(hour=0, minute=0), False)

        # stock_topic_model.start_scheduler()
        # record_topic_model.start_scheduler()

        await asyncio.gather(
            mention_model.watch_new_action_routine(),
            # tarot_topic_model.watch_new_post_routine(),
            # stock_topic_model.watch_new_post_routine(),
            # record_topic_model.watch_new_post_routine(),
        )

        # stock_topic_model.stop_scheduler()
        # record_topic_model.stop_scheduler()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the Shuiyuan auto-reply bot.")
    parser.add_argument(
        "persona",
        nargs="?",
        default="wolf_lumine",
        help="指定要运行的人物模型 (例如 wolf_lumine), 如果不指定则默认为 wolf_lumine",
    )
    args = parser.parse_args()

    print(f"当前使用的人物模型: {args.persona}")
    
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

    asyncio.run(main(args.persona))
