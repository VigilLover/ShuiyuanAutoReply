import argparse
import asyncio
import sys

from shuiyuan_auto_reply.interfaces.worker.main import configure_logging, run_worker


def bot_main() -> None:
    parser = argparse.ArgumentParser(description="Run the Shuiyuan auto-reply bot.")
    parser.add_argument(
        "persona",
        nargs="?",
        default="wolf_lumine",
        help="指定要运行的人物模型 (例如 wolf_lumine), 如果不指定则默认为 wolf_lumine",
    )
    args = parser.parse_args()
    configure_logging()
    print(f"当前使用的人物模型: {args.persona}")
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(run_worker(args.persona))


def api_main() -> None:
    try:
        import uvicorn
    except ImportError as exc:
        raise SystemExit(
            "FastAPI server dependencies are not installed. "
            "Install with: pip install 'shuiyuan-auto-reply[server]'"
        ) from exc
    uvicorn.run(
        "shuiyuan_auto_reply.interfaces.api.app:app",
        host="0.0.0.0",
        port=11451,
        reload=False,
    )
