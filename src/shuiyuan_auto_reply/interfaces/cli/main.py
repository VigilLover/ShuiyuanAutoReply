import argparse
import asyncio
import logging
import sys

from shuiyuan_auto_reply.bootstrap.deployment import (
    add_config_arguments,
    load_deployment,
)


def configure_logging():
    from shuiyuan_auto_reply.interfaces.worker.main import (
        configure_logging as configure,
    )

    configure()


async def run_worker(persona):
    import signal

    from shuiyuan_auto_reply.interfaces.worker.main import run_worker as run

    loop = asyncio.get_running_loop()
    task = asyncio.current_task()
    installed = False
    if sys.platform != "win32":
        loop.add_signal_handler(signal.SIGTERM, task.cancel)
        installed = True
    try:
        await run(persona)
    finally:
        if installed:
            loop.remove_signal_handler(signal.SIGTERM)


async def _run_worker_with_web(persona: str, host: str, port: int) -> None:
    try:
        import uvicorn
    except ImportError as exc:
        raise SystemExit(
            "FastAPI server dependencies are not installed. "
            "Install with: pip install 'shuiyuan-auto-reply[server]'"
        ) from exc

    server = uvicorn.Server(
        uvicorn.Config(
            "shuiyuan_auto_reply.interfaces.api.app:app",
            host=host,
            port=port,
            reload=False,
        )
    )

    async def supervise_worker():
        while True:
            try:
                await run_worker(persona)
                return
            except Exception:
                logging.exception(
                    "Forum worker unavailable; management stays online, retrying in 30s"
                )
                await asyncio.sleep(30)

    worker_task = asyncio.create_task(supervise_worker(), name="forum-worker")
    web_task = asyncio.create_task(server.serve(), name="management-web")
    try:
        done, _ = await asyncio.wait(
            {worker_task, web_task}, return_when=asyncio.FIRST_COMPLETED
        )
        if worker_task in done:
            server.should_exit = True
            await web_task
            await worker_task
            return

        web_error = web_task.exception()
        if web_error is not None:
            logging.error(
                "管理站启动或运行失败；论坛 Worker 将继续运行：%s",
                web_error,
            )
            await worker_task
        elif server.should_exit:
            worker_task.cancel()
            await asyncio.gather(worker_task, return_exceptions=True)
        else:
            logging.error("管理站意外停止；论坛 Worker 将继续运行。")
            await worker_task
    finally:
        server.should_exit = True
        for task in (web_task, worker_task):
            if not task.done():
                task.cancel()
        await asyncio.gather(web_task, worker_task, return_exceptions=True)


def bot_main() -> None:
    parser = argparse.ArgumentParser(description="Run the Shuiyuan auto-reply bot.")
    add_config_arguments(parser)
    parser.add_argument(
        "persona",
        nargs="?",
        default="wolf_lumine",
        help="指定要运行的人物模型 (例如 wolf_lumine), 如果不指定则默认为 wolf_lumine",
    )
    parser.add_argument(
        "--web",
        action="store_true",
        help="同时启动本地 FastAPI 管理站和 Vue 页面（默认关闭）",
    )
    parser.add_argument("--web-host", default=None, help="管理站监听地址")
    parser.add_argument("--web-port", type=int, default=None, help="管理站监听端口")
    args = parser.parse_args()
    deployment = load_deployment(args.config, args.profile)
    args.web_host = args.web_host or deployment.section("web")["host"]
    args.web_port = args.web_port or deployment.section("web")["port"]
    configure_logging()
    print(f"当前使用的人物模型: {args.persona}")
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    if args.web:
        asyncio.run(_run_worker_with_web(args.persona, args.web_host, args.web_port))
    else:
        asyncio.run(run_worker(args.persona))


def api_main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the Shuiyuan management API and UI."
    )
    add_config_arguments(parser)
    parser.add_argument("--host", default=None)
    parser.add_argument("--port", type=int, default=None)
    args = parser.parse_args()
    deployment = load_deployment(args.config, args.profile)
    args.host = args.host or deployment.section("web")["host"]
    args.port = args.port or deployment.section("web")["port"]
    try:
        import uvicorn
    except ImportError as exc:
        raise SystemExit(
            "FastAPI server dependencies are not installed. "
            "Install with: pip install 'shuiyuan-auto-reply[server]'"
        ) from exc
    uvicorn.run(
        "shuiyuan_auto_reply.interfaces.api.app:app",
        host=args.host,
        port=args.port,
        reload=False,
    )
