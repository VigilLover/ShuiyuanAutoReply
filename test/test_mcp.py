import os
import asyncio
import logging
from langchain_mcp_adapters.client import MultiServerMCPClient

import dotenv
import pytest
dotenv.load_dotenv()

pytestmark = pytest.mark.live

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

async def _mcp_connection():
    mcp_server_url = os.getenv("MCP_SERVER_URL")
    
    if not mcp_server_url:
        logging.error("MCP_SERVER_URL is not set. Please set the environment variable or add it to .env.")
        return
        
    logging.info(f"Attempting to connect to MCP server at: {mcp_server_url}")
    
    try:
        client = MultiServerMCPClient(
            {
                "default": {
                    "transport": "sse",
                    "url": mcp_server_url,
                }
            }
        )
        
        logging.info("MultiServerMCPClient created. Fetching tools...")
        # Get tools from the MCP server
        mcp_tools = await client.get_tools()
        
        if not mcp_tools:
            logging.warning("Connection was successful, but no tools were returned.")
        else:
            logging.info(f"Successfully loaded {len(mcp_tools)} tools from MCP server.")
            logging.info("==> [MCP Tools Loaded via HTTP]:")
            for tool in mcp_tools:
                logging.info(f"{tool.name}: {tool.description}")
                
    except Exception as e:
        logging.error(f"Failed to connect to MCP server or fetch tools: {e}")


def test_mcp_connection():
    asyncio.run(_mcp_connection())

if __name__ == "__main__":
    asyncio.run(_mcp_connection())
