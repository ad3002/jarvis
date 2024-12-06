import asyncio
import os
from agent import Agent

# Set base working directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

async def main():
    api_key = os.getenv("OPENAI_KEY")
    if not api_key:
        raise ValueError("Please set OPENAI_API_KEY environment variable")
    
    agent = Agent(api_key, base_dir=BASE_DIR)
    
    # Example task
    task = "Create a simple Python script that prints 'Hello, World!'"
    result = await agent.execute(task)
    print(f"Task result: {result}")

if __name__ == "__main__":
    asyncio.run(main())
