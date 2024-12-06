import asyncio
import os
from agent import Agent
from rich.console import Console
from rich.prompt import Prompt
from rich.panel import Panel

console = Console()

async def main():
    api_key = os.getenv("OPENAI_KEY")
    if not api_key:
        raise ValueError("Please set OPENAI_API_KEY environment variable")
    
    agent = Agent(api_key, base_dir=os.path.dirname(os.path.abspath(__file__)))
    
    console.print(Panel(f"[bold blue]🤖 {agent.personality.name} - AI Assistant[/]"))
    console.print(Panel(agent.personality.tone_patterns[agent.personality.tone]["greeting"]))
    
    while True:
        task = Prompt.ask("\n[bold green]Enter your task[/] (or 'exit' to quit)")
        
        if task.lower() == 'exit':
            break
            
        try:
            result = await agent.execute(task)
        except Exception as e:
            console.print(
                agent.personality.format_message("error", f"[bold red]Error:[/] {str(e)}"),
                style="red"
            )

if __name__ == "__main__":
    asyncio.run(main())
