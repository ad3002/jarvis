import asyncio
import os
from agent import AutonomousAgent
from rich.console import Console
from rich.prompt import Prompt, Confirm
from rich.panel import Panel

console = Console()

async def main():
    api_key = os.getenv("OPENAI_KEY")
    if not api_key:
        raise ValueError("Please set OPENAI_API_KEY environment variable")
    
    agent = AutonomousAgent(api_key, base_dir=os.path.dirname(os.path.abspath(__file__)))
    
    console.print(Panel(f"[bold blue]🤖 {agent.personality.name} - AI Assistant[/]"))
    console.print(Panel(
        """[bold green]Verbose Mode Active[/]
        • Showing detailed task breakdowns
        • Displaying thought processes
        • Tracking execution statistics
        • Visualizing task dependencies"""
    ))
    console.print(Panel(agent.personality.tone_patterns[agent.personality.tone]["greeting"]))
    
    autonomous_mode = Confirm.ask("\n[bold blue]Enable autonomous mode?[/]")
    if autonomous_mode:
        agent.start_autonomous_mode()
    
    while True:
        task = Prompt.ask("\n[bold green]Enter your task[/] (or 'exit' to quit)")
        
        if task.lower() == 'exit':
            break
            
        try:
            if autonomous_mode:
                await agent.autonomous_execute(task)
            else:
                await agent.execute(task)
        except Exception as e:
            console.print(
                agent.personality.format_message("error", f"[bold red]Error:[/] {str(e)}"),
                style="red"
            )

if __name__ == "__main__":
    asyncio.run(main())
