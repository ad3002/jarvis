from typing import List, Dict, Optional
import openai
from openai import AsyncOpenAI
import os
import json
import logging
import datetime
from dataclasses import dataclass, asdict
import hashlib
from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown
from rich.syntax import Syntax

console = Console()

@dataclass
class HistoryEntry:
    timestamp: str
    task: str
    thoughts: str
    tool: str
    args: Dict
    result: str
    status: str

@dataclass
class Memory:
    max_entries: int = 10
    entries: List[Dict] = None

    def __post_init__(self):
        self.entries = self.entries or []

    def add(self, task: str, thoughts: str, result: str, status: str):
        entry = {
            "timestamp": datetime.datetime.now().isoformat(),
            "task": task,
            "thoughts": thoughts,
            "result": result,
            "status": status
        }
        self.entries.append(entry)
        if len(self.entries) > self.max_entries:
            self.entries.pop(0)

    def get_context(self) -> str:
        if not self.entries:
            return "No previous interactions."
        
        context = "Previous interactions:\n"
        for entry in self.entries:
            context += f"\nTask: {entry['task']}\n"
            context += f"Outcome: {'✓ Success' if entry['status'] == 'success' else '✗ Failed'}\n"
            if entry['status'] == 'error':
                context += f"Error: {entry['result']}\n"
        return context

@dataclass
class Personality:
    name: str = "Jarvis"
    tone: str = "professional"  # professional, friendly, humorous
    emoji_style: bool = True
    
    def __post_init__(self):
        self.tone_patterns = {
            "professional": {
                "greeting": "Greetings. How may I assist you?",
                "thinking": "Analyzing task...",
                "success": "Task completed successfully.",
                "error": "An error has occurred.",
                "emoji": {
                    "think": "🤔",
                    "success": "✅",
                    "error": "❌",
                    "working": "⚙️"
                },
                "suitable_for": ["technical", "complex", "critical", "security", "deployment"]
            },
            "friendly": {
                "greeting": "Hey there! Ready to help!",
                "thinking": "Let me think about this...",
                "success": "Great! All done!",
                "error": "Oops! Something went wrong.",
                "emoji": {
                    "think": "💭",
                    "success": "🌟",
                    "error": "😅",
                    "working": "💪"
                },
                "suitable_for": ["learning", "guidance", "explanation", "help", "simple"]
            },
            "humorous": {
                "greeting": "Beep boop! Your friendly AI is here!",
                "thinking": "Computing... *makes robot noises*",
                "success": "Mission accomplished! *virtual high five*",
                "error": "Whoopsie! Even AIs have bad days!",
                "emoji": {
                    "think": "🤖",
                    "success": "🎉",
                    "error": "🙈",
                    "working": "⚡"
                },
                "suitable_for": ["creative", "fun", "experimental", "casual", "exploration"]
            }
        }

    def choose_tone_for_task(self, task: str) -> str:
        task_lower = task.lower()
        scores = {tone: 0 for tone in self.tone_patterns.keys()}
        
        for tone, pattern in self.tone_patterns.items():
            for keyword in pattern["suitable_for"]:
                if keyword in task_lower:
                    scores[tone] += 1
                    
        # Default to professional if no clear match
        return max(scores.items(), key=lambda x: x[1])[0] or "professional"

    def format_message(self, message_type: str, message: str) -> str:
        tone = self.tone_patterns[self.tone]
        emoji = tone["emoji"][message_type] if self.emoji_style else ""
        return f"{emoji} {message}"

    def get_thinking_prompt(self) -> str:
        return self.tone_patterns[self.tone]["thinking"]

class Tool:
    def __init__(self, name: str, description: str, base_dir: str):
        self.name = name
        self.description = description
        self.base_dir = base_dir

    def resolve_path(self, filepath: str) -> str:
        if os.path.isabs(filepath):
            return filepath
        return os.path.join(self.base_dir, filepath)

    async def execute(self, *args, **kwargs):
        raise NotImplementedError

class CreateFileTool(Tool):
    def __init__(self, base_dir: str):
        super().__init__("CREATE_FILE", "Creates a new file with specified content", base_dir)
    
    async def execute(self, filepath: str, content: str):
        full_path = self.resolve_path(filepath)
        os.makedirs(os.path.dirname(full_path), exist_ok=True)
        with open(full_path, 'w') as f:
            f.write(content)
        return f"File created: {full_path}"

class RunFileTool(Tool):
    def __init__(self, base_dir: str):
        super().__init__("RUN_FILE", "Executes a file", base_dir)
    
    async def execute(self, filepath: str):
        full_path = self.resolve_path(filepath)
        if filepath.endswith('.py'):
            output = os.popen(f"python {full_path}").read()
        else:
            output = os.popen(f"./{full_path}").read()
        return output

class RunLinuxCommandTool(Tool):
    def __init__(self, base_dir: str):
        super().__init__("RUN_LINUX_COMMAND", "Executes a Linux command", base_dir)
    
    async def execute(self, command: str):
        # Change to base directory before executing command
        current_dir = os.getcwd()
        os.chdir(self.base_dir)
        output = os.popen(command).read()
        os.chdir(current_dir)
        return output

class Agent:
    def __init__(self, api_key: str, base_dir: str):
        self.base_dir = base_dir
        self.tools = {
            "CREATE_FILE": CreateFileTool(base_dir),
            "RUN_FILE": RunFileTool(base_dir),
            "RUN_LINUX_COMMAND": RunLinuxCommandTool(base_dir)
        }
        self.client = AsyncOpenAI(api_key=api_key)
        self.history = []
        self.memory = Memory()
        self.personality = Personality()
        self.setup_logging()
        self.console = Console()
        
    def setup_logging(self):
        log_dir = os.path.join(self.base_dir, 'logs')
        os.makedirs(log_dir, exist_ok=True)
        logging.basicConfig(
            filename=os.path.join(log_dir, f'agent_{datetime.datetime.now().strftime("%Y%m%d")}.log'),
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        
    def validate_command(self, command: str) -> bool:
        dangerous_commands = ['rm -rf', 'sudo', '> /dev', '| sh', '& ', ';']
        return not any(cmd in command for cmd in dangerous_commands)
    
    async def execute_sequence(self, tasks: List[str]) -> List[Dict]:
        results = []
        for task in tasks:
            try:
                result = await self.execute(task)
                results.append({"task": task, "result": result, "status": "success"})
            except Exception as e:
                results.append({"task": task, "result": str(e), "status": "error"})
                logging.error(f"Error executing task '{task}': {e}")
        return results
    
    def save_history(self):
        history_file = os.path.join(self.base_dir, 'history.json')
        with open(history_file, 'w') as f:
            json.dump([asdict(entry) for entry in self.history], f, indent=2)
    
    def load_history(self) -> Optional[List[HistoryEntry]]:
        history_file = os.path.join(self.base_dir, 'history.json')
        if os.path.exists(history_file):
            with open(history_file, 'r') as f:
                data = json.load(f)
                return [HistoryEntry(**entry) for entry in data]
        return []

    async def think(self, task: str) -> Dict:
        # Dynamically adjust personality based on task
        suggested_tone = self.personality.choose_tone_for_task(task)
        if self.personality.tone != suggested_tone:
            self.personality.tone = suggested_tone
            self.console.print(Panel(
                f"Adjusting tone to [bold blue]{suggested_tone}[/] for this task"
            ))

        self.console.print(Panel(
            self.personality.format_message("think", f"Thinking about task: [bold blue]{task}[/]")
        ))
        
        prompt = """Given the following task: {task}
Working directory: {base_dir}

{memory_context}

Respond in the style of {personality_desc}

Available tools and their required arguments:
1. CREATE_FILE:
   - filepath: str (path to the file to create)
   - content: str (content to write to the file)

2. RUN_FILE:
   - filepath: str (path to the file to execute)

3. RUN_LINUX_COMMAND:
   - command: str (command to execute)

Consider previous interactions when making decisions. If similar tasks failed before, try a different approach.

Provide your response in JSON format with the following structure:
{{
    "thoughts": "your step-by-step reasoning",
    "tool": "tool name to use",
    "args": {{
        "filepath": "path/to/file",  # for CREATE_FILE and RUN_FILE
        "content": "file content",    # for CREATE_FILE only
        "command": "command string"   # for RUN_LINUX_COMMAND only
    }}
}}

IMPORTANT: Use exact argument names as specified above.
Your response must be valid JSON.""".format(
            task=task,
            base_dir=self.base_dir,
            memory_context=self.memory.get_context(),
            personality_desc=f"a {self.personality.tone} assistant named {self.personality.name}"
        )
        
        response = await self.client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a helpful AI that always responds with valid JSON using the exact argument names specified"},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
        )
        
        content = response.choices[0].message.content.strip()
        try:
            parsed = json.loads(content)
            # Validate args match expected parameters
            tool_name = parsed["tool"]
            if tool_name == "CREATE_FILE":
                required = {"filepath", "content"}
                if not all(arg in parsed["args"] for arg in required):
                    raise ValueError(f"Missing required arguments for CREATE_FILE. Required: {required}")
            elif tool_name == "RUN_FILE":
                if "filepath" not in parsed["args"]:
                    raise ValueError("Missing required argument 'filepath' for RUN_FILE")
            elif tool_name == "RUN_LINUX_COMMAND":
                if "command" not in parsed["args"]:
                    raise ValueError("Missing required argument 'command' for RUN_LINUX_COMMAND")
            
            self.console.print(Panel(Markdown(f"💭 Thoughts: {parsed['thoughts']}")))
            self.console.print(f"🛠 Using tool: [bold green]{parsed['tool']}[/]")
            
            return parsed
        except json.JSONDecodeError as e:
            logging.error(f"Failed to parse JSON response: {content}")
            raise ValueError("Invalid JSON response from OpenAI") from e

    async def execute(self, task: str):
        try:
            logging.info(f"Executing task: {task}")
            decision = await self.think(task)
            
            # No need to parse JSON here anymore as think() now returns a dict
            tool_name = decision["tool"]
            if tool_name == "RUN_LINUX_COMMAND":
                if not self.validate_command(decision["args"]["command"]):
                    raise ValueError("Potentially dangerous command detected")
            
            tool = self.tools[tool_name]
            result = await tool.execute(**decision["args"])
            
            # Save to both history and memory
            entry = HistoryEntry(
                timestamp=datetime.datetime.now().isoformat(),
                task=task,
                thoughts=decision.get("thoughts", ""),
                tool=tool_name,
                args=decision["args"],
                result=result,
                status="success"
            )
            self.history.append(entry)
            self.save_history()
            self.memory.add(task, decision.get("thoughts", ""), result, "success")
            
            logging.info(f"Task completed successfully: {task}")
            
            # Pretty print the result based on content
            if isinstance(result, str) and result.startswith("File created:"):
                self.console.print(Panel(
                    self.personality.format_message("success", f"[bold green]{result}[/]")
                ))
            elif result.strip():
                if "```" in result or result.count('\n') > 1:
                    self.console.print(Panel(Syntax(result, "python")))
                else:
                    self.console.print(Panel(
                        self.personality.format_message("working", f"Output: {result}")
                    ))
            
            return result
            
        except Exception as e:
            logging.error(f"Error executing task '{task}': {e}")
            entry = HistoryEntry(
                timestamp=datetime.datetime.now().isoformat(),
                task=task,
                thoughts="",
                tool="",
                args={},
                result=str(e),
                status="error"
            )
            self.history.append(entry)
            self.save_history()
            self.memory.add(task, "", str(e), "error")
            self.console.print(
                self.personality.format_message("error", f"[bold red]Error:[/] {str(e)}"),
                style="red"
            )
            raise
