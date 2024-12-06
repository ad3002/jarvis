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
        self.console = Console()
    
    def _display_file_content(self, filepath: str, content: str):
        """Display the created file content with syntax highlighting."""
        file_ext = os.path.splitext(filepath)[1].lower()
        language = {
            '.py': 'python',
            '.js': 'javascript',
            '.ts': 'typescript',
            '.html': 'html',
            '.css': 'css',
            '.json': 'json',
            '.md': 'markdown',
            '.yaml': 'yaml',
            '.yml': 'yaml',
        }.get(file_ext, 'text')
        
        self.console.print("\n[bold blue]📄 Created File Content:[/]")
        self.console.print(
            Panel(
                Syntax(
                    content,
                    language,
                    theme="monokai",
                    line_numbers=True
                ),
                title=filepath
            )
        )
    
    async def execute(self, filepath: str, content: str):
        full_path = self.resolve_path(filepath)
        os.makedirs(os.path.dirname(full_path), exist_ok=True)
        with open(full_path, 'w') as f:
            f.write(content)
        self._display_file_content(filepath, content)
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

    def sanitize_json_string(self, text: str) -> str:
        """Remove invalid Unicode characters and normalize JSON string."""
        # Remove any unknown Unicode characters
        cleaned = ''.join(char for char in text if ord(char) < 0x10000)
        # Normalize quotes and whitespace
        cleaned = cleaned.replace('"', '"').replace('"', '"')
        cleaned = cleaned.replace("'", "'").replace("'", "'")
        return cleaned.strip()

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
            # Sanitize the JSON string before parsing
            cleaned_content = self.sanitize_json_string(content)
            parsed = json.loads(cleaned_content)
            
            # Additional validation to ensure all strings are properly encoded
            def validate_strings(obj):
                if isinstance(obj, str):
                    return obj.encode('utf-8', errors='replace').decode('utf-8')
                elif isinstance(obj, dict):
                    return {k: validate_strings(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [validate_strings(item) for item in obj]
                return obj
            
            parsed = validate_strings(parsed)
            
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
            logging.error(f"Failed to parse JSON response: {cleaned_content}")
            logging.error(f"JSON Error: {str(e)}")
            raise ValueError(f"Invalid JSON response from OpenAI: {str(e)}") from e
        except Exception as e:
            logging.error(f"Unexpected error processing response: {str(e)}")
            raise ValueError(f"Error processing AI response: {str(e)}") from e

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

@dataclass
class Task:
    description: str
    priority: int = 1
    dependencies: List[str] = None
    completed: bool = False
    result: Optional[str] = None

    def __post_init__(self):
        self.dependencies = self.dependencies or []

class TaskPlanner:
    def __init__(self):
        self.tasks = []
        self.console = Console()
        
    def _display_task_tree(self, tasks: List[Task]):
        """Display task dependencies as a tree."""
        from rich.tree import Tree
        
        task_tree = Tree("[bold blue]📋 Task Breakdown")
        task_dict = {task.description: task for task in tasks}
        
        def add_task_node(parent, task):
            status = "✅" if task.completed else "⏳"
            node = parent.add(f"{status} [{'green' if task.completed else 'yellow'}]{task.description}[/] (Priority: {task.priority})")
            for dep in task.dependencies:
                if dep in task_dict:
                    add_task_node(node, task_dict[dep])
        
        root_tasks = [t for t in tasks if not t.dependencies]
        for task in root_tasks:
            add_task_node(task_tree, task)
            
        self.console.print(Panel(task_tree))
        
    async def plan_tasks(self, main_task: str, client: AsyncOpenAI) -> List[Task]:
        self.console.print(Panel("[bold blue]🤔 Planning task breakdown...[/]"))
        prompt = f"""Break down this task into smaller subtasks: {main_task}
        
        Return a JSON array of subtasks in this format:
        [{{
            "description": "subtask description",
            "priority": priority_number (1-5),
            "dependencies": ["description of dependent task 1", "description of dependent task 2"]
        }}]
        
        Ensure subtasks are atomic and can be executed independently when dependencies are met."""
        
        response = await client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a task planning AI. Break down complex tasks into smaller, manageable subtasks."},
                {"role": "user", "content": prompt}
            ]
        )
        
        content = response.choices[0].message.content
        subtasks_data = json.loads(content)
        tasks = [Task(**task_data) for task_data in subtasks_data]
        
        self.console.print("[bold green]📊 Task Analysis:[/]")
        self.console.print(f"Total subtasks: {len(tasks)}")
        self.console.print(f"Independent tasks: {len([t for t in tasks if not t.dependencies])}")
        self._display_task_tree(tasks)
        
        return tasks

class AutonomousAgent(Agent):
    def __init__(self, api_key: str, base_dir: str):
        super().__init__(api_key, base_dir)
        self.planner = TaskPlanner()
        self.active = False
        self.task_queue = []
        self.execution_stats = {
            "total_tasks": 0,
            "completed_tasks": 0,
            "failed_tasks": 0,
            "retries": 0
        }
        
    def start_autonomous_mode(self):
        """Enable autonomous mode."""
        self.active = True
        self.console.print(Panel(
            self.personality.format_message("success", "[bold green]Autonomous mode activated[/]")
        ))
        
    def stop_autonomous_mode(self):
        """Disable autonomous mode."""
        self.active = False
        self.console.print(Panel(
            self.personality.format_message("error", "[bold red]Autonomous mode deactivated[/]")
        ))

    def _display_execution_stats(self):
        """Display current execution statistics."""
        from rich.table import Table
        
        stats_table = Table(title="Execution Statistics")
        stats_table.add_column("Metric", style="cyan")
        stats_table.add_column("Value", style="magenta")
        
        for key, value in self.execution_stats.items():
            stats_table.add_row(key.replace('_', ' ').title(), str(value))
            
        self.console.print(stats_table)

    async def evaluate_result(self, task: str, result: str) -> Dict:
        """Enhanced evaluation with detailed feedback."""
        prompt = f"""Task: {task}
        Result: {result}
        
        Provide detailed evaluation in JSON format:
        {{
            "success": true/false,
            "confidence": 0-100,
            "reason": "detailed explanation",
            "suggestions": ["improvement suggestion 1", "improvement suggestion 2"],
            "warnings": ["warning 1", "warning 2"]
        }}"""
        
        self.console.print(Panel("[bold yellow]🔍 Evaluating task result...[/]"))
        
        response = await self.client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a task evaluation AI. Provide detailed analysis."},
                {"role": "user", "content": prompt}
            ]
        )
        
        evaluation = json.loads(response.choices[0].message.content)
        
        # Display evaluation details
        self.console.print(Panel.fit(
            f"""[bold]{'✅ Success' if evaluation['success'] else '❌ Failed'}[/]
            Confidence: [cyan]{evaluation['confidence']}%[/]
            Reason: [yellow]{evaluation['reason']}[/]
            """,
            title="Evaluation Results"
        ))
        
        if evaluation.get('suggestions'):
            self.console.print("[bold blue]💡 Suggestions:[/]")
            for suggestion in evaluation['suggestions']:
                self.console.print(f"  • {suggestion}")
                
        if evaluation.get('warnings'):
            self.console.print("[bold red]⚠️ Warnings:[/]")
            for warning in evaluation['warnings']:
                self.console.print(f"  • {warning}")
        
        return evaluation

    async def autonomous_execute(self, main_task: str):
        """Enhanced autonomous execution with detailed progress tracking."""
        from rich.progress import Progress, SpinnerColumn, TimeElapsedColumn
        
        self.console.print(Panel(f"[bold blue]🎯 Main Task:[/] {main_task}"))
        self.task_queue = await self.planner.plan_tasks(main_task, self.client)
        self.execution_stats["total_tasks"] = len(self.task_queue)
        
        with Progress(
            SpinnerColumn(),
            *Progress.get_default_columns(),
            TimeElapsedColumn(),
            console=self.console
        ) as progress:
            overall_task = progress.add_task("[bold blue]Overall Progress", total=len(self.task_queue))
            
            while self.active and self.task_queue:
                available_tasks = [
                    task for task in self.task_queue 
                    if not task.completed and 
                    all(any(t.completed and t.description == dep 
                           for t in self.task_queue) 
                        for dep in task.dependencies)
                ]
                
                if not available_tasks:
                    break
                    
                current_task = sorted(available_tasks, key=lambda t: t.priority)[0]
                
                self.console.print(Panel(
                    f"[bold blue]📌 Current Task:[/] {current_task.description}\n"
                    f"[bold cyan]Priority:[/] {current_task.priority}\n"
                    f"[bold yellow]Dependencies:[/] {', '.join(current_task.dependencies) or 'None'}"
                ))
                
                try:
                    result = await self.execute(current_task.description)
                    evaluation = await self.evaluate_result(current_task.description, result)
                    
                    if evaluation["success"]:
                        current_task.completed = True
                        current_task.result = result
                        self.execution_stats["completed_tasks"] += 1
                        progress.advance(overall_task)
                    else:
                        self.execution_stats["failed_tasks"] += 1
                        if evaluation["confidence"] < 50:  # Retry threshold
                            self.execution_stats["retries"] += 1
                            self.console.print("[bold yellow]🔄 Scheduling task for retry...[/]")
                            continue
                            
                except Exception as e:
                    self.execution_stats["failed_tasks"] += 1
                    self.console.print(f"[bold red]❌ Error:[/] {str(e)}")
                
                self._display_execution_stats()
                
        success = all(task.completed for task in self.task_queue)
        self.console.print(Panel(
            f"[bold {'green' if success else 'red'}]"
            f"{'✅ All tasks completed successfully!' if success else '❌ Some tasks failed!'}"
            f"[/]"
        ))
        return success
