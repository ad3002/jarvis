from typing import List, Dict, Optional
import openai
from openai import AsyncOpenAI
import os
import json
import logging
import datetime
from dataclasses import dataclass, asdict
import hashlib

@dataclass
class HistoryEntry:
    timestamp: str
    task: str
    thoughts: str
    tool: str
    args: Dict
    result: str
    status: str

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
        self.setup_logging()
        
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
        prompt = """Given the following task: {task}
Working directory: {base_dir}
Available tools:
- CREATE_FILE: Creates a new file with content (paths relative to working directory)
- RUN_FILE: Executes a file (paths relative to working directory)
- RUN_LINUX_COMMAND: Executes a Linux command (from working directory)

Provide your response in JSON format with the following structure:
{{
    "thoughts": "your step-by-step reasoning",
    "tool": "tool name to use",
    "args": {{"param1": "value1", "param2": "value2"}}
}}

Your response must be valid JSON.""".format(
            task=task,
            base_dir=self.base_dir
        )
        
        response = await self.client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a helpful AI that always responds with valid JSON"},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
        )
        
        content = response.choices[0].message.content.strip()
        try:
            return json.loads(content)
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
            
            # Save to history
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
            
            logging.info(f"Task completed successfully: {task}")
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
            raise
