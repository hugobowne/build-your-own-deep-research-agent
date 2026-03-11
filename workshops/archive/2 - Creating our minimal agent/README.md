# 2 - Creating our minimal agent

This workshop will cover the first pass of the main agent runtime.

Rough shape:

- a minimal agent loop
- a simple queue for interrupts and steering
- a way for the user to redirect the agent while it is running

Files:

- `main.py`: minimal async agent loop where `/steer` injects a message after the next tool call and normal input is queued by default
- `agent_tools.py`: local tool runtime with `ReadFile`, `Write`, `Edit`, and `Bash`
