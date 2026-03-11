# 2 - migrating to Textual

This workshop is the first pass of moving the agent UI from plain terminal I/O to Textual.

Rough shape:

- a persistent input box at the bottom of the screen
- a transcript area above it
- a reusable shell for wiring in the queue and agent runtime later

Files:

- `shell.py`: reusable Textual shell with a pinned composer, scrollable transcript, inline updatable entries, and named regions
