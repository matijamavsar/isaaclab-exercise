#!/bin/bash

# Create a new tmux session and start with a single window
tmux new-session -d

# Split the window into 3 vertical panes (for 6 total panes)
tmux split-window -v
tmux split-window -h
tmux select-pane -t 0
tmux split-window -h
tmux select-pane -t 3
tmux split-window -v
tmux select-pane -t 3
tmux split-window -v

# Send the command to all panes
tmux send -t 0 "PYTHONBREAKPOINT='ipdb.set_trace' sudo python3 docker/container.py enter" C-m
tmux send -t 1 "PYTHONBREAKPOINT='ipdb.set_trace' sudo python3 docker/container.py enter" C-m
tmux send -t 2 "PYTHONBREAKPOINT='ipdb.set_trace' sudo python3 docker/container.py enter" C-m
tmux send -t 3 "PYTHONBREAKPOINT='ipdb.set_trace' sudo python3 docker/container.py enter" C-m
tmux send -t 4 "PYTHONBREAKPOINT='ipdb.set_trace' sudo python3 docker/container.py enter" C-m
tmux send -t 5 "PYTHONBREAKPOINT='ipdb.set_trace' sudo python3 docker/container.py enter" C-m

tmux send -t 1 " htop" C-m
tmux send -t 2 " ranger" C-m
tmux send -t 4 " filebrowser -p 8080 -r logs" C-m
tmux send -t 5 " tensorboard --logdir logs/rsl_rl --port 6006" C-m

# Attach to the tmux session
tmux attach
