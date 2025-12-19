### Tmux
```sudo apt install tmux```
tmux remembers which process was ran and then if vscode loses the process, we can recover it using attach -t,
run:
```tmux new -s [my_process_name]``` to launch server process on remote machine (Remote-SSH) that manages shell and processes inside
```run your training...```
then if VSCode or internet cuts out (only client is killed), we can recover seamlessly by reattaching our view to already running session using:
```tmux attach -t [my_process_name]```

to delete specific sessilon:
```tmux kill-session -t [my_process_name]```