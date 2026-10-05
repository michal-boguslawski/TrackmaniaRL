# MLflow UI server. Detached, so it survives tmux detach and logout; output
# goes to logs/mlflow-server.log.
scripts/mlflow_server.sh start
scripts/mlflow_server.sh status
scripts/mlflow_server.sh logs
scripts/mlflow_server.sh stop

# Reach the UI from your laptop, then browse to http://localhost:5001/.
#
# The LOCAL port (left of the colon) is 5001, not 5000, because something on
# this laptop already holds 127.0.0.1:5000 and SSH fails with
# "bind [127.0.0.1]:5000: Permission denied". Note that Windows reports a bind
# conflict as "Permission denied" rather than "Address already in use" when the
# holder bound the port exclusively (SO_EXCLUSIVEADDRUSE), which Docker, WSL and
# several Windows services do. Find the holder with:
#   netstat -ano | findstr :5000
#   Get-Process -Id <pid>          # from the LISTENING row's last column
# The server-side target stays 127.0.0.1:5000 regardless of the local port.
#
# The target is resolved on the server, so 127.0.0.1:5000 is right even though
# the container's own address is the private 172.17.0.3. -N holds the forward
# without a remote shell; ExitOnForwardFailure makes SSH fail loudly rather than
# silently forwarding nothing.
ssh -N -o ServerAliveInterval=30 -o ExitOnForwardFailure=yes \
  -L 5001:127.0.0.1:5000 root@159.48.242.27 -p 48707

# 159.48.242.27 is this host's public address and 48707 the port its provider
# forwards to the container's sshd on 22; neither is reachable from inside the
# container. Watch traffic arrive with: scripts/mlflow_server.sh logs