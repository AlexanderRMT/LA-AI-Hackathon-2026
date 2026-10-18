#!/bin/sh
# run_mirage.sh: start Mirage on the Pi. Run as root from the folder with the binaries.   owner: Alex
# usage: ./run_mirage.sh <dashboard_ip> ./mirage_vision [vision args...]
#
# The supervisor sets its own priority (50). The vision process runs at 20 and is restarted
# if it ever dies; the supervisor fails closed in the meantime and stays closed until reset.
DASH="$1"
shift
./supervisor $DASH &
sleep 1
while true; do
    on -p 20 "$@"
    echo "[run_mirage] vision exited with $?, restarting in 1 s"
    sleep 1
done
