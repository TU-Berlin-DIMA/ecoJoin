{ cat /proc/stat; sleep 1; cat /proc/stat; } |     awk '/^cpu / {usr=$2-usr; sys=$4-sys; idle=$5-idle; iow=$6-iow}  END {total=usr+sys+idle+iow; printf "%.2f\n", (total-idle)*100/total}'
