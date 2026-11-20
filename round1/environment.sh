#!/usr/bin/env bash
# Individual read failures remain in the log. No power or clock settings are changed.
set -u
probe() {
    printf '\n$ '
    printf '%q ' "$@"
    printf '\n'
    "$@"
    printf '[exit %d]\n' "$?"
}

probe date -Is
probe uname -smr
probe bash -c 'tr -d "\\000" </proc/device-tree/model; echo'
probe bash -c 'tr "\\000" "\\n" </proc/device-tree/compatible'
probe cat /etc/os-release /etc/nv_tegra_release
probe dpkg-query -W nvidia-jetpack nvidia-l4t-core nvidia-l4t-cuda 'cuda-compat-*' 'cuda-toolkit-*'
probe cat /proc/driver/nvidia/version
probe nvidia-smi -q
probe nvpmodel -q
probe bash -c 'for tool in /usr/local/cuda*/bin/{nvcc,tileiras}; do if [[ -x $tool ]]; then echo "$tool"; "$tool" --version; fi; done'
probe gcc --version
probe bash -c 'for candidate in "$HOME/opencodeWorkspace/cutlass" "$HOME/cutlass"; do if test -d "$candidate"; then echo "$candidate"; git -C "$candidate" rev-parse HEAD; git -C "$candidate" status --short; fi; done'
probe bash -c 'for node in /sys/class/devfreq/*; do if [[ -r $node/cur_freq ]]; then echo "node=$node device=$(readlink -f "$node/device")"; for name in name cur_freq min_freq max_freq governor; do if [[ -r $node/$name ]]; then printf "%s=" "$name"; cat "$node/$name"; fi; done; fi; done'
probe bash -c 'for node in /sys/class/thermal/thermal_zone*; do printf "%s " "$node"; paste -d" " "$node/type" "$node/temp"; done'
probe ps -eo pid,comm,pcpu,pmem --sort=-pcpu
probe free -h
probe df -h .
probe timeout 3s tegrastats --interval 1000
