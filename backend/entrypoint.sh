#!/bin/sh
# Start a headless Xorg server on the NVIDIA GPU so MediaSDK can run its GPU
# blending path. MediaSDK 3.1.5 renders garbage without an X display, so if
# Xorg cannot be started we fall back to CPU image processing.

XORG_DISPLAY="${XORG_DISPLAY:-:99}"
NVIDIA_XORG_DIR=/usr/lib/x86_64-linux-gnu/nvidia/xorg

log() { echo "[entrypoint] $*"; }

use_cpu() {
    log "WARNING: $1; falling back to CPU image processing"
    unset DISPLAY
    export IMAGE_PROCESSING_ACCEL=cpu
}

start_xorg() {
    if [ "${ENABLE_GPU_DISPLAY:-1}" = "0" ]; then
        use_cpu "GPU display disabled (ENABLE_GPU_DISPLAY=0)"
        return
    fi
    if [ ! -f "$NVIDIA_XORG_DIR/nvidia_drv.so" ]; then
        use_cpu "NVIDIA Xorg driver not found in $NVIDIA_XORG_DIR (needs the NVIDIA container runtime)"
        return
    fi

    # nvidia-smi reports e.g. 00000000:03:00.0; Xorg wants decimal PCI:3:0:0.
    bus_id=$(nvidia-smi --query-gpu=pci.bus_id --format=csv,noheader 2>/dev/null | head -n1)
    if [ -z "$bus_id" ]; then
        use_cpu "could not determine GPU PCI bus id"
        return
    fi
    bus=$(printf '%d' "0x$(echo "$bus_id" | cut -d: -f2)")
    dev=$(printf '%d' "0x$(echo "$bus_id" | cut -d: -f3 | cut -d. -f1)")
    fn=$(printf '%d' "0x$(echo "$bus_id" | cut -d. -f2)")

    cat > /etc/X11/xorg.conf <<EOF
Section "Files"
    ModulePath "$NVIDIA_XORG_DIR"
    ModulePath "/usr/lib/xorg/modules"
EndSection
Section "Device"
    Identifier "Device0"
    Driver "nvidia"
    BusID "PCI:$bus:$dev:$fn"
    Option "AllowEmptyInitialConfiguration" "true"
    Option "UseDisplayDevice" "None"
EndSection
Section "Screen"
    Identifier "Screen0"
    Device "Device0"
    DefaultDepth 24
    SubSection "Display"
        Depth 24
        Virtual 1920 1080
    EndSubSection
EndSection
EOF

    display_num="${XORG_DISPLAY#:}"
    rm -f "/tmp/.X${display_num}-lock" "/tmp/.X11-unix/X${display_num}"
    Xorg "$XORG_DISPLAY" -config /etc/X11/xorg.conf -noreset -nolisten tcp \
        -novtswitch -sharevts -keeptty >/var/log/xorg-stdout.log 2>&1 &
    xorg_pid=$!

    i=0
    while [ $i -lt 20 ]; do
        if ! kill -0 "$xorg_pid" 2>/dev/null; then
            use_cpu "Xorg exited during startup (see /var/log/Xorg.${display_num}.log)"
            return
        fi
        if [ -S "/tmp/.X11-unix/X${display_num}" ]; then
            export DISPLAY="$XORG_DISPLAY"
            export IMAGE_PROCESSING_ACCEL="${IMAGE_PROCESSING_ACCEL:-auto}"
            log "Xorg running on $DISPLAY (GPU PCI:$bus:$dev:$fn); GPU image processing enabled"
            return
        fi
        sleep 0.5
        i=$((i + 1))
    done
    kill "$xorg_pid" 2>/dev/null
    use_cpu "Xorg did not become ready within 10s"
}

start_xorg
exec python3 /opt/insta360/run.py "$@"
