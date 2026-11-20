"""Read OS-visible objects. Every observation retains its source and unit."""

import ctypes
import ctypes.util
import os
import platform
import re
import stat
import subprocess
from pathlib import Path

SYS = Path("/sys")
CPU = SYS / "devices/system/cpu"
MEDIA = {
    "NVENC": (
        ["libnvidia-encode.so.1", "libnvidia-encode.so"],
        ["NvEncodeAPICreateInstance", "NvEncodeAPIInitialize", "nvEncOpen"],
        ["nvenc", "msenc"],
    ),
    "NVDEC": (
        ["libnvcuvid.so.1", "libnvcuvid.so"],
        ["cuvidCreateVideoParser"],
        ["nvdec"],
    ),
    "NVJPEG": (
        ["libnvjpeg.so.13", "libnvjpeg.so"],
        ["nvjpegCreate", "nvjpegCreateSimple"],
        ["nvjpg", "nvjpeg"],
    ),
    "PVA": (["libcuppva.so", "libcupva_host.so"], ["CupvaGetHardwareInfo"], ["pva"]),
    "OFA": (
        ["libvpi.so", "libnvvpi.so.4"],
        [
            "vpiBackendGetAvailableBackends",
            "vpiAlgorithmCreate",
            "vpiCreateStereoDisparityEstimator",
        ],
        ["ofa"],
    ),
    "ISP": ([], [], ["isp", "video"]),
    "VIC": ([], [], ["vic"]),
}


def reading(path, unit="text", integer=False):
    path = Path(path)
    record = {"source": str(path), "unit": unit, "value": None}
    try:
        value = path.read_text().strip().replace("\x00", " ").strip()
        record.update(status="queried", value=int(value) if integer else value)
        if unit == "Mbit/s" and record["value"] < 0:
            record.update(
                status="unavailable",
                raw=value,
                value=None,
                error="negative speed sentinel",
            )
    except (OSError, ValueError) as error:
        record["status"] = (
            "absent"
            if isinstance(error, FileNotFoundError)
            else "permission_denied"
            if isinstance(error, PermissionError)
            else "unavailable"
        )
        record["error"] = str(error)
    return record


def disk_fields(base, fields):
    return {key: reading(base / key) for key in fields.split()}


def link_name(path):
    try:
        return {
            "status": "queried",
            "source": str(path),
            "value": path.resolve(strict=True).name,
        }
    except OSError as error:
        return {
            "status": "unavailable",
            "source": str(path),
            "value": None,
            "error": str(error),
        }


def directory(path):
    path = Path(path)
    try:
        return {
            "source": str(path),
            "status": "queried",
            "entries": sorted(os.listdir(path)),
        }
    except OSError as error:
        return {
            "source": str(path),
            "status": "permission_denied"
            if isinstance(error, PermissionError)
            else "absent"
            if isinstance(error, FileNotFoundError)
            else "unavailable",
            "entries": None,
            "error": str(error),
        }


def command(argv):
    try:
        p = subprocess.run(argv, capture_output=True, text=True, timeout=12)
        return {
            "argv": argv,
            "exit": p.returncode,
            "stdout": p.stdout,
            "stderr": p.stderr,
            "status": "queried" if p.returncode == 0 else "unavailable",
        }
    except (OSError, subprocess.TimeoutExpired) as error:
        return {
            "argv": argv,
            "exit": None,
            "error": str(error),
            "status": "unavailable",
        }


def cpu():
    per_core = {}
    for d in sorted(CPU.glob("cpu[0-9]*")):
        if not d.name[3:].isdigit():
            continue
        per_core[d.name] = {
            "topology": disk_fields(
                d / "topology",
                "core_id cluster_id physical_package_id core_siblings_list thread_siblings_list",
            ),
            "freq_khz": {
                f: reading(d / "cpufreq" / f, "kHz", True)
                for f in (
                    "scaling_cur_freq",
                    "scaling_min_freq",
                    "scaling_max_freq",
                    "cpuinfo_max_freq",
                )
            },
            "caches": {
                c.name: disk_fields(
                    c, "level type size coherency_line_size shared_cpu_list"
                )
                for c in sorted((d / "cache").glob("index*"))
            },
        }
    return {
        "online": reading(CPU / "online"),
        "present": reading(CPU / "present"),
        "processor_identity_features": reading("/proc/cpuinfo"),
        "lscpu": command(["lscpu", "-J"]),
        "per_core": per_core,
    }


def cuda():
    lib = ctypes.CDLL("libcuda.so.1")
    i = ctypes.c_int

    def required(rc, name):
        if rc:
            raise RuntimeError(f"{name}: CUresult={rc}")

    required(lib.cuInit(0), "cuInit")
    count, version = i(), i()
    required(lib.cuDeviceGetCount(ctypes.byref(count)), "cuDeviceGetCount")
    required(lib.cuDriverGetVersion(ctypes.byref(version)), "cuDriverGetVersion")
    header = (
        Path(os.environ.get("CUDA_HOME", "/usr/local/cuda-13.3")) / "include/cuda.h"
    )
    content = header.read_text()
    enums = {}
    for name, value, doc in re.findall(
        r"^\s*(CU_DEVICE_ATTRIBUTE_\w+)\s*=\s*(\d+)\s*,([^\n]*)", content, re.M
    ):
        enums.setdefault(int(value), []).append((name, doc.strip(" /*<>!")))
    devices = []
    for index in range(count.value):
        d = i()
        required(lib.cuDeviceGet(ctypes.byref(d), index), "cuDeviceGet")
        name, bus = ctypes.create_string_buffer(256), ctypes.create_string_buffer(64)
        mem = ctypes.c_size_t()
        required(lib.cuDeviceGetName(name, len(name), d), "cuDeviceGetName")
        required(lib.cuDeviceGetPCIBusId(bus, len(bus), d), "cuDeviceGetPCIBusId")
        required(lib.cuDeviceTotalMem_v2(ctypes.byref(mem), d), "cuDeviceTotalMem_v2")
        attrs = {}
        for ident, names in sorted(enums.items()):
            result = i()
            code = lib.cuDeviceGetAttribute(ctypes.byref(result), ident, d)
            key, description = names[0]
            attrs[key] = {
                "id": ident,
                "header_description": description,
                "result": code,
                "value": result.value if code == 0 else None,
            }
        devices.append(
            {
                "index": index,
                "name": name.value.decode(),
                "pci_bus_id": bus.value.decode(),
                "total_memory_bytes": mem.value,
                "attributes": attrs,
            }
        )
    return {
        "source": "libcuda.so.1: cuDeviceGet*/cuDeviceTotalMem_v2",
        "enum_source": str(header),
        "driver_api": version.value,
        "device_count": count.value,
        "devices": devices,
    }


def io_inventory():
    pcie = {
        d.name: disk_fields(
            d,
            "vendor device class current_link_speed max_link_speed current_link_width max_link_width",
        )
        for d in sorted((SYS / "bus/pci/devices").glob("*"))
    }
    network = {
        d.name: {
            "state": reading(d / "operstate"),
            "driver": link_name(d / "device/driver"),
            "declared_speed": reading(d / "speed", "Mbit/s", True),
        }
        for d in sorted((SYS / "class/net").glob("*"))
    }
    displays = {
        d.name: disk_fields(d, "status enabled modes")
        for d in sorted((SYS / "class/drm").glob("card*-*"))
    }
    return {
        "pcie": pcie,
        "net_interfaces": network,
        "drm_connectors": displays,
        "drm_source": directory("/sys/class/drm"),
        "net_source": directory("/sys/class/net"),
        "pci_source": directory("/sys/bus/pci/devices"),
    }


def media():
    result = {}
    devnodes = sorted(Path("/dev").iterdir())
    for node in Path("/dev").glob("nvhost*"):
        if node.is_dir():
            devnodes.extend(node.iterdir())
    freqs = sorted((SYS / "class/devfreq").glob("*"))
    ld_cache = command(["ldconfig", "-p"])
    for part, (libs, symbols, patterns) in MEDIA.items():
        observations = []
        candidates = list(libs)
        for line in ld_cache.get("stdout", "").splitlines():
            if any(
                line.lstrip().startswith(lib.split(".so")[0] + ".so") for lib in libs
            ):
                candidate = line.split("=>")[-1].strip()
                if candidate not in candidates:
                    candidates.append(candidate)
        if part == "OFA":
            candidates.extend(
                str(p)
                for p in sorted(Path("/opt/nvidia").glob("vpi*/lib*/**/libnvvpi.so*"))
            )
        for lib in candidates:
            try:
                handle = ctypes.CDLL(lib)
                observations.append(
                    {
                        "library": lib,
                        "loaded": True,
                        "symbols": {
                            symbol: hasattr(handle, symbol) for symbol in symbols
                        },
                    }
                )
                if all(observations[-1]["symbols"].values()):
                    break
            except OSError as error:
                observations.append(
                    {"library": lib, "loaded": False, "error": str(error)}
                )
        clock = {
            d.name: reading(d / "cur_freq", "Hz", True)
            for d in freqs
            if any(x in d.name for x in patterns)
        }
        clock["tegra_profiler"] = reading(
            SYS / "kernel/debug/tegra_profiler" / f"{part.lower()}_clk_rate", "Hz", True
        )
        for name in {
            "NVENC": ["nvenc", "nvenc0", "nvenc1"],
            "NVDEC": ["nvdec", "nvdec0", "nvdec1"],
            "NVJPEG": ["nvjpg", "nvjpg1"],
            "PVA": ["pva", "pva0_vps0"],
            "OFA": ["ofa"],
            "ISP": ["isp"],
            "VIC": ["VIC", "vic"],
        }[part]:
            clock["bpmp_" + name] = reading(
                SYS / "kernel/debug/bpmp/debug/clk" / name / "rate", "Hz", True
            )
        result[part] = {
            "libraries": observations,
            "nodes": [str(d) for d in devnodes if any(x in d.name for x in patterns)],
            "node_source": "/dev/ (incl. nvhost directories)",
            "clocks": clock,
            "v4l2_names": {
                p.parent.name: reading(p)
                for p in (SYS / "class/video4linux").glob("*/name")
            }
            if part == "ISP"
            else {},
        }
        result[part]["character_devices"] = [
            str(d)
            for d in devnodes
            if str(d) in result[part]["nodes"]
            and d.exists()
            and stat.S_ISCHR(d.stat().st_mode)
        ]
        if part == "ISP":
            result[part]["v4l2_source"] = directory("/sys/class/video4linux")
    return result


def memory_devices():
    # Read SMBIOS Type 17 only. Do not record physical module serials/addresses.
    table = SYS / "firmware/dmi/tables/DMI"
    records = []
    try:
        blob = table.read_bytes()
        offset = 0
        while offset + 4 <= len(blob):
            typ, length = blob[offset : offset + 2]
            end = blob.find(b"\x00\x00", offset + length)
            if length < 4 or end < 0:
                raise ValueError("invalid SMBIOS record")
            if typ == 17 and length >= 0x13:
                total = int.from_bytes(blob[offset + 8 : offset + 10], "little")
                width = int.from_bytes(blob[offset + 10 : offset + 12], "little")
                records.append(
                    {
                        "total_width_bits": None if total == 65535 else total,
                        "data_width_bits": None if width == 65535 else width,
                        "type_code": blob[offset + 0x12],
                    }
                )
            offset = end + 2
            if typ == 127:
                break
        dmi = {"source": str(table), "status": "queried", "type17": records}
    except (OSError, ValueError) as error:
        dmi = {
            "source": str(table),
            "status": "permission_denied"
            if isinstance(error, PermissionError)
            else "unavailable",
            "value": None,
            "error": str(error),
        }
    return {
        "meminfo": reading("/proc/meminfo"),
        "smbios": dmi,
        "edac_types": {
            str(p): reading(p)
            for p in (SYS / "devices/system/edac/mc").glob("mc*/dimm*/dimm_mem_type")
        },
        "theoretical_bandwidth": {
            "value": None,
            "unit": "GB/s",
            "status": "unconfirmed_sku",
        },
    }


def collect():
    return {
        "platform": {
            "kernel": platform.uname()._asdict(),
            "board": reading("/proc/device-tree/model"),
            "device_tree_compatible": reading("/proc/device-tree/compatible"),
            "sku": reading("/sys/class/dmi/id/product_sku"),
            "l4t": reading("/etc/nv_tegra_release"),
            "power_mode": command(["nvpmodel", "-q"]),
        },
        "cuda": cuda(),
        "cpu": cpu(),
        "memory": memory_devices(),
        "devfreq_policy": {
            d.name: disk_fields(
                d, "name governor min_freq max_freq available_frequencies"
            )
            for d in (SYS / "class/devfreq").glob("*")
        },
        "io": io_inventory(),
        "media": media(),
    }
