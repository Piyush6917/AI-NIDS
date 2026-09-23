import os
import time
import subprocess
import threading
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn


BASE = Path(__file__).resolve().parent
LIVE_DIR = BASE / "data" / "live"
LIVE_CSV = LIVE_DIR / "live_flows.csv"
OUT_CSV = LIVE_DIR / "live_predictions.csv"
LAB_PCAP = LIVE_DIR / "lab_capture.pcap"
LAB_CSV = LIVE_DIR / "lab_flows.csv"

SCALER_FILE = BASE / "data" / "final" / "features" / "standard_scaler.joblib"
BINARY_FILE = BASE / "models" / "transformer_binary_gpu.pth"
MULTI_FILE = BASE / "models" / "transformer_multiclass_gpu_v2.pth"

CICFLOWMETER = Path(
    r"C:\Users\piyus\anaconda3\envs\AI-NIDS\Scripts\cicflowmeter.exe"
)

LAB_NETWORK = "ai-nids-lab"
LAB_CLIENT = "nids-client"
LAB_TARGET = "juice-shop"
LAB_TARGET_URL = "http://juice-shop:3000"

BINARY_THRESHOLD = 0.80

CLASSES = [
    "BENIGN",
    "DoS_DDoS",
    "PortScan",
    "BruteForce",
    "WebAttack",
    "Bot",
    "Infiltration",
]

FEATURE_MAP = {
    "Destination Port": "dst_port",
    "Flow Duration": "flow_duration",
    "Total Fwd Packets": "tot_fwd_pkts",
    "Total Backward Packets": "tot_bwd_pkts",
    "Total Length of Fwd Packets": "totlen_fwd_pkts",
    "Total Length of Bwd Packets": "totlen_bwd_pkts",
    "Fwd Packet Length Max": "fwd_pkt_len_max",
    "Fwd Packet Length Min": "fwd_pkt_len_min",
    "Fwd Packet Length Mean": "fwd_pkt_len_mean",
    "Fwd Packet Length Std": "fwd_pkt_len_std",
    "Bwd Packet Length Max": "bwd_pkt_len_max",
    "Bwd Packet Length Min": "bwd_pkt_len_min",
    "Bwd Packet Length Mean": "bwd_pkt_len_mean",
    "Bwd Packet Length Std": "bwd_pkt_len_std",
    "Flow Bytes/s": "flow_byts_s",
    "Flow Packets/s": "flow_pkts_s",
    "Flow IAT Mean": "flow_iat_mean",
    "Flow IAT Std": "flow_iat_std",
    "Flow IAT Max": "flow_iat_max",
    "Flow IAT Min": "flow_iat_min",
    "Fwd IAT Total": "fwd_iat_tot",
    "Fwd IAT Mean": "fwd_iat_mean",
    "Fwd IAT Std": "fwd_iat_std",
    "Fwd IAT Max": "fwd_iat_max",
    "Fwd IAT Min": "fwd_iat_min",
    "Bwd IAT Total": "bwd_iat_tot",
    "Bwd IAT Mean": "bwd_iat_mean",
    "Bwd IAT Std": "bwd_iat_std",
    "Bwd IAT Max": "bwd_iat_max",
    "Bwd IAT Min": "bwd_iat_min",
    "Fwd PSH Flags": "fwd_psh_flags",
    "Fwd Header Length": "fwd_header_len",
    "Bwd Header Length": "bwd_header_len",
    "Fwd Packets/s": "fwd_pkts_s",
    "Bwd Packets/s": "bwd_pkts_s",
    "MinPacket Length": "pkt_len_min",
    "Min Packet Length": "pkt_len_min",
    "Max Packet Length": "pkt_len_max",
    "Packet Length Mean": "pkt_len_mean",
    "Packet Length Std": "pkt_len_std",
    "Packet Length Variance": "pkt_len_var",
    "FIN Flag Count": "fin_flag_cnt",
    "SYN Flag Count": "syn_flag_cnt",
    "RST Flag Count": "rst_flag_cnt",
    "PSH Flag Count": "psh_flag_cnt",
    "ACK Flag Count": "ack_flag_cnt",
    "URG Flag Count": "urg_flag_cnt",
    "ECE Flag Count": "ece_flag_cnt",
    "Down/Up Ratio": "down_up_ratio",
    "Average Packet Size": "pkt_size_avg",
    "Init_Win_bytes_forward": "init_fwd_win_byts",
    "Init_Win_bytes_backward": "init_bwd_win_byts",
    "act_data_pkt_fwd": "fwd_act_data_pkts",
    "min_seg_size_forward": "fwd_seg_size_min",
    "Active Mean": "active_mean",
    "Active Std": "active_std",
    "Active Max": "active_max",
    "Active Min": "active_min",
    "Idle Mean": "idle_mean",
    "Idle Std": "idle_std",
    "Idle Max": "idle_max",
    "Idle Min": "idle_min",
}


class NetworkTransformer(nn.Module):
    def __init__(
        self,
        num_classes,
        num_features=61,
        d_model=64,
        nhead=4,
        num_layers=2,
        dim_feedforward=256,
        dropout=0.1,
    ):
        super().__init__()
        self.feature_embedding = nn.Linear(1, d_model)
        self.feature_position = nn.Parameter(
            torch.randn(1, num_features, d_model)
        )
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(d_model)
        self.classifier = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        x = self.feature_embedding(x.unsqueeze(-1))
        x = x + self.feature_position
        x = self.transformer(x)
        x = self.norm(x.mean(dim=1))
        return self.classifier(x)


def load_model(path, classes, device):
    model = NetworkTransformer(classes)
    ckpt = torch.load(path, map_location=device, weights_only=False)
    state = ckpt.get("model_state_dict", ckpt.get("state_dict", ckpt))
    model.load_state_dict(state, strict=True)
    return model.to(device).eval()


class LiveNIDS:
    """
    AI-NIDS live backend.

    Modes:
      REAL: captures the selected Windows interface with CICFlowMeter.
      LAB: captures packets inside the isolated Docker nids-client
           namespace, then converts the PCAP with CICFlowMeter.

    The AI/scaler/prediction pipeline is identical in both modes.
    """

    def __init__(self):
        self.capture_process = None
        self.capture_mode = "REAL"
        self.capture_interface = ""
        self.lab_capture_process = None
        self.lab_converter_process = None
        self.lab_capture_files = set()
        self.lab_action_process = None
        self.lab_action_name = ""
        self.lab_chunk_counter = 0

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.scaler = None
        self.binary_model = None
        self.multi_model = None

        self.processed_rows = 0
        self.predictions = []
        self.last_error = ""
        self.last_lab_message = ""

        self.total_flows = 0
        self.total_benign = 0
        self.total_attacks = 0
        self.high_severity = 0

    @property
    def is_running(self):
        processes = [self.capture_process, self.lab_capture_process]
        return any(p is not None and p.poll() is None for p in processes)

    def load_models(self):
        if self.scaler is None:
            if not SCALER_FILE.exists():
                raise FileNotFoundError(f"StandardScaler not found: {SCALER_FILE}")
            self.scaler = joblib.load(SCALER_FILE)

        if self.binary_model is None:
            if not BINARY_FILE.exists():
                raise FileNotFoundError(f"Binary model not found: {BINARY_FILE}")
            self.binary_model = load_model(BINARY_FILE, 2, self.device)

        if self.multi_model is None:
            if not MULTI_FILE.exists():
                raise FileNotFoundError(f"Multi-class model not found: {MULTI_FILE}")
            self.multi_model = load_model(MULTI_FILE, 7, self.device)

    def reset_session(self):
        LIVE_DIR.mkdir(parents=True, exist_ok=True)

        for file in (LIVE_CSV, OUT_CSV, LAB_PCAP, LAB_CSV):
            if file.exists():
                try:
                    file.unlink()
                except PermissionError:
                    pass

        self.processed_rows = 0
        self.predictions = []
        self.total_flows = 0
        self.total_benign = 0
        self.total_attacks = 0
        self.high_severity = 0
        self.last_error = ""
        self.last_lab_message = ""

    def _check_lab(self):
        try:
            result = subprocess.run(
                ["docker", "network", "inspect", LAB_NETWORK],
                capture_output=True,
                text=True,
                timeout=10,
            )
            if result.returncode != 0:
                raise RuntimeError(
                    f"Docker network '{LAB_NETWORK}' is not available."
                )

            for name in (LAB_CLIENT, LAB_TARGET):
                result = subprocess.run(
                    ["docker", "inspect", name],
                    capture_output=True,
                    text=True,
                    timeout=10,
                )
                if result.returncode != 0:
                    raise RuntimeError(
                        f"Docker container '{name}' is not available."
                    )

        except FileNotFoundError:
            raise RuntimeError(
                "Docker CLI was not found in PATH. Start Docker Desktop first."
            )

    def _install_tcpdump_if_needed(self):
        # Alpine nids-client does not include tcpdump by default.
        check = subprocess.run(
            ["docker", "exec", LAB_CLIENT, "sh", "-c", "command -v tcpdump"],
            capture_output=True,
            text=True,
            timeout=15,
        )
        if check.returncode == 0:
            return

        self.last_lab_message = "Installing tcpdump in the lab client..."
        install = subprocess.run(
            [
                "docker", "exec", LAB_CLIENT,
                "sh", "-c",
                "apk add --no-cache tcpdump busybox-extras",
            ],
            capture_output=True,
            text=True,
            timeout=120,
        )
        if install.returncode != 0:
            raise RuntimeError(
                "Could not install tcpdump in nids-client.\n"
                + install.stderr[-1500:]
            )

    def start_capture(self, interface):
        """Backward-compatible REAL mode entry point."""
        self.start_real_capture(interface)

    def start_real_capture(self, interface):
        self.last_error = ""
        if self.is_running:
            return

        self.load_models()
        self.reset_session()

        if not CICFLOWMETER.exists():
            raise FileNotFoundError(
                f"CICFlowMeter executable not found: {CICFLOWMETER}"
            )

        self.capture_mode = "REAL"
        self.capture_interface = interface

        command = [
            str(CICFLOWMETER),
            "-i",
            interface,
            "-c",
            str(LIVE_CSV),
        ]

        flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0

        try:
            self.capture_process = subprocess.Popen(
                command,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                creationflags=flags,
                text=True,
            )
        except Exception as exc:
            self.capture_process = None
            raise RuntimeError(f"Could not start CICFlowMeter: {exc}")

        time.sleep(1.5)

        if self.capture_process.poll() is not None:
            error_text = ""
            try:
                error_text = self.capture_process.stderr.read()
            except Exception:
                pass
            self.capture_process = None
            raise RuntimeError(
                "CICFlowMeter stopped immediately.\n"
                + (error_text.strip() or "Check Npcap, adapter name and permissions.")
            )

    def start_lab_capture(self):
        """
        LAB capture strategy:
        tcpdump runs inside nids-client, so Docker Desktop's internal
        bridge does not have to be visible to Windows/Npcap.
        The PCAP is streamed to the host through docker exec.
        """
        self.last_error = ""
        if self.is_running:
            return

        self.load_models()
        self.reset_session()
        self._check_lab()

        if not CICFLOWMETER.exists():
            raise FileNotFoundError(
                f"CICFlowMeter executable not found: {CICFLOWMETER}"
            )

        self._install_tcpdump_if_needed()

        self.capture_mode = "LAB"
        self.capture_interface = "Docker Lab / nids-client"
        self.last_lab_message = "Starting isolated Docker lab capture..."

        # tcpdump writes raw PCAP to stdout; Python saves it on the host.
        # -U makes packet-buffered output, useful for live capture.
        command = [
            "docker", "exec", LAB_CLIENT,
            "tcpdump",
            "-i", "eth0",
            "-U",
            "-s", "0",
            "-w", "-",
        ]

        try:
            stdout_file = open(LAB_PCAP, "wb")
            self.lab_capture_process = subprocess.Popen(
                command,
                stdout=stdout_file,
                stderr=subprocess.PIPE,
            )
            self._lab_stdout_file = stdout_file
        except Exception as exc:
            try:
                stdout_file.close()
            except Exception:
                pass
            self.lab_capture_process = None
            raise RuntimeError(f"Could not start Docker lab capture: {exc}")

        time.sleep(1.5)

        if self.lab_capture_process.poll() is not None:
            error_text = ""
            try:
                error_text = self.lab_capture_process.stderr.read().decode(
                    errors="replace"
                )
            except Exception:
                pass
            self.stop_capture()
            raise RuntimeError(
                "Docker lab packet capture stopped.\n"
                + (error_text.strip() or "Check tcpdump inside nids-client.")
            )

    def _copy_completed_lab_pcaps(self):
        """
        Copy completed 5-second PCAP chunks from the Docker client.
        The newest file is skipped because tcpdump may still be writing it.
        """
        try:
            result = subprocess.run(
                [
                    "docker", "exec", LAB_CLIENT, "sh", "-c",
                    "ls -1t /tmp/ai-nids-lab-*.pcap 2>/dev/null || true",
                ],
                capture_output=True,
                text=True,
                timeout=10,
            )
        except Exception:
            return []

        remote_files = [x.strip() for x in result.stdout.splitlines() if x.strip()]
        if len(remote_files) <= 1:
            return []

        copied = []

        # Skip newest/current chunk.
        for remote in remote_files[1:]:
            if remote in self.lab_capture_files:
                continue

            local = LIVE_DIR / Path(remote).name

            try:
                cp = subprocess.run(
                    ["docker", "cp", f"{LAB_CLIENT}:{remote}", str(local)],
                    capture_output=True,
                    text=True,
                    timeout=20,
                )
                if cp.returncode != 0:
                    continue

                if local.exists() and local.stat().st_size > 100:
                    self.lab_capture_files.add(remote)
                    copied.append(local)
                    subprocess.run(
                        ["docker", "exec", LAB_CLIENT, "rm", "-f", remote],
                        capture_output=True,
                        timeout=10,
                    )
            except Exception:
                continue

        return copied

    def _convert_lab_pcap(self, pcap_path):
        if not pcap_path.exists() or pcap_path.stat().st_size < 100:
            return

        tmp_csv = LIVE_DIR / f"{pcap_path.stem}_flows.csv"

        try:
            if tmp_csv.exists():
                tmp_csv.unlink()
        except PermissionError:
            return

        command = [
            str(CICFLOWMETER),
            "-f",
            str(pcap_path),
            "-c",
            str(tmp_csv),
        ]

        flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0

        try:
            result = subprocess.run(
                command,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                creationflags=flags,
                text=True,
                timeout=60,
            )
        except subprocess.TimeoutExpired:
            self.last_lab_message = f"CICFlowMeter timed out for {pcap_path.name}"
            return
        except Exception as exc:
            self.last_error = f"LAB CICFlowMeter failed: {exc}"
            return

        if result.returncode != 0:
            self.last_lab_message = (
                result.stderr.strip()[-1200:]
                if result.stderr
                else f"CICFlowMeter failed for {pcap_path.name}"
            )
            return

        if not tmp_csv.exists() or tmp_csv.stat().st_size <= 20:
            return

        try:
            flows = pd.read_csv(tmp_csv, on_bad_lines="skip")
        except Exception:
            return

        if flows.empty:
            return

        # Append all completed lab flow chunks to the common live CSV.
        header = not LIVE_CSV.exists()
        flows.to_csv(
            LIVE_CSV,
            mode="a",
            header=header,
            index=False,
        )

        self.last_lab_message = f"Processed {pcap_path.name}"

        try:
            pcap_path.unlink()
            tmp_csv.unlink()
        except Exception:
            pass

    def _process_lab_chunks(self):
        for pcap in self._copy_completed_lab_pcaps():
            self._convert_lab_pcap(pcap)

    def stop_capture(self):
        self.stop_lab_action()

        for attr in ("capture_process", "lab_capture_process", "lab_converter_process"):
            p = getattr(self, attr, None)
            setattr(self, attr, None)

            if p is None:
                continue

            try:
                if p.poll() is None:
                    p.terminate()
                    p.wait(timeout=4)
            except Exception:
                try:
                    p.kill()
                except Exception:
                    pass

        self.lab_capture_files = set()

    def make_features(self, df):
        expected = list(self.scaler.feature_names_in_)
        resolved = []
        missing = []

        for feature in expected:
            live_column = FEATURE_MAP.get(feature)

            if live_column is None:
                missing.append(f"No mapping for: {feature}")
                continue

            if live_column not in df.columns:
                missing.append(f"{feature} -> {live_column}")
                continue

            resolved.append(live_column)

        if missing:
            raise RuntimeError(
                "Missing/unmapped CICFlowMeter columns:\n"
                + "\n".join(missing)
            )

        x = df[resolved].copy()
        x = x.apply(pd.to_numeric, errors="coerce")
        x = x.replace([np.inf, -np.inf], np.nan)
        x = x.fillna(0)

        x = self.scaler.transform(
            x.to_numpy(dtype=np.float64)
        )

        return x.astype(np.float32)

    def predict(self, x):
        if len(x) == 0:
            return []

        tensor = torch.from_numpy(x).to(self.device)

        with torch.inference_mode():
            bp = torch.softmax(self.binary_model(tensor), dim=1)
            mp = torch.softmax(self.multi_model(tensor), dim=1)

        out = []

        for i in range(len(x)):
            attack = float(bp[i, 1])
            idx = int(torch.argmax(mp[i]))
            family = CLASSES[idx]
            conf = float(mp[i, idx])

            if attack >= BINARY_THRESHOLD:
                pred = "ATTACK"

                if family == "BENIGN":
                    for j in torch.argsort(mp[i], descending=True):
                        j = int(j)
                        if CLASSES[j] != "BENIGN":
                            family = CLASSES[j]
                            conf = float(mp[i, j])
                            break
            else:
                pred = "BENIGN"
                family = "BENIGN"

            severity = (
                "NONE"
                if pred == "BENIGN"
                else (
                    "HIGH"
                    if family in ["DoS_DDoS", "Infiltration"]
                    else "MEDIUM"
                )
            )

            out.append((pred, attack, family, conf, severity))

        return out

    def process_new_flows(self):
        if not self.is_running:
            return

        if self.capture_mode == "LAB":
            self._process_lab_chunks()

        source = LIVE_CSV

        if not source.exists():
            return

        try:
            df = pd.read_csv(
                source,
                on_bad_lines="skip",
            )
        except (pd.errors.EmptyDataError, pd.errors.ParserError):
            return

        if len(df) <= self.processed_rows:
            return

        new = df.iloc[self.processed_rows:].copy()

        try:
            results = self.predict(self.make_features(new))
        except Exception as exc:
            self.last_error = repr(exc)
            return

        rows = []

        for raw, result in zip(new.to_dict("records"), results):
            pred, attack, family, conf, severity = result

            item = {
                "timestamp": raw.get("timestamp", ""),
                "src_ip": raw.get("src_ip", ""),
                "src_port": raw.get("src_port", ""),
                "dst_ip": raw.get("dst_ip", ""),
                "dst_port": raw.get("dst_port", ""),
                "protocol": raw.get("protocol", ""),
                "prediction": pred,
                "attack_probability": attack,
                "attack_family": family,
                "family_confidence": conf,
                "severity": severity,
            }

            rows.append(item)
            self.predictions.append(item)
            self.total_flows += 1

            if pred == "ATTACK":
                self.total_attacks += 1
            else:
                self.total_benign += 1

            if severity == "HIGH":
                self.high_severity += 1

        self.processed_rows = len(df)
        self.predictions = self.predictions[-300:]

        if rows:
            pd.DataFrame(rows).to_csv(
                OUT_CSV,
                mode="a",
                header=not OUT_CSV.exists(),
                index=False,
            )

        self.last_error = ""

    def run_lab_action(self, action):
        """
        Start a browser-triggered lab traffic scenario asynchronously.
        All scenarios target only Juice Shop inside ai-nids-lab.
        """
        self._check_lab()

        if hasattr(self, "lab_action_process") and self.lab_action_process is not None:
            if self.lab_action_process.poll() is None:
                raise RuntimeError(
                    f"Lab action '{getattr(self, 'lab_action_name', 'current')}' is still running."
                )

        commands = {
            "normal": (
                "end=$(( $(date +%s) + 20 )); "
                "while [ $(date +%s) -lt $end ]; do "
                f"wget -qO- {LAB_TARGET_URL}/ >/dev/null 2>&1; "
                f"wget -qO- '{LAB_TARGET_URL}/rest/products/search?q=apple' >/dev/null 2>&1; "
                "sleep 2; done"
            ),
            "portscan": (
                "for p in 22 80 443 3000 3001 5000 8000 8080 9000; do "
                "nc -z -w 1 172.18.0.2 $p >/dev/null 2>&1 || true; "
                "done"
            ),
            "webtest": (
                "for q in \"'\" \"<script>test</script>\" \"admin\"; do "
                f"wget -qO- --post-data=\"q=$q\" "
                f"{LAB_TARGET_URL}/rest/products/search >/dev/null 2>&1 || true; "
                "done"
            ),
            "stress": (
                "end=$(( $(date +%s) + 8 )); "
                "while [ $(date +%s) -lt $end ]; do "
                f"wget -qO- {LAB_TARGET_URL}/ >/dev/null 2>&1; "
                "done"
            ),
        }

        if action not in commands:
            raise ValueError(f"Unknown lab action: {action}")

        self.lab_action_name = action
        self.lab_action_process = subprocess.Popen(
            [
                "docker", "exec", LAB_CLIENT,
                "sh", "-c", commands[action],
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )

        return action

    @property
    def lab_action_running(self):
        p = getattr(self, "lab_action_process", None)
        return bool(p is not None and p.poll() is None)

    def stop_lab_action(self):
        p = getattr(self, "lab_action_process", None)
        if p is None:
            return

        try:
            if p.poll() is None:
                p.terminate()
                p.wait(timeout=3)
        except Exception:
            try:
                p.kill()
            except Exception:
                pass

        self.lab_action_process = None
        self.lab_action_name = ""

    # ---------------- LAB traffic controls ----------------

    def _lab_exec(self, shell_command, timeout=30):
        result = subprocess.run(
            ["docker", "exec", LAB_CLIENT, "sh", "-c", shell_command],
            capture_output=True,
            text=True,
            timeout=timeout,
        )

        if result.returncode != 0:
            raise RuntimeError(
                f"Lab command failed:\n{result.stderr[-1500:]}"
            )

        return result.stdout

    def lab_normal_traffic(self, seconds=20):
        self._check_lab()
        # Low-rate legitimate browsing pattern.
        cmd = (
            f"end=$(( $(date +%s) + {int(seconds)} )); "
            f"while [ $(date +%s) -lt $end ]; do "
            f"wget -qO- {LAB_TARGET_URL}/ >/dev/null 2>&1; "
            f"wget -qO- {LAB_TARGET_URL}/rest/products/search?q=apple >/dev/null 2>&1; "
            f"sleep 2; "
            f"done"
        )
        return self._lab_exec(cmd, timeout=int(seconds) + 15)

    def lab_port_scan_test(self):
        self._check_lab()
        # Controlled localhost-lab scan: only the Juice Shop container IP
        # and a small fixed list of ports are tested.
        target = "172.18.0.2"
        ports = "22 80 443 3000 3001 5000 8000 8080 9000"
        cmd = (
            f"for p in {ports}; do "
            f"nc -z -w 1 {target} $p >/dev/null 2>&1 || true; "
            f"done"
        )
        return self._lab_exec(cmd, timeout=20)

    def lab_web_test(self):
        self._check_lab()
        # Harmless security-testing strings sent only to the local OWASP
        # Juice Shop search endpoint. This is not a remote target.
        cmd = (
            f"for q in \"'\" \"<script>test</script>\" \"admin\"; do "
            f"wget -qO- --post-data=\"q=$q\" "
            f"{LAB_TARGET_URL}/rest/products/search >/dev/null 2>&1 || true; "
            f"done"
        )
        return self._lab_exec(cmd, timeout=20)

    def lab_stress_test(self, seconds=8):
        """
        Small, bounded local stress test for demonstration.
        It is deliberately rate-limited and targets only Juice Shop
        inside ai-nids-lab.
        """
        self._check_lab()
        seconds = max(3, min(int(seconds), 10))
        cmd = (
            f"end=$(( $(date +%s) + {seconds} )); "
            f"while [ $(date +%s) -lt $end ]; do "
            f"wget -qO- {LAB_TARGET_URL}/ >/dev/null 2>&1; "
            f"done"
        )
        return self._lab_exec(cmd, timeout=seconds + 15)

    def reset_all(self):
        self.stop_capture()
        self.reset_session()
        self.capture_interface = ""
        self.capture_mode = "REAL"

    def shutdown(self):
        self.stop_capture()
