#!/usr/bin/env bash
# freetoken standalone Test — läuft im Repo-Root (dort, wo pyproject.toml liegt).
# Erzeugt eine frische venv, installiert den Fork, prüft Versionen/Imports/Unit-Tests
# und macht einen echten Engine-Smoke-Test (serve + chat) mit einem lokalen Checkpoint.
#
#   ./scripts/run-tests.sh                      # nutzt Standard-Modellpfad (siehe unten)
#   MODEL_PATH=/pfad/zum/checkpoint ./scripts/run-tests.sh
#
# Der vollständige Report landet in ft-test-report.txt im Repo-Root.

set -u
# Egal ob aus scripts/ oder dem Root gestartet: ins Repo-Root wechseln.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if ROOT="$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel 2>/dev/null)"; then
    cd "$ROOT"
else
    cd "$SCRIPT_DIR/.."
fi

REPORT="ft-test-report.txt"
PORT="${PORT:-1929}"
MODEL_PATH="${MODEL_PATH:-/path/to/model}"
VENV=".ft-test-venv"
PY="$VENV/bin/python"
PIP="$VENV/bin/pip"
LOAD_TIMEOUT="${LOAD_TIMEOUT:-720}"   # Sekunden bis der Chat-Test aufgibt
PASS=0; FAIL=0

say()  { printf '%s\n' "$*" | tee -a "$REPORT"; }
pass() { PASS=$((PASS+1)); say "PASS  $*"; }
fail() { FAIL=$((FAIL+1)); say "FAIL  $*"; }
hdr()  { say ""; say "==================================================="; say "== $*"; say "==================================================="; }

: > "$REPORT"
say "freetoken standalone Test — $(date -u '+%Y-%m-%d %H:%M:%S UTC')"
say "Repo: $(pwd)"

# =============================================================
hdr "0. Umgebung"
# =============================================================
say "Kernel:   $(uname -srm)"
say "OS:       $(grep -E '^(NAME|VERSION)=' /etc/os-release 2>/dev/null | tr '\n' ' ')"
say "Python:   $(python3 --version 2>&1)  ($(command -v python3 || echo 'NICHT GEFUNDEN'))"
say "Git:      $(git rev-parse --short HEAD 2>/dev/null || echo 'kein git-Repo')  Branch: $(git branch --show-current 2>/dev/null || echo '-')"
NVML_BROKEN=0
if command -v nvidia-smi >/dev/null 2>&1; then
    # NVML kann bei Treiber-/Library-Mismatch (>Update ohne Reboot) Freitext statt
    # GPU-Zeilen liefern — das darf spätere numerische Checks nicht vergiften.
    GPU_OUT=$(nvidia-smi --query-gpu=index,name,driver_version,memory.used,memory.total --format=csv,noheader 2>&1)
    if printf '%s' "$GPU_OUT" | grep -qiE "nvml|version mismatch|driver/library|no devices"; then
        NVML_BROKEN=1; GPU_COUNT=0
        say "GPU:      $GPU_OUT"
        fail "NVML nicht benutzbar (Treiber-/Library-Mismatch?) — GPU-Tests/Engine werden übersprungen; nach Treiber-Reload/Reboot erneut versuchen."
    else
        printf '%s\n' "$GPU_OUT" | while read -r l; do say "GPU:      $l"; done
        GPU_COUNT=$(printf '%s\n' "$GPU_OUT" | grep -c '^[0-9]')
    fi
else
    GPU_COUNT=0; fail "nvidia-smi nicht gefunden — keine CUDA-Umgebung?"
fi
say "GPUs:     $GPU_COUNT"
if command -v nvcc >/dev/null 2>&1; then
    say "nvcc:     $(nvcc --version 2>&1 | grep -o 'release [0-9.]*')"
else
    say "nvcc:     NICHT VORHANDEN (nur relevant, wenn freetoken-kernel-cache fehlt: erste Kernel-Nutzung kompiliert dann selbst)"
fi
say "Modell:   $MODEL_PATH"
if [ -d "$MODEL_PATH" ]; then
    say "Modellgröße: $(du -sh --apparent-size "$MODEL_PATH" 2>/dev/null | cut -f1), config.json: $([ -f "$MODEL_PATH/config.json" ] && echo ja || echo FEHLT)"
else
    fail "Modellpfad existiert nicht — bitte MODEL_PATH=/pfad/zum/checkpoint setzen"
fi

# =============================================================
hdr "1. Frische venv"
# =============================================================
if [ "${REUSE:-0}" = "1" ] && [ -x "$PY" ]; then
    pass "venv wiederverwendet (REUSE=1): $VENV"
else
rm -rf "$VENV"
if python3 -m venv "$VENV" 2>>"$REPORT"; then
    "$PIP" install --quiet --upgrade pip 2>>"$REPORT" && pass "venv erstellt ($($PY --version 2>&1))" || fail "pip-upgrade in venv fehlgeschlagen"
else
    fail "venv-Erstellung fehlgeschlagen (python3-venv installiert?)"; say ""; say "Abbruch — Report: $REPORT"; exit 1
fi
fi

# =============================================================
hdr "2. Dependency-Resolution (trocken, ohne Download)"
# =============================================================
# Zieht die Kern-Abhängigkeiten aus pyproject.toml und prüft, ob pip sie auflösen
# kann — ohne etwas zu installieren. Fehler hier = Resolver-/Plattform-Problem.
DEPS=$("$PY" - <<'EOF'
import re, sys
try:
    t = open("pyproject.toml").read()
except FileNotFoundError:
    sys.exit(1)
block = t.split("dependencies = [", 1)[1].split("]", 1)[0]
for line in block.splitlines():
    line = line.split("#", 1)[0].split(";", 1)[0].strip().strip('",').strip()
    if line:
        print(line)
EOF
)
if [ -z "$DEPS" ]; then
    fail "konnte keine Dependencies aus pyproject.toml lesen — bin du im Repo-Root?"
else
    if $PIP install --dry-run $DEPS >"$REPORT.tmp-resolve" 2>&1; then
        pass "alle Kern-Dependencies auflösbar ($(echo "$DEPS" | wc -l) Pakete)"
    else
        fail "RESOLUTION FEHLGESCHLAGEN — Details:"
        tail -25 "$REPORT.tmp-resolve" | while read -r l; do say "      $l"; done
    fi
    rm -f "$REPORT.tmp-resolve"
fi

# =============================================================
hdr "3. Installation (freetoken aus diesem Repo)"
# =============================================================
# NB: Build-Isolation lädt für den Build-Backend ein eigenes torch (~2-3 GB).
# Bei langsamem Netz hier Timeout => Punkt 3 ist der Verdächtige.
if [ "${REUSE:-0}" = "1" ] && "$PY" -c "import freetoken" 2>/dev/null; then
    pass "Installation übersprungen (REUSE=1, freetoken bereits installiert)"
else
INSTALL_START=$(date +%s)
if $PIP install -v . >"$REPORT.tmp-install" 2>&1; then
    pass "pip install . erfolgreich ($(date +%s -d @$(( $(date +%s) - INSTALL_START )) -u +%Mm%Ss) Dauer, $(du -sh "$VENV" | cut -f1))"
else
    fail "pip install . FEHLGESCHLAGEN — letzte 30 Zeilen:"
    tail -30 "$REPORT.tmp-install" | while read -r l; do say "      $l"; done
    say ""; say "Abbruch — Report: $REPORT"; exit 1
fi
rm -f "$REPORT.tmp-install"
fi

# =============================================================
hdr "4. Versions-Audit (installierte Versionen vs. pyproject-Bereiche)"
# =============================================================
"$PY" - <<'EOF'
import importlib.metadata as md
import re, sys

try:
    t = open("pyproject.toml").read()
    block = t.split("dependencies = [", 1)[1].split("]", 1)[0]
except FileNotFoundError:
    sys.exit(1)
ranges = {}
for line in block.splitlines():
    line = line.split("#", 1)[0].strip().strip('",').strip()
    if not line:
        continue
    m = re.match(r'([A-Za-z0-9_.-]+)\s*(.*)', line)
    if m:
        ranges[m.group(1).lower().replace("_", "-")] = m.group(2).strip().strip('"')

KEY = ["torch", "triton", "flashlib", "apache-tvm-ffi", "safetensors", "transformers",
       "numpy", "huggingface_hub", "gguf", "einops", "pydantic", "uvicorn", "fastapi"]
bad = 0
for k in KEY:
    rng = ranges.get(k, "?")
    try:
        v = md.version(k)
    except md.PackageNotFoundError:
        print(f"  FEHLT   {k:22s} (benötigt: {rng})"); bad += 1; continue
    # grobe Bereichsprüfung über pip (nutzt denselben Spec-Parser)
    import subprocess
    ok = subprocess.run([sys.executable, "-m", "pip", "install", "--dry-run", "--quiet",
                         f"{k}{rng}" if rng not in ("", "?") else k],
                        capture_output=True).returncode == 0
    status = "ok " if ok else "DRIN ABER AUSSERHALB DES BEREICHS"
    if not ok:
        bad += 1
    print(f"  {status:34s} {k:22s} {v:14s} (benötigt: {rng})")
sys.exit(1 if bad else 0)
EOF
if [ $? -eq 0 ]; then pass "Versions-Audit: alle Schlüssel-Pakete vorhanden und im Bereich"; else fail "Versions-Audit: Abweichungen (siehe oben)"; fi

# =============================================================
hdr "5. Import-Proben (Runtime-Module)"
# =============================================================
"$PY" - <<'EOF'
import importlib, sys
fails = 0
for mod in ["freetoken", "freetoken.version", "safetensors", "flashlib.kernels.slot_cache",
            "tvm_ffi", "gguf", "transformers", "torch"]:
    try:
        m = importlib.import_module(mod)
        v = getattr(m, "__version__", "-")
        print(f"  ok      {mod:32s} {v}")
    except Exception as e:
        print(f"  FEHLER  {mod:32s} {type(e).__name__}: {e}")
        fails += 1
try:
    import freetoken_kernel_cache
    print("  ok      freetoken_kernel_cache (vorgebaute Kernel vorhanden)")
except Exception as e:
    print("  HINWEIS freetoken-kernel-cache nicht installiert (nicht auf PyPI) —"
          " erste Kernel-Nutzung kompiliert mit nvcc selbst")
print("freetoken", end=" ")
import freetoken.version as v
print(v.__version__)
sys.exit(1 if fails else 0)
EOF
if [ $? -eq 0 ]; then pass "alle Kern-Module importierbar"; else fail "Import-Fehler (siehe oben)"; fi

# =============================================================
hdr "6. Bugfix-Unit-Tests (pytest, CPU-only)"
# =============================================================
$PIP install --quiet pytest > /dev/null 2>&1
if "$PY" -m pytest tests/checkpoint/test_extract_ple_table.py \
                   tests/checkpoint/test_ftw_side_tables.py \
                   tests/models/test_modelopt_nested_config.py \
                   -m "not slow and not needs_weights" -q >>"$REPORT" 2>&1; then
    pass "Bugfix-Regressionen grün (PLE-FP8-Extraktion, ModelOpt nested config, FTW side tables)"
else
    fail "Unit-Tests fehlgeschlagen (Details im Report oben)"
fi

# =============================================================
hdr "7. Engine-Smoke-Test (ft serve + chat)"
# =============================================================
if [ "$GPU_COUNT" -eq 0 ]; then
    if [ "${NVML_BROKEN:-0}" = "1" ]; then
        fail "Engine-Test übersprungen — NVML/Treiber-Mismatch (siehe Abschnitt 0): erst Treiber reladen/Reboot."
    else
        fail "keine GPU — Engine-Test übersprungen"
    fi
elif [ ! -d "$MODEL_PATH" ]; then
    fail "Modellpfad fehlt — Engine-Test übersprungen (MODEL_PATH setzen)"
else
    # FREE_MIN kann bei NVML-Problemen Freitext sein — erst numerisch prüfen,
    # sonst bricht `[ ... -lt ... ]` mit "Ganzzahliger Ausdruck erwartet" ab.
    FREE_MIN=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | sort -n | head -1)
    case "$FREE_MIN" in
        ''|*[!0-9]*)
            say "HINWEIS: VRAM-Probe nicht verfügbar (nvidia-smi: '${FREE_MIN:-leer}') — Budget-Check übersprungen." ;;
        *)
            if [ "$FREE_MIN" -lt 20000 ]; then
                say "HINWEIS: wenig freier VRAM (${FREE_MIN} MiB) — andere GPU-Prozesse stoppen, sonst Budget-Crash möglich:"
                nvidia-smi --query-compute-apps=pid,name,used_memory --format=csv,noheader 2>/dev/null | while read -r l; do say "         belegt: $l"; done
            fi ;;
    esac
    # Manche Umgebungen (Container/Headless) können die SM-Version nicht über NVML
    # ableiten; tvm_ffi bricht dann beim Kernel-JIT ab ("Could not detect CUDA
    # compute_cap"). Wenn NVML die compute_cap kennt, dem JIT das Ziel vorgeben.
    if [ -z "${TVM_FFI_CUDA_ARCH_LIST:-}" ]; then
        ARCHS=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | sort -u | paste -sd' ' -)
        case "$ARCHS" in
            ''|*[!0-9.\ ]*) : ;;
            *) export TVM_FFI_CUDA_ARCH_LIST="$ARCHS"; say "TVM_FFI_CUDA_ARCH_LIST=$ARCHS (Kernel-JIT-Ziel)" ;;
        esac
    fi
    SERVE_LOG="ft-serve.log"
    TP="$GPU_COUNT"
    say "Starte: ft serve --model-path $MODEL_PATH --tensor-parallel-size $TP --port $PORT (Log: $SERVE_LOG)"
    rm -f "$SERVE_LOG"
    "$VENV/bin/ft" serve --model-path "$MODEL_PATH" \
        --served-model-name qwen38-standalone-test \
        --moe-backend offload --memory-ratio 0.9 --kv-reserve-tokens 200000 \
        --max-output-tokens 32768 --max-running-requests 4 \
        --tensor-parallel-size "$TP" --ple-backend pinned \
        --host 127.0.0.1 --port "$PORT" >>"$SERVE_LOG" 2>&1 &
    FT_PID=$!
    SAY_READY=""
    START=$(date +%s)
    while [ $(( $(date +%s) - START )) -lt "$LOAD_TIMEOUT" ]; do
        sleep 10
        if ! kill -0 "$FT_PID" 2>/dev/null; then
            fail "Engine-Prozess EXITIERT VORZEITIG (nach $(( $(date +%s) - START ))s) — letzte 25 Logzeilen:"
            tail -25 "$SERVE_LOG" | while read -r l; do say "      $l"; done
            break
        fi
        RESP=$("$PY" -c "
import json, urllib.request
try:
    r = urllib.request.urlopen('http://127.0.0.1:$PORT/v1/models', timeout=3)
    print(json.load(r)['data'][0]['id'])
except Exception:
    pass" 2>/dev/null)
        if [ -n "$RESP" ]; then
            pass "/v1/models antwortet: $RESP (nach $(( $(date +%s) - START ))s)"
            SAY_READY=1
            break
        fi
        say "      ... lädt ($(( $(date +%s) - START ))s): $(tail -1 "$SERVE_LOG" | cut -c1-110)"
    done
    if [ -n "$SAY_READY" ]; then
        # Chat-Test mit Retry (Model kann nach /v1/models noch PLE-Table + MoE-Cache laden — mehrere Minuten)
        CHAT=""
        for i in $(seq 1 60); do
            CHAT=$("$PY" -c "
import json, urllib.request
body = json.dumps({'model':'qwen38-standalone-test',
                   'messages':[{'role':'user','content':'Antworte mit genau: TEST OK'}],
                   'max_tokens':200}).encode()
req = urllib.request.Request('http://127.0.0.1:$PORT/v1/chat/completions', data=body,
                             headers={'Content-Type':'application/json'})
try:
    r = json.load(urllib.request.urlopen(req, timeout=120))
    c = r['choices'][0]['message']
    print('CHAT-OK |', (c.get('content') or '')[:80].replace(chr(10),' '), '| usage:', r['usage']['total_tokens'])
except Exception as e:
    print('CHAT-ERR', e)
" 2>/dev/null)
            case "$CHAT" in CHAT-OK*) break;; esac
            if ! kill -0 "$FT_PID" 2>/dev/null; then
                fail "Engine-Prozess während des Chat-Tests abgestürzt — letzte 25 Logzeilen:"
                tail -25 "$SERVE_LOG" | while read -r l; do say "      $l"; done
                CHAT="ENGINE-DEAD"; break
            fi
            say "      ... Chat-Versuch $i: $CHAT"
            sleep 10
        done
        case "$CHAT" in
            CHAT-OK*) pass "Chat-Completion: $CHAT";;
            ENGINE-DEAD) : ;;  # Absturz oben bereits diagnostiziert
            *)        fail "Chat-Completion fehlgeschlagen — letzte 20 Logzeilen:"
                      tail -20 "$SERVE_LOG" | while read -r l; do say "      $l"; done;;
        esac
    elif kill -0 "$FT_PID" 2>/dev/null; then
        fail "Engine nach ${LOAD_TIMEOUT}s nicht bereit — HÄNGT vermutlich. Letzte 30 Logzeilen:"
        tail -30 "$SERVE_LOG" | while read -r l; do say "      $l"; done
    fi
    kill "$FT_PID" 2>/dev/null && say "Engine gestoppt."
fi

# =============================================================
hdr "Zusammenfassung"
# =============================================================
say "PASS: $PASS   FAIL: $FAIL"
say "Vollständiger Report: $(pwd)/$REPORT"
say "Engine-Log (falls Punkt 7 lief): $(pwd)/ft-serve.log"
[ "$FAIL" -eq 0 ] && say "=> ALLES GRÜN." || say "=> Es gibt $FAIL Fehler — Report ansehen."
exit 0
