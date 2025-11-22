#!/usr/bin/env python3
import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import re
import json
from statistics import median
from collections import Counter, defaultdict

# Optional progress bar
try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False

# ---------- Regex helpers ----------
NUM_RE = re.compile(r"[-+]?(?:\d+\.\d+|\d+)")
ERROR_KWS = re.compile(
    r"\b(error|exception|fatal|panic|stacktrace|refused|failed|failure|crash)\b",
    re.I
)
SECURITY_KWS = re.compile(
    r"\b(unauthori[sz]ed|forbidden|permission denied|invalid cert|certificate|ssl|tls|"
    r"handshake|token expired|csrf|xss|secrets?|private key|keystore|vault|kms)\b", re.I
)
WARNING_KWS = re.compile(
    r"\b(timeout|retry|backoff|degrad|slow|throttl|overload|queue length|high latency)\b",
    re.I
)
INFO_KWS = re.compile(
    r"\b(subscribed|assigned|connected|started|completed|initialized|init|ready|healthy|"
    r"joined|sync(ed)?|refresh(ed)?)\b",
    re.I
)
NOISE_KWS = re.compile(
    r"\b(heartbeat|keepalive|ping|pong|health check|status check|alive|polling|"
    r"routine|periodic|scheduled|maintenance|vacuum|cleanup|gc collection|"
    r"received request|sending response|request completed|debug|trace level|"
    r"info level|verbose|accepted connection|closed connection|idle)\b",
    re.I
)

# Canonical alias tags for semantic duplicates (domain-agnostic)
ALIASES = [
    (
        re.compile(
            r"(timeout.*(db|database)|cannot.*(db|database).*reach|connection.*(db|database).*(timeout|refused))",
            re.I,
        ),
        "DB_CONNECT_TIMEOUT",
    ),
    (
        re.compile(
            r"(connection.*refused|econnrefused|broken pipe|connection reset)",
            re.I,
        ),
        "NET_CONNECT_REFUSED",
    ),
    (
        re.compile(
            r"(host.*unreachable|no route to host|network.*unreachable)",
            re.I,
        ),
        "NET_UNREACHABLE",
    ),
    (
        re.compile(
            r"(authentication failed|auth.*failed|invalid cred|bad credentials)",
            re.I,
        ),
        "AUTH_FAIL",
    ),
]

# Context chains (ordered markers)
CONTEXT_CHAINS = [
    (["offset commit failed", "rebalance", "partition revoked"], "KAFKA_CONSUMER_INSTABILITY"),
    (["connection timeout", "retry", "backoff"], "NET_RETRY_BACKOFF"),
]

# ---- Domain-specific extension points ----
# Domain-specific NOISE patterns (routine/irrelevant logs)
DOMAIN_NOISE_PATTERNS = {
    "apache": [
        re.compile(r"\b200\s+(OK|GET|POST)\b", re.I),  # Successful requests
        re.compile(r"\bGET\s+/.*\.(css|js|png|jpg|gif|ico|woff|ttf)\b", re.I),  # Static assets
        re.compile(r"\b304\s+Not Modified\b", re.I),  # Cache hits
    ],
    "zookeeper": [
        re.compile(r"Notification time out", re.I),
        re.compile(r"Accepted socket connection", re.I),
        re.compile(r"ProcessThread.*Processed session termination", re.I),
    ],
    "hdfs": [
        re.compile(r"Receiving (block|BP-)", re.I),
        re.compile(r"PacketResponder.*type=LAST_IN_PIPELINE", re.I),
        re.compile(r"Verification succeeded", re.I),
    ],
    "spark": [
        re.compile(r"Added broadcast", re.I),
        re.compile(r"Registering block manager", re.I),
        re.compile(r"Started daemon with process name", re.I),
    ],
    "healthapp": [
        re.compile(r"getTodayTotalDetailSteps", re.I),
        re.compile(r"BatteryStats|Battery level", re.I),
        re.compile(r"flush sensor data|sensor background flush", re.I),
    ],

    "openstack": [
        # Routine API polling and successful operations
        re.compile(r"GET /v\d+/.*status:\s*200", re.I),  # Successful GET requests
        re.compile(r"POST /v\d+/.*status:\s*20[012]", re.I),  # Successful POST requests
        re.compile(r"/servers/detail.*status:\s*200", re.I),  # Server list polling
        re.compile(r"metadata\.json.*status:\s*200", re.I),  # Metadata queries
        re.compile(r"vendor_data\.json.*status:\s*200", re.I),  # Vendor data queries
        re.compile(r"VM (Started|Paused|Resumed) \(Lifecycle Event\)", re.I),  # Normal VM lifecycle
        re.compile(r"During sync_power_state the instance has a pending task", re.I),  # Expected behavior
        re.compile(r"image.*checking$", re.I),  # Image cache routine checks
        re.compile(r"in use:.*local.*sharing", re.I),  # Image usage stats
        re.compile(r"Active base files:", re.I),  # Base file listing
        re.compile(r"Auditing locally available compute resources", re.I),  # Resource auditing
        re.compile(r"Total usable (vcpus|memory|disk):", re.I),  # Resource reports
        re.compile(r"Compute_service record updated", re.I),  # Service heartbeat
        re.compile(r"Final resource view:", re.I),  # Resource summary
        re.compile(r"Claim successful", re.I),  # Resource claim success
        re.compile(r"Instance spawned successfully", re.I),  # Success message (debatable - could be INFO)
        re.compile(r"Took \d+\.\d+ seconds to (spawn|build|destroy)", re.I),  # Timing info
    ],

    "linux": [
        # Routine system operations and scheduled tasks
        re.compile(r"session opened for user (cyrus|news|postgres|backup)", re.I),  # Cron job users
        re.compile(r"session closed for user (cyrus|news|postgres|backup)", re.I),  # Cron job cleanup
        re.compile(r"ftpd\[\d+\]: connection from", re.I),  # FTP connections (routine)
        re.compile(r"sshd.*session opened for user.*by \(uid=\d+\)", re.I),  # Successful SSH logins
        re.compile(r"sshd.*session closed for user", re.I),  # SSH logouts
        re.compile(r"CRON\[\d+\].*session (opened|closed)", re.I),  # Cron sessions
        re.compile(r"systemd.*Started|Starting|Reached target", re.I),  # Systemd routine starts
        re.compile(r"kernel:.*audit\(\d+\.\d+:\d+\)", re.I),  # Audit trail (unless contains error)
        re.compile(r"anacron|CROND", re.I),  # Scheduled tasks
        re.compile(r"dhclient.*bound to|DHCPACK", re.I),  # DHCP renewals
        re.compile(r"kernel:.*Link is (Up|Down)", re.I),  # Network link status (routine)
        re.compile(r"su\(pam_unix\).*session (opened|closed) for user (root|cyrus|news)", re.I),  # System user switches
        re.compile(r"postfix.*pickup|cleanup|qmgr", re.I),  # Mail queue processing
        re.compile(r"named\[\d+\].*client.*query:", re.I),  # DNS queries
        re.compile(r"ntpd.*synchronized to", re.I),  # NTP sync (routine)
    ],
}

DOMAIN_ALIASES = {

    # ------------------ APACHE HTTPD ------------------ #
    "apache": [
        (re.compile(r"\b(404|Not\s*Found)\b", re.I), "APACHE_NOT_FOUND"),
        (re.compile(r"\b(500|Internal Server Error)\b", re.I), "APACHE_SERVER_ERROR"),
        (re.compile(r"\bclient denied by server configuration\b", re.I), "APACHE_ACCESS_DENIED"),
        (re.compile(r"\bFile does not exist\b", re.I), "APACHE_MISSING_FILE"),
        (re.compile(r"\bAH\d{4,}\b.*timeout", re.I), "APACHE_HANDLER_TIMEOUT"),
        (re.compile(r"\bmod_ssl\b.*(handshake|tls|ssl)", re.I), "APACHE_SSL_ERROR"),
    ],

    # ------------------ ZOOKEEPER ------------------ #
    "zookeeper": [
        (re.compile(r"Expired session \d+", re.I), "ZK_SESSION_EXPIRED"),
        (re.compile(r"EndOfStreamException", re.I), "ZK_END_OF_STREAM"),
        (re.compile(r"KeeperErrorCode = ConnectionLoss", re.I), "ZK_CONNECTION_LOSS"),
        (re.compile(r"KeeperErrorCode = \w+", re.I), "ZK_OP_ERROR"),
        (re.compile(r"Session \d+ closed", re.I), "ZK_SESSION_CLOSED"),
        (re.compile(r"Exceeded watch limit", re.I), "ZK_WATCH_OVERFLOW"),
    ],

    # ------------------ HDFS / HADOOP ------------------ #
    "hdfs": [
        (re.compile(r"org\.apache\.hadoop\.hdfs\..*IOException", re.I), "HDFS_IO_EXCEPTION"),
        (re.compile(r"Failed to obtain lease", re.I), "HDFS_LEASE_ERROR"),
        (re.compile(r"BlockMissingException|Missing block", re.I), "HDFS_MISSING_BLOCK"),
        (re.compile(r"QuotaExceededException", re.I), "HDFS_QUOTA_EXCEEDED"),
        (re.compile(r"NameNode.*(safe mode|safemode)", re.I), "HDFS_SAFE_MODE"),
        (re.compile(r"Datanode.*shutdown", re.I), "HDFS_DATANODE_SHUTDOWN"),
    ],

    # ------------------ SPARK ------------------ #
    "spark": [
        (re.compile(r"lost executor|executor \d+ exited", re.I), "SPARK_EXECUTOR_LOST"),
        (re.compile(r"job aborted|stage \d+ failed", re.I), "SPARK_STAGE_FAILED"),
        (re.compile(r"TaskSetManager.*failed", re.I), "SPARK_TASKSET_FAILED"),
        (re.compile(r"Container killed by YARN", re.I), "SPARK_YARN_KILLED_CONTAINER"),
    ],

    # ------------------ HEALTH APP (MOBILE SENSOR LOGS) ------------------ #
    "healthapp": [
        (re.compile(r"screen[_ ]?(on|off)", re.I), "HEALTHAPP_SCREEN_STATE"),
        (re.compile(r"flush sensor data", re.I), "HEALTHAPP_SENSOR_FLUSH"),
        (re.compile(r"getTodayTotalDetailSteps", re.I), "HEALTHAPP_STEP_METRICS"),
        (re.compile(r"calculateCaloriesWithCache", re.I), "HEALTHAPP_CALORIE_CALC"),
        (re.compile(r"REPORT\s*:", re.I), "HEALTHAPP_ACTIVITY_REPORT"),
    ],

    # ------------------ OPENSTACK ------------------ #
    "openstack": [
        (re.compile(r"NoValidHost|No valid host", re.I), "OPENSTACK_SCHEDULING_FAILED"),
        (re.compile(r"InstanceNotFound|Instance.*not found", re.I), "OPENSTACK_INSTANCE_NOT_FOUND"),
        (re.compile(r"QuotaError|Quota exceeded", re.I), "OPENSTACK_QUOTA_EXCEEDED"),
        (re.compile(r"PortBindingFailed|Port binding failed", re.I), "OPENSTACK_NETWORK_BINDING_ERROR"),
        (re.compile(r"VolumeAttachFailed|Failed to attach volume", re.I), "OPENSTACK_VOLUME_ATTACH_ERROR"),
        (re.compile(r"ImageNotFound|Image.*not found", re.I), "OPENSTACK_IMAGE_NOT_FOUND"),
        (re.compile(r"Terminating instance|Instance destroyed", re.I), "OPENSTACK_INSTANCE_TERMINATED"),
        (re.compile(r"BuildAbortException|Build.*abort", re.I), "OPENSTACK_BUILD_ABORTED"),
        (re.compile(r"No instances found for any event", re.I), "OPENSTACK_EVENT_NO_INSTANCE"),
        (re.compile(r"Unknown base file", re.I), "OPENSTACK_UNKNOWN_BASE_FILE"),
    ],

    # ------------------ LINUX / SYSLOG ------------------ #
    "linux": [
        (re.compile(r"authentication failure|auth.*fail", re.I), "LINUX_AUTH_FAILURE"),
        (re.compile(r"check pass; user unknown", re.I), "LINUX_UNKNOWN_USER_ATTEMPT"),
        (re.compile(r"ALERT exited abnormally", re.I), "LINUX_PROCESS_ABNORMAL_EXIT"),
        (re.compile(r"kernel:.*Out of memory|OOM", re.I), "LINUX_OOM_KILLER"),
        (re.compile(r"segfault|segmentation fault", re.I), "LINUX_SEGFAULT"),
        (re.compile(r"kernel:.*hung task", re.I), "LINUX_HUNG_TASK"),
        (re.compile(r"disk.*full|No space left", re.I), "LINUX_DISK_FULL"),
        (re.compile(r"Too many open files", re.I), "LINUX_FILE_LIMIT"),
        (re.compile(r"kernel:.*I/O error", re.I), "LINUX_IO_ERROR"),
        (re.compile(r"systemd.*failed|service.*failed", re.I), "LINUX_SERVICE_FAILED"),
    ],
}

# Domain-specific context chains
DOMAIN_CONTEXT_CHAINS = {

    "apache": [
        (["AH", "timeout"], "APACHE_TIMEOUT_CHAIN"),
        (["client denied", "File does not exist"], "APACHE_ACCESS_SEQ"),
    ],

    "zookeeper": [
        (["ConnectionLoss", "Retry"], "ZK_RETRY_CHAIN"),
        (["Session", "Expired"], "ZK_SESSION_EXPIRY_CHAIN"),
        (["EndOfStreamException", "closed"], "ZK_STREAM_CLOSED_CHAIN"),
    ],

    "hdfs": [
        (["safe mode", "retry"], "HDFS_SAFEMODE_RETRY"),
        (["Missing block", "replica"], "HDFS_MISSING_BLOCK_CHAIN"),
        (["lease", "recover"], "HDFS_LEASE_RECOVERY"),
    ],

    "spark": [
        (["lost executor", "resubmitting"], "SPARK_EXECUTOR_FLAP"),
        (["TaskSetManager", "stage failed"], "SPARK_TASK_CHAIN"),
    ],

    "healthapp": [
        (["SCREEN_OFF", "flush sensor data"], "HEALTHAPP_SCREEN_OFF_FLUSH"),
        (["calculateCalories", "REPORT"], "HEALTHAPP_ACTIVITY_CHAIN"),
    ],

    "openstack": [
        (["NoValidHost", "retry", "rescheduled"], "OPENSTACK_SCHEDULING_RETRY"),
        (["PortBindingFailed", "network-vif-plugged"], "OPENSTACK_NETWORK_SETUP_ISSUE"),
        (["Terminating instance", "Instance destroyed"], "OPENSTACK_TEARDOWN_CHAIN"),
        (["BuildAbortException", "deleted"], "OPENSTACK_BUILD_ABORT_CLEANUP"),
        (["QuotaError", "exceeded"], "OPENSTACK_QUOTA_CHAIN"),
        (["Unknown base file", "Removable base files"], "OPENSTACK_IMAGE_CLEANUP"),
    ],

    "linux": [
        (["authentication failure", "check pass", "user unknown"], "LINUX_BRUTE_FORCE_ATTEMPT"),
        (["Out of memory", "OOM killer", "killed process"], "LINUX_OOM_KILL_CHAIN"),
        (["disk full", "No space left"], "LINUX_DISK_FULL_CHAIN"),
        (["segfault", "core dumped"], "LINUX_CRASH_CHAIN"),
        (["service failed", "systemd", "restart"], "LINUX_SERVICE_RESTART_CHAIN"),
        (["kernel panic", "not syncing"], "LINUX_KERNEL_PANIC_CHAIN"),
    ],
}

# Active alias/context/noise lists (base + domain-specific; set in main())
ACTIVE_ALIASES = ALIASES
ACTIVE_CONTEXT_CHAINS = CONTEXT_CHAINS
ACTIVE_NOISE_PATTERNS = []


def build_active_rules(domain: str):
    """
    Build ACTIVE_ALIASES, ACTIVE_CONTEXT_CHAINS, and ACTIVE_NOISE_PATTERNS based on domain.

    - domain in {"all", "generic", "*"} → union of ALL domain rules
    - domain == specific key (e.g. "apache") → only that domain's rules
    """
    domain = (domain or "all").lower()

    # start with global/generic rules
    aliases = list(ALIASES)
    ctx_chains = list(CONTEXT_CHAINS)
    noise_patterns = []

    if domain in ("all", "generic", "*"):
        # union of all domain-specific rules
        for rules in DOMAIN_ALIASES.values():
            aliases.extend(rules)
        for chains in DOMAIN_CONTEXT_CHAINS.values():
            ctx_chains.extend(chains)
        for patterns in DOMAIN_NOISE_PATTERNS.values():
            noise_patterns.extend(patterns)
    else:
        # only that domain's specific rules
        aliases.extend(DOMAIN_ALIASES.get(domain, []))
        ctx_chains.extend(DOMAIN_CONTEXT_CHAINS.get(domain, []))
        noise_patterns.extend(DOMAIN_NOISE_PATTERNS.get(domain, []))

    return aliases, ctx_chains, noise_patterns


# ---------- Data utils ----------
def load_mapping(mapping_csv: Path) -> pd.DataFrame:
    """
    Normalizes your combined templates/logs CSV.

    Expected from drain3_mine.py:
    - line_idx, template_id, template_text, raw, domain_id
    
    Also supports alternative column names:
    - index, templatedid, templatetext, rawline, domainid
    """
    df = pd.read_csv(mapping_csv)

    cols = set(df.columns)

    # Map alternative column names to standard names
    if "line_idx" not in cols and "index" in cols:
        df = df.rename(columns={"index": "line_idx"})
    if "template_id" not in cols and "templatedid" in cols:
        df = df.rename(columns={"templatedid": "template_id"})
    if "template_text" not in cols and "templatetext" in cols:
        df = df.rename(columns={"templatetext": "template_text"})
    if "raw" not in cols and "rawline" in cols:
        df = df.rename(columns={"rawline": "raw"})
    if "domain_id" not in cols and "domainid" in cols:
        df = df.rename(columns={"domainid": "domain_id"})

    required = {"line_idx", "template_id", "template_text", "raw"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"CSV is missing required columns: {missing}\n"
            f"Expected output from drain3_mine.py with columns: "
            f"line_idx, template_id, template_text, raw, domain_id"
        )

    return df.sort_values("line_idx").reset_index(drop=True)


def extract_numbers(text: str):
    return [float(x) for x in NUM_RE.findall(text)]


def majority(iterable):
    c = Counter(iterable)
    if not c:
        return None
    return c.most_common(1)[0][0]


def compute_template_stats(df: pd.DataFrame):
    """
    - freq per template
    - numeric baselines per template (median of all numbers seen in lines of that template)
    """
    tpl_freq = df["template_id"].value_counts().to_dict()
    tpl_num_values = defaultdict(list)
    for _, row in df.iterrows():
        nums = extract_numbers(str(row["raw"]))
        if nums:
            tpl_num_values[row["template_id"]].extend(nums)
    tpl_num_median = {k: (median(v) if v else None) for k, v in tpl_num_values.items()}
    return tpl_freq, tpl_num_median


def canonical_tags_for_text(text: str):
    tags = []
    for rx, tag in ACTIVE_ALIASES:
        if rx.search(text):
            tags.append(tag)
    return tags


def has_context_chain(raw_texts_lower):
    joined = " || ".join(raw_texts_lower)
    for markers, tag in ACTIVE_CONTEXT_CHAINS:
        pos = 0
        ok = True
        for m in markers:
            idx = joined.find(m)
            if idx == -1 or idx < pos:
                ok = False
                break
            pos = idx
        if ok:
            return True, tag
    return False, None


def drift_against_template_baseline(raw_texts, tpl_ids, tpl_medians, drift_factor=3.0):
    """
    If any numeric value in a line exceeds (median * drift_factor) for that template,
    we consider it performance drift.
    """
    reasons = []
    for raw, tid in zip(raw_texts, tpl_ids if tpl_ids is not None else []):
        med = tpl_medians.get(tid)
        if med is None or med <= 0:
            continue
        nums = extract_numbers(str(raw))
        for v in nums:
            if v > med * drift_factor:
                reasons.append(f"value {v} > {drift_factor}x median {med} for {tid}")
                return True, reasons
    return False, reasons


# ---------- Core taxonomy classifier (same as original, but reused with seq_score=0.5) ----------
def classify_sequence(
    seq_score: float,
    seq_tpl_ids,
    seq_raws,
    tpl_freq: dict,
    tpl_medians: dict,
    hi_thresh: float,
    lo_thresh: float,
    heartbeat_freq_threshold: int = 50,
    drift_factor: float = 3.0,
    aggressive_noise: bool = False,
):
    """
    Returns (category, sub_type, confidence, reasons)
    Categories: {Noise, Info, Warning, Error, Security, Unknown}
    SubTypes : {Heartbeat, NormalOperation, RecoverableAnomaly, PerformanceDrift,
                SemanticDuplicate, ContextChain, HiddenAnomaly, FailureEvent,
                PolicyViolation, OOD}
    """
    reasons = []
    raws = [str(x) for x in seq_raws]
    raws_lower = [r.lower() for r in raws]

    # Band by score (window_score or line_score)
    if seq_score >= hi_thresh:
        label_band = "relevant"
    elif seq_score <= lo_thresh:
        label_band = "irrelevant"
    else:
        label_band = "unknown"

    # NEW: only fast-exit on explicit 'irrelevant'
    if label_band == "irrelevant":
        reasons.append("model_irrelevant")
        return ("Noise", "Heartbeat", float(1.0 - seq_score), reasons)

    # From here: 'relevant' or 'unknown' → run rules.

    # Noise patterns (check early to avoid false positives from other rules)
    noise_check_condition = (label_band != "relevant") if not aggressive_noise else (label_band != "relevant" or label_band == "unknown")
    
    if any(NOISE_KWS.search(r) for r in raws) and noise_check_condition:
        reasons.append("noise_keyword_hit")
        return ("Noise", "Heartbeat", float(1.0 - seq_score), reasons)
    
    # Domain-specific noise patterns
    if ACTIVE_NOISE_PATTERNS:
        for r in raws:
            for noise_pattern in ACTIVE_NOISE_PATTERNS:
                if noise_pattern.search(r) and noise_check_condition:
                    reasons.append("domain_noise_pattern_hit")
                    return ("Noise", "Heartbeat", float(1.0 - seq_score), reasons)

    # Security
    if any(SECURITY_KWS.search(r) for r in raws):
        reasons.append("security_keyword_hit")
        return ("Security", "PolicyViolation", float(max(seq_score, hi_thresh)), reasons)

    # Errors
    if any(ERROR_KWS.search(r) for r in raws):
        reasons.append("error_keyword_hit")
        return ("Error", "FailureEvent", float(max(seq_score, hi_thresh)), reasons)

    # Context chain detection
    ctx_hit, ctx_tag = has_context_chain(raws_lower)
    if ctx_hit:
        reasons.append(f"context_chain:{ctx_tag}")
        return ("Warning", "ContextChain", float(max(seq_score, lo_thresh)), reasons)

    # Performance drift detection
    drift_hit, drift_reasons = drift_against_template_baseline(
        raws, seq_tpl_ids, tpl_medians or {}, drift_factor=drift_factor
    )
    if drift_hit:
        reasons.extend(drift_reasons)
        return ("Warning", "PerformanceDrift", float(max(seq_score, lo_thresh)), reasons)

    # Semantic alias / duplicate
    tags = []
    for r in raws:
        tags.extend(canonical_tags_for_text(r))
    if len(set(tags)) >= 1 and len(tags) >= 2:
        reasons.append(f"semantic_aliases:{sorted(set(tags))}")
        cat = "Error" if label_band == "relevant" else "Warning"
        sub = "FailureEvent" if cat == "Error" else "SemanticDuplicate"
        return (cat, sub, float(max(seq_score, lo_thresh)), reasons)

    # Heartbeat / Noise (use template IDs for frequency)
    if tpl_freq and seq_tpl_ids:
        seq_tpl_freqs = [tpl_freq.get(t, 0) for t in seq_tpl_ids]
        median_freq = np.median(seq_tpl_freqs)
        max_freq = max(seq_tpl_freqs) if seq_tpl_freqs else 0
        
        # Catch heartbeat logs by frequency
        if median_freq >= heartbeat_freq_threshold and label_band != "relevant":
            reasons.append(f"heartbeat_freq_median>={heartbeat_freq_threshold}")
            return ("Noise", "Heartbeat", float(1.0 - seq_score), reasons)
        
        # Also catch very high frequency templates (even if not in median)
        if max_freq >= heartbeat_freq_threshold * 2 and label_band == "unknown":
            reasons.append(f"very_high_freq_template>={heartbeat_freq_threshold * 2}")
            return ("Noise", "Heartbeat", float(1.0 - seq_score), reasons)

    # Warnings and info
    if any(WARNING_KWS.search(r) for r in raws):
        reasons.append("warning_keyword_hit")
        return ("Warning", "RecoverableAnomaly", float(max(seq_score, lo_thresh)), reasons)

    # Check for INFO before falling through - more aggressive detection
    has_info = any(INFO_KWS.search(r) for r in raws)
    has_error_warning = any(ERROR_KWS.search(r) or WARNING_KWS.search(r) for r in raws)
    
    if has_info and not has_error_warning and label_band != "relevant":
        reasons.append("info_keyword_hit")
        return ("Info", "NormalOperation", float(1.0 - seq_score), reasons)

    # Fallbacks (if still unknown, use keyword hints)
    if label_band == "unknown":
        if any(ERROR_KWS.search(r) for r in raws):
            reasons.append("error_keyword_hit_fallback")
            return ("Error", "FailureEvent", float(max(seq_score, hi_thresh)), reasons)
        if any(WARNING_KWS.search(r) for r in raws):
            reasons.append("warning_keyword_hit_fallback")
            return ("Warning", "RecoverableAnomaly", float(max(seq_score, lo_thresh)), reasons)
        if any(INFO_KWS.search(r) for r in raws):
            reasons.append("info_keyword_hit_fallback")
            return ("Info", "NormalOperation", float(1.0 - seq_score), reasons)

    # Final fallback by band
    if label_band == "relevant":
        reasons.append("high_score_band")
        return ("Warning", "RecoverableAnomaly", float(seq_score), reasons)

    reasons.append("no_rule_matched")
    return ("Unknown", "OOD", float(seq_score), reasons)


def band_from_category(category: str) -> str:
    """
    Map rule-based category to a coarse band for training:
      - relevant   → Error, Warning, Security
      - irrelevant → Noise, Info
      - unknown    → everything else
    """
    category = (category or "").strip()
    if category in {"Error", "Warning", "Security"}:
        return "relevant"
    if category in {"Noise", "Info"}:
        return "irrelevant"
    return "unknown"


def classify_templates_df(
    mapping_df: pd.DataFrame,
    domain: str = "all",
    hi_thresh: float = 0.75,
    lo_thresh: float = 0.25,
    heartbeat_freq_threshold: int = 50,
    drift_factor: float = 3.0,
    aggressive_noise: bool = False,
    show_progress: bool = True
) -> pd.DataFrame:
    """
    Classify templates using rule-based logic (Python API for notebooks/scripts).
    
    Args:
        mapping_df: DataFrame with columns [line_idx, template_id, template_text, raw, domain_id]
        domain: Domain identifier (all, apache, linux, openstack, etc.)
        hi_thresh: High threshold for confidence scaling (default: 0.75)
        lo_thresh: Low threshold for confidence scaling (default: 0.25)
        heartbeat_freq_threshold: Median template frequency to classify as Heartbeat/Noise (default: 50)
        drift_factor: Multiplier for template median to detect performance drift (default: 3.0)
        aggressive_noise: Be more aggressive in labeling as noise/irrelevant (default: False)
        show_progress: Show progress bar if tqdm is available (default: True)
    
    Returns:
        DataFrame with additional columns: category, sub_type, confidence, reasons, label_band
    
    Example:
        >>> import pandas as pd
        >>> df = pd.read_csv("logs_to_templates_apache.csv")
        >>> classified = classify_templates_df(df, domain="apache")
        >>> print(classified["label_band"].value_counts())
    """
    global ACTIVE_ALIASES, ACTIVE_CONTEXT_CHAINS, ACTIVE_NOISE_PATTERNS
    
    # Validate input
    required = {"line_idx", "template_id", "template_text", "raw"}
    missing = required - set(mapping_df.columns)
    if missing:
        raise ValueError(
            f"DataFrame is missing required columns: {missing}\n"
            f"Expected: line_idx, template_id, template_text, raw, domain_id"
        )
    
    # Remove null templates
    df = mapping_df.dropna(subset=["template_text"]).copy()
    
    # Build domain-specific rules
    domain = (domain or "all").lower()
    ACTIVE_ALIASES, ACTIVE_CONTEXT_CHAINS, ACTIVE_NOISE_PATTERNS = build_active_rules(domain)
    
    # Compute stats
    tpl_freq, tpl_medians = compute_template_stats(df)
    
    # Classify each row
    rows = []
    iterator = tqdm(df.iterrows(), total=len(df), desc="Classifying") if (HAS_TQDM and show_progress) else df.iterrows()
    
    for _, row in iterator:
        seq_score = 0.5  # Neutral score
        
        category, sub_type, confidence, reasons = classify_sequence(
            seq_score=float(seq_score),
            seq_tpl_ids=[row["template_id"]],
            seq_raws=[str(row["raw"])],
            tpl_freq=tpl_freq,
            tpl_medians=tpl_medians,
            hi_thresh=hi_thresh,
            lo_thresh=lo_thresh,
            heartbeat_freq_threshold=heartbeat_freq_threshold,
            drift_factor=drift_factor,
            aggressive_noise=aggressive_noise,
        )
        label_band = band_from_category(category)
        
        rows.append({
            "line_idx": row["line_idx"],
            "template_id": row["template_id"],
            "template_text": str(row["template_text"]),
            "raw": str(row["raw"]),
            "domain_id": row.get("domain_id", domain),
            "category": category,
            "sub_type": sub_type,
            "confidence": round(float(confidence), 6),
            "reasons": "|".join(reasons),
            "label_band": label_band,
        })
    
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser(
        description="Initial rule-only classifier for bootstrap training labels"
    )
    ap.add_argument(
        "--mapping_csv",
        type=Path,
        required=True,
        help="Combined templates/logs CSV (12k rows)",
    )
    ap.add_argument(
        "--output_csv",
        type=Path,
        required=True,
        help="Where to write rule-labeled output CSV",
    )
    ap.add_argument(
        "--domain_id",
        default="all",
        help=(
            "Logical domain for logs. "
            "'all'/'generic' = union of all domain rules; "
            "or choose one: apache, zookeeper, hdfs, spark, healthapp"
        ),
    )
    ap.add_argument(
        "--hi_thresh",
        type=float,
        default=0.75,
        help="High threshold used only for confidence scaling (we pass seq_score=0.5)",
    )
    ap.add_argument(
        "--lo_thresh",
        type=float,
        default=0.25,
        help="Low threshold (we pass seq_score=0.5 so band='unknown')",
    )
    ap.add_argument(
        "--heartbeat_freq_threshold",
        type=int,
        default=50,
        help="Median template frequency to call Heartbeat/Noise",
    )
    ap.add_argument(
        "--drift_factor",
        type=float,
        default=3.0,
        help="x * template median numeric value → PerformanceDrift",
    )
    ap.add_argument(
        "--aggressive_noise",
        action="store_true",
        help="Be more aggressive in labeling logs as noise/irrelevant (increases irrelevant samples)",
    )

    args = ap.parse_args()

    # Validate domain
    valid_domains = ["all", "generic", "*", "apache", "zookeeper", "hdfs", 
                     "spark", "healthapp", "openstack", "linux"]
    domain = (args.domain_id or "all").lower()
    if domain not in valid_domains:
        print(f"[WARNING] Unknown domain '{domain}'. Valid options: {', '.join(valid_domains)}")
        print(f"[WARNING] Continuing with domain='{domain}' (will use generic rules if not found)")

    # Configure domain-specific rules
    global ACTIVE_ALIASES, ACTIVE_CONTEXT_CHAINS, ACTIVE_NOISE_PATTERNS
    ACTIVE_ALIASES, ACTIVE_CONTEXT_CHAINS, ACTIVE_NOISE_PATTERNS = build_active_rules(domain)
    print(
        f"[domain] Using domain_id='{domain}' "
        f"({len(ACTIVE_ALIASES)} alias rules, {len(ACTIVE_CONTEXT_CHAINS)} context chains, "
        f"{len(ACTIVE_NOISE_PATTERNS)} noise patterns)"
    )

    # Load mapping
    print(f"[INFO] Loading mapping CSV from: {args.mapping_csv}")
    df = load_mapping(args.mapping_csv)
    
    # Validate template_text exists and is not empty
    null_templates = df["template_text"].isna().sum()
    if null_templates > 0:
        print(f"[WARNING] {null_templates} rows have null template_text - these will be skipped")
        df = df.dropna(subset=["template_text"])
    
    print(f"[INFO] Loaded {len(df)} rows from mapping CSV")

    # Common stats for rule helpers
    tpl_freq, tpl_medians = compute_template_stats(df)

    rows = []
    iterator = tqdm(df.iterrows(), total=len(df), desc="Classifying logs") if HAS_TQDM else df.iterrows()
    
    for _, row in iterator:
        idx = row["line_idx"]
        tid = row["template_id"]
        template_text = str(row["template_text"])
        raw = str(row["raw"])
        domain_id = row.get("domain_id", domain)

        # Neutral score: 0.5 so label_band inside classify_sequence = 'unknown'
        seq_score = 0.5

        category, sub_type, confidence, reasons = classify_sequence(
            seq_score=float(seq_score),
            seq_tpl_ids=[tid],
            seq_raws=[raw],
            tpl_freq=tpl_freq,
            tpl_medians=tpl_medians,
            hi_thresh=args.hi_thresh,
            lo_thresh=args.lo_thresh,
            heartbeat_freq_threshold=args.heartbeat_freq_threshold,
            drift_factor=args.drift_factor,
            aggressive_noise=args.aggressive_noise,
        )
        label_band = band_from_category(category)

        rows.append({
            "line_idx": idx,
            "template_id": tid,
            "template_text": template_text,
            "raw": raw,
            "domain_id": domain_id,
            "category": category,
            "sub_type": sub_type,
            "confidence": round(float(confidence), 6),
            "reasons": "|".join(reasons),
            "label_band": label_band,  # relevant / irrelevant / unknown
        })

    out_df = pd.DataFrame(rows)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(args.output_csv, index=False)
    
    print(f"\n{'='*60}")
    print(f"[OK] Wrote rule-only labels to {args.output_csv}")
    print(f"{'='*60}")
    print(f"[INFO] Total rows processed: {len(out_df)}")
    print(f"[INFO] Unique templates: {out_df['template_text'].nunique()}")
    print(f"[INFO] Unique domains: {out_df['domain_id'].nunique()}")
    
    print(f"\n{'='*60}")
    print("[LABEL BANDS] Distribution:")
    print(f"{'='*60}")
    band_counts = out_df["label_band"].value_counts(dropna=False)
    for band, count in band_counts.items():
        pct = 100 * count / len(out_df)
        print(f"  {band:12s}: {count:6d} ({pct:5.1f}%)")
    
    print(f"\n{'='*60}")
    print("[CATEGORIES] Distribution:")
    print(f"{'='*60}")
    cat_counts = out_df["category"].value_counts(dropna=False)
    for cat, count in cat_counts.items():
        pct = 100 * count / len(out_df)
        print(f"  {cat:12s}: {count:6d} ({pct:5.1f}%)")
    
    print(f"\n{'='*60}")
    print("[SAMPLE] First 5 classified rows:")
    print(f"{'='*60}")
    sample_cols = ["template_text", "category", "sub_type", "label_band"]
    for idx, row in out_df.head().iterrows():
        print(f"\n{idx+1}. {row['category']:8s} / {row['sub_type']:20s} → {row['label_band']}")
        template_preview = row['template_text'][:80] + "..." if len(row['template_text']) > 80 else row['template_text']
        print(f"   Template: {template_preview}")
    print(f"\n{'='*60}")


if __name__ == "__main__":
    main()
