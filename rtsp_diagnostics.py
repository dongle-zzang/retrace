"""Safe RTSP classifications and URI shapes; no GPU imports or raw error output."""
import re
from urllib.parse import unquote, urlsplit

RTSP_REASONS = frozenset({'rtsp_connection_failed', 'rtsp_auth_failed', 'rtsp_not_found',
                         'rtsp_timeout', 'decoder_error', 'rtsp_error', 'source_caps_error', 'source_link_error'})
DOMAINS = {'gst-resource-error-quark': 'resource', 'gst-stream-error-quark': 'stream',
           'gst-core-error-quark': 'core', 'gst-library-error-quark': 'library',
           'g-io-error-quark': 'io'}
CODECS = frozenset({'H264', 'H265', 'JPEG', 'VP8', 'VP9', 'MP4V-ES'})
FACTORY_NAME = re.compile(r'[a-z0-9_]{1,32}')
DECODERS = frozenset({'nvv4l2decoder', 'avdec_h264', 'avdec_h265', 'avdec_mjpeg',
                      'openh264dec', 'jpegdec'})


def source_failure_detail(reason):
    """Classify only app-owned helper failures, returning fixed safe tokens."""
    if 'not NVIDIA NVMM' in reason:
        return 'source_nvmm_gate', 'source_caps_error'
    if 'no caps' in reason:
        return 'source_pad_caps', 'source_caps_error'
    if 'multiple video' in reason or 'link decoder' in reason:
        return 'source_ghost_link', 'source_link_error'
    return 'source_setup', 'rtsp_error'


def uri_shape(uri):
    parsed = urlsplit(uri)
    parts = [part for part in unquote(parsed.path).split('/') if part]
    return {'scheme': 'rtsp' if parsed.scheme.lower() == 'rtsp' else 'other',
            'path_depth': len(parts), 'profile': next((p for p in ('profile1', 'profile2')
                                                     if p in parts), 'other_or_unspecified'),
            'query_present': bool(parsed.query), 'credentials_present': parsed.username is not None,
            'path_percent_encoded': '%' in parsed.path}


def classify_error(domain, code, message='', debug='', factory='other'):
    """Inspect upstream text in memory; return only fixed codes and integers.

    GStreamer enums are from gsterror.h. OPEN_READ alone is ambiguous; classify
    network/timeout only when recognizable evidence exists, otherwise retain
    rtsp_error. URI text is removed before matching to avoid password/path words
    being mistaken for evidence. No caller should print message/debug.
    """
    safe_domain = DOMAINS.get(domain, 'unknown')
    safe_code = int(code) if isinstance(code, int) and 0 <= code <= 2147483647 else None
    reason = 'rtsp_error'
    text = re.sub(r'(?i)\brtsps?://[^\s<>"\']+', '<uri>', str(message or '') + '\n' + str(debug or ''))
    if safe_domain == 'core' and safe_code in (7, 10):
        reason = 'source_caps_error'
    elif safe_domain == 'core' and safe_code == 5:
        reason = 'source_link_error'
    elif re.search(r'(?i)\bnot-negotiated\b', text):
        reason = 'source_caps_error'
    elif (safe_domain == 'stream' and safe_code in (6, 7)) or factory in DECODERS:
        reason = 'decoder_error'
    elif safe_domain == 'resource' and safe_code == 15:
        reason = 'rtsp_auth_failed'
    elif safe_domain == 'resource' and safe_code == 3:
        reason = 'rtsp_not_found'
    elif re.search(r'(?i)\b(?:unauthorized\s*\(401\)|forbidden\s*\(403\))', text):
        reason = 'rtsp_auth_failed'
    elif re.search(r'(?i)\bnot found\s*\(404\)', text):
        reason = 'rtsp_not_found'
    elif safe_domain == 'core' and safe_code == 12 and re.search(r'(?i)\bmissing decoder\b', text):
        reason = 'decoder_error'
    elif re.search(r'(?i)\b(?:timed out|timeout|time-out)\b', text):
        reason = 'rtsp_timeout'
    elif re.search(r'(?i)\b(?:could not connect|failed to connect|connection refused|no route to host|'
                   r'network is unreachable|could not resolve|name or service not known)\b', text):
        reason = 'rtsp_connection_failed'
    return {'reason': reason, 'domain': safe_domain, 'code': safe_code}


def safe_factory(element):
    """Plugin factory name of a message source as a fixed-charset token, else 'other'."""
    try:
        factory = element.get_factory() if element is not None else None
        name = factory.get_name() if factory is not None else None
        return name if isinstance(name, str) and FACTORY_NAME.fullmatch(name) else 'other'
    except Exception:
        return 'other'


def classify_bus_error(message, warning=False):
    """GI adapter kept duck-typed for CPU regression tests."""
    factory = safe_factory(getattr(message, 'src', None))
    try:
        error, debug = message.parse_warning() if warning else message.parse_error()
        return {**classify_error(error.domain, error.code, error.message, debug, factory), 'factory': factory}
    except Exception:
        return {'reason': 'rtsp_error', 'domain': 'unknown', 'code': None, 'factory': factory}
