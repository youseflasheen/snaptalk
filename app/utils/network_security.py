from __future__ import annotations

import ipaddress
import socket
from urllib.parse import urlparse

import httpx


def parse_allowed_hosts(raw_hosts: str) -> set[str]:
    hosts: set[str] = set()
    for host in raw_hosts.split(","):
        normalized = host.strip().lower()
        if normalized:
            hosts.add(normalized)
    return hosts


def _is_blocked_ip(ip: ipaddress.IPv4Address | ipaddress.IPv6Address) -> bool:
    return (
        ip.is_private
        or ip.is_loopback
        or ip.is_link_local
        or ip.is_multicast
        or ip.is_reserved
        or ip.is_unspecified
    )


def validate_external_url(url: str, allowed_hosts: set[str] | None = None, allow_private: bool = False) -> None:
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"}:
        raise ValueError("Only http and https URLs are allowed")

    hostname = (parsed.hostname or "").strip().lower()
    if not hostname:
        raise ValueError("URL host is required")

    allowed = allowed_hosts or set()
    if hostname in allowed:
        return

    if hostname in {"localhost", "localhost.localdomain"} and not allow_private:
        raise ValueError("Localhost URLs are not allowed")

    try:
        ip = ipaddress.ip_address(hostname)
        resolved_ips = {ip}
    except ValueError:
        try:
            infos = socket.getaddrinfo(hostname, parsed.port or 80, type=socket.SOCK_STREAM)
        except socket.gaierror as exc:
            raise ValueError("URL host could not be resolved") from exc

        resolved_ips = set()
        for info in infos:
            resolved_ips.add(ipaddress.ip_address(info[4][0]))

    if not allow_private:
        for resolved in resolved_ips:
            if _is_blocked_ip(resolved):
                raise ValueError("Private or local network URLs are not allowed")


def download_bytes_with_limit(
    url: str,
    timeout_seconds: float,
    max_bytes: int,
    allowed_hosts: set[str] | None = None,
    allow_private: bool = False,
    accepted_content_prefixes: tuple[str, ...] | None = None,
) -> bytes:
    validate_external_url(url, allowed_hosts=allowed_hosts, allow_private=allow_private)

    with httpx.Client(timeout=timeout_seconds, follow_redirects=True) as client:
        with client.stream("GET", url) as response:
            response.raise_for_status()

            final_url = str(response.url)
            validate_external_url(final_url, allowed_hosts=allowed_hosts, allow_private=allow_private)

            if accepted_content_prefixes:
                content_type = response.headers.get("content-type", "").lower()
                if not any(content_type.startswith(prefix) for prefix in accepted_content_prefixes):
                    raise ValueError("Unsupported remote content type")

            content_length = response.headers.get("content-length")
            if content_length is not None:
                try:
                    remote_size = int(content_length)
                except ValueError:
                    raise ValueError("Invalid content length from remote source")
                if remote_size > max_bytes:
                    raise ValueError("Remote content is too large")

            data = bytearray()
            for chunk in response.iter_bytes(chunk_size=65536):
                data.extend(chunk)
                if len(data) > max_bytes:
                    raise ValueError("Remote content exceeds size limit")

    return bytes(data)
