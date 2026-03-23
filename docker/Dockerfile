FROM nvidia/cuda:12.6.3-runtime-ubuntu24.04

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl git python3.12 python3.12-venv python3-pip && \
    rm -rf /var/lib/apt/lists/*

# Install modl binary
COPY target/x86_64-unknown-linux-gnu/release/modl /usr/local/bin/modl

# Non-interactive init (no models — user pulls what they need)
RUN modl init --defaults

# Pre-install Python runtime (so first generate doesn't wait 2 min)
RUN modl runtime bootstrap

COPY docker/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

EXPOSE 3939

ENTRYPOINT ["/entrypoint.sh"]
