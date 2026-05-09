FROM nvidia/cuda:12.9.1-runtime-ubuntu24.04

ARG DOCKER_LIBRARY_SHA=b2bffb1e479e614f1a65ffdf8e9120200b76d08b
ARG MOBY_SHA=0498c5457790c544a9c3768772dc36f308662260

RUN apt-get update && \
    apt-get install --no-install-recommends -y \
        apt-utils \
        ca-certificates \
        openssh-client \
        curl \
        iptables \
        gnupg

RUN curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg && \
    curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
        | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
        | tee /etc/apt/sources.list.d/nvidia-container-toolkit.list && \
    apt-get update && \
    apt-get install --no-install-recommends -y nvidia-container-toolkit docker.io && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

ADD https://raw.githubusercontent.com/docker-library/docker/${DOCKER_LIBRARY_SHA}/modprobe.sh /usr/local/bin/modprobe
ADD https://raw.githubusercontent.com/docker-library/docker/${DOCKER_LIBRARY_SHA}/dockerd-entrypoint.sh /usr/local/bin/
ADD https://raw.githubusercontent.com/docker-library/docker/${DOCKER_LIBRARY_SHA}/docker-entrypoint.sh /usr/local/bin/
ADD https://raw.githubusercontent.com/moby/moby/${MOBY_SHA}/hack/dind /usr/local/bin/dind

RUN chmod +x /usr/local/bin/dockerd-entrypoint.sh /usr/local/bin/docker-entrypoint.sh /usr/local/bin/dind

RUN groupadd -g 42420 app && \
    useradd -u 42420 -g app -m -s /bin/bash app && \
    mkdir -p /var/lib/docker && \
    chown -R app:app /var/lib/docker

USER app

VOLUME /var/lib/docker
EXPOSE 2375 2376

HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:2375 || exit 1

ENTRYPOINT ["dockerd-entrypoint.sh"]
CMD []
