FROM python:3.12-slim AS linux-base

ARG _USERNAME=driverless
ENV USERNAME=${_USERNAME}

RUN groupadd --gid 1000 $_USERNAME \
    && useradd --uid 1000 --gid 1000 -m $_USERNAME \
    #
    # [Optional] Add sudo support. Omit if you don't need to install software after connecting.
    && apt-get update \
    && apt-get install -y sudo wget curl git htop less rsync screen vim nano wget build-essential \
    && echo $_USERNAME ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/$_USERNAME \
    && chmod 0440 /etc/sudoers.d/$_USERNAME \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /home/${_USERNAME}
USER $_USERNAME

# Environment variables
ENV UV_LINK_MODE=copy
ENV UV_PYTHON=python3.12
ENV PATH="$UV_PROJECT_ENVIRONMENT/bin:$PATH"


# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/
