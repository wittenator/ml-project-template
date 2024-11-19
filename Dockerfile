FROM --platform=$BUILDPLATFORM mambaorg/micromamba:1.5.8-jammy-cuda-12.1.1

USER root

# Clean up
RUN rm -f /etc/apt/sources.list.d/*.list

# Utilities
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential sudo curl git htop less rsync screen vim nano wget

# Environment
COPY environment.yaml /tmp/environment.yaml
RUN micromamba install -y -n base -f /tmp/environment.yaml -v && micromamba clean -qya

# SLURM
RUN groupadd -g 12067 slurm
RUN useradd  -m -d /tmp -u 13504 -g slurm -s /bin/false slurm
RUN groupadd -g 119 munge
RUN useradd  -m -d /nonexistent -u 114 -g munge -s /usr/sbin/nologin munge

# Workdir
RUN mkdir /srv/repo/ && chmod 777 /srv/repo
ENV PYTHONPATH=$PYTHONPATH:/srv/repo
ENV PATH=$PATH:$MAMBA_ROOT_PREFIX/bin
WORKDIR /srv/repo
