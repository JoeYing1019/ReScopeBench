# The release's Hermes runtime with the harness changes (see apply_hermes_patches.py).
# Base image: load it from the release first,
#   docker load --input ../rescopebench/vendor/hermes-runtime-portable-reviewed18-v2-20261015.tar.gz
ARG BASE_IMAGE=rescopebench-portable:reviewed18-20261015
FROM ${BASE_IMAGE}
USER 0:0
COPY apply_hermes_patches.py context_lengths.json /tmp/rescope-harness/
RUN /usr/local/bin/python3 /tmp/rescope-harness/apply_hermes_patches.py /opt/hermes /tmp/rescope-harness/context_lengths.json \
 && rm -rf /tmp/rescope-harness
LABEL io.rescopebench.harness="hermes timeouts, no Kimi schema rewrite, context windows"
