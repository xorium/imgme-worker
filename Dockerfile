FROM bugfuzzer/worker-base:v1.0

# The base image comes with many system dependencies pre-installed to help you get started quickly.
# Please refer to the base image's Dockerfile for more information before adding additional dependencies.
# IMPORTANT: The base image overrides the default huggingface cache location.


# --- Optional: System dependencies ---
# COPY builder/setup.sh /setup.sh
# RUN /bin/bash /setup.sh && \
#     rm /setup.sh

# NOTE: The base image comes with multiple Python versions pre-installed.
#       It is reccommended to specify the version of Python when running your code.

# Скачиваем модели в отдельный шаг
COPY builder/download_models.py /download_models.py
RUN python3.11 -u /download_models.py

# Add src files (Worker Template)
ADD src .

CMD python3.11 -u /handler.py
