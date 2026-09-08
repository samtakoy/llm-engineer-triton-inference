# из полного образа берём бэкенд onnxruntime
# FROM nvcr.io/nvidia/tritonserver:25.05-py3 AS ort
# Зеркало без oauth
FROM hubimage/nvcr-io-nvidia-tritonserver:25.05-py3 AS ort


# основа — образ с vllm: vllm, его бэкенд и transformers
FROM nvcr.io/nvidia/tritonserver:25.05-vllm-python-py3

COPY --from=ort /opt/tritonserver/backends/onnxruntime \
                /opt/tritonserver/backends/onnxruntime

# sentencepiece нужен токенизатору e5 (словарь XLM-R)
RUN pip install --no-cache-dir transformers sentencepiece
# RUN pip install --no-cache-dir --upgrade transformers sentencepiece

