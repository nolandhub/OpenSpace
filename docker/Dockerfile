FROM nvcr.io/nvidia/tritonserver:25.01-py3

# espeak-ng là tokenizer bắt buộc của ZipVoice cho tiếng Việt
RUN apt-get update && apt-get install -y --no-install-recommends \
        espeak-ng espeak-ng-data git \
    && rm -rf /var/lib/apt/lists/*

COPY constraints.txt /etc/pip-constraints.txt
ENV PIP_CONSTRAINT=/etc/pip-constraints.txt

# Ghim torch/torchaudio TRƯỚC khi cài ZipVoice.
# requirements.txt của ZipVoice để torch không ghim version. Cứ để pip tự chọn
# thì nó kéo torch mới nhất kèm nguyên bộ thư viện CUDA 13 - vừa phình image
# thêm gần chục GB, vừa đụng với onnxruntime-gpu vốn dựng cho CUDA 12.
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cu124 \
        torch==2.5.1 torchaudio==2.5.1

# ZipVoice không có gói trên PyPI, phải clone.
# k2 chỉ cần cho training nên bỏ qua - inference chạy không cần k2.
RUN git clone --depth 1 https://github.com/k2-fsa/ZipVoice.git /opt/ZipVoice

# torch/torchaudio đã có sẵn nên pip bỏ qua, chỉ kéo phần còn lại.
# File requirements.txt của ZipVoice có sẵn dòng --find-links cho piper_phonemize.
RUN pip install --no-cache-dir -r /opt/ZipVoice/requirements.txt

# Dependency riêng của project: ORT chạy decoder/joiner, sentencepiece dịch token id.
# soundfile do lhotse kéo về rồi nên không ghim lại ở đây.
# urllib3: lhotse import nó nhưng không khai trong dependency của mình
RUN pip install --no-cache-dir \
        onnxruntime-gpu==1.20.2 \
        sentencepiece==0.2.0 \
        urllib3

# Không `pip install /opt/ZipVoice`: pyproject.toml của nó không khai báo
# package discovery nên setuptools thấy 4 thư mục top-level rồi từ chối build.
# Repo vốn được thiết kế để chạy tại chỗ, thêm vào PYTHONPATH là đủ.
ENV PYTHONPATH=/opt/ZipVoice:${PYTHONPATH}

# Chốt lại: numpy phải là 1.x, và toàn bộ chuỗi import của ZipVoice phải chạy.
# Import cả infer_zipvoice để bắt dependency thiếu ngay lúc build thay vì lúc chạy.
RUN python3 -c "import numpy; assert numpy.__version__.startswith('1.'), numpy.__version__" \
    && python3 -c "import torch, torchaudio, onnxruntime, sentencepiece, vocos, soundfile" \
    && python3 -c "from zipvoice.bin.infer_zipvoice import generate_sentence, get_vocoder; print('deps ok')"
