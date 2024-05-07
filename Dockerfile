FROM 763104351884.dkr.ecr.ap-northeast-1.amazonaws.com/pytorch-inference:2.2.0-gpu-py310

# faster-whisper のインストール
RUN pip install faster-whisper==1.0.2
RUN pip install --force-reinstall ctranslate2==3.24.0
