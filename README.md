# 환경 설치(가상환경):
.\.venv\Scripts\Activate
pip install --upgrade pip
pip install --no-cache-dir -r requirements.txt

# 버전 확인:
python -c "import numpy, torch, torchvision, torchaudio, huggingface_hub, pytorch_lightning, torchmetrics; \
print('numpy', numpy.__version__, '| torch', torch.__version__, '| tv', torchvision.__version__, '| ta', torchaudio.__version__, '| hub', huggingface_hub.__version__, '| pl', pytorch_lightning.__version__, '| tm', torchmetrics.__version__)"

--> 이렇게 나와야 함.
numpy 1.26.4 | torch 2.3.1 | tv 0.18.1 | ta 2.3.1 | hub 0.13.4 | pl 1.9.5 | tm 0.11.4

# 실행법: 
python enhance_dccrn.py --in noisy\noisy1.wav --out clean\noisy1_clean.wav

# 또는 후처리 버전:
python enhance_dccrn_post.py --in noisy\noisy1.wav --out clean\noisy1_post.wav --pre_gain_db 12 --mix 0.8
