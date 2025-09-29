import argparse, os, sys, tempfile
import numpy as np
import soundfile as sf
import librosa

try:
    import torch
except ImportError:
    print("PyTorch가 필요합니다. 먼저 torch를 설치하세요.")
    sys.exit(1)

from mayavoz.models import Mayamodel

MODEL_ID = "shahules786/mayavoz-dccrn-valentini-28spk"
TARGET_SR = 16000

def load_audio_mono_16k(path: str):
    # 어떤 샘플레이트/채널이 와도 16kHz, mono로 변환
    y, sr = sf.read(path, dtype="float32", always_2d=False)
    if y.ndim > 1:
        y = np.mean(y, axis=1)  # 스테레오 -> 모노
    if sr != TARGET_SR:
        y = librosa.resample(y, orig_sr=sr, target_sr=TARGET_SR)
    return y, TARGET_SR

def save_wav(path: str, y: np.ndarray, sr: int = TARGET_SR):
    y = np.asarray(y)

    # ⬇️ 모양 보정: (1,1,N)→(N,), (C,N)/(N,C)→모노
    y = np.squeeze(y)
    if y.ndim == 2:  # (channels, samples) or (samples, channels)
        # 채널 수가 작으면 (C, N)일 확률이 높음 → 채널 평균
        if y.shape[0] <= 8:
            y = y.mean(axis=0)
        else:
            y = y.mean(axis=1)
    elif y.ndim > 2:
        # 마지막 축을 샘플로 가정하고 나머지 축 평균
        axes = tuple(range(y.ndim - 1))
        y = y.mean(axis=axes)

    # (선택) 클리핑 방지용 정규화
    peak = np.max(np.abs(y)) + 1e-12
    if peak > 1.0:
        y = y / peak

    sf.write(path, y.astype(np.float32), sr)

def enhance_file(model: Mayamodel, in_wav: str, out_wav: str):
    # mayavoz.enhance 가 경로/파형 두 방식 모두 가능한 버전이 있어, 호환 처리
    # 1) 파형으로 직접 넣기 시도
    y, sr = load_audio_mono_16k(in_wav)
    try:
        denoised = model.enhance(y, sr)  # 일부 버전은 (ndarray, sr) 허용
    except Exception:
        # 2) 임시파일 경로로 처리
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp_in = tmp.name
        save_wav(tmp_in, y, sr)
        denoised = model.enhance(tmp_in)
        try:
            os.remove(tmp_in)
        except Exception:
            pass

    # 반환 타입 호환(텐서/넘파이/경로)
    if isinstance(denoised, torch.Tensor):
        denoised = denoised.squeeze().detach().cpu().numpy()
    elif isinstance(denoised, np.ndarray):
        denoised = np.squeeze(denoised)
    elif isinstance(denoised, str) and os.path.exists(denoised):
        # 어떤 구현은 파일로 바로 저장하고 경로를 반환할 수 있음
        # 이 경우 그대로 복사 저장
        y2, sr2 = sf.read(denoised, dtype="float32")
        if y2.ndim > 1: y2 = np.mean(y2, axis=1)
        save_wav(out_wav, y2, TARGET_SR)
        return

    save_wav(out_wav, denoised, TARGET_SR)

def main():
    ap = argparse.ArgumentParser(description="DCCRN noise suppression (mayavoz)")
    ap.add_argument("--in", dest="in_path", help="입력 WAV 파일 경로")
    ap.add_argument("--out", dest="out_path", help="출력 WAV 파일 경로")
    ap.add_argument("--in_dir", help="일괄 처리: 입력 폴더(*.wav)")
    ap.add_argument("--out_dir", help="일괄 처리: 출력 폴더")
    args = ap.parse_args()

    print("[+] 모델 로딩 중... (최초 1회는 다운로드)")
    model = Mayamodel.from_pretrained(MODEL_ID)

    if args.in_path and args.out_path:
        os.makedirs(os.path.dirname(args.out_path) or ".", exist_ok=True)
        enhance_file(model, args.in_path, args.out_path)
        print(f"[✓] 저장 완료: {args.out_path}")
        return

    if args.in_dir and args.out_dir:
        os.makedirs(args.out_dir, exist_ok=True)
        wavs = [f for f in os.listdir(args.in_dir) if f.lower().endswith(".wav")]
        if not wavs:
            print("입력 폴더에 .wav 파일이 없습니다.")
            return
        for fname in wavs:
            in_wav = os.path.join(args.in_dir, fname)
            base, _ = os.path.splitext(fname)
            out_wav = os.path.join(args.out_dir, base + "_dccrn.wav")
            print(f" -> {fname}")
            enhance_file(model, in_wav, out_wav)
        print(f"[✓] 일괄 처리 완료: {len(wavs)}개 파일")
        return

    ap.print_help()

if __name__ == "__main__":
    main()
