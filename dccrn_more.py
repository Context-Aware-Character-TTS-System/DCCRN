import argparse, os, sys, tempfile, math
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

# ----------------- 유틸 -----------------
def apply_gain_db(x, db):
    g = 10 ** (db / 20.0)
    return x * g

def peak_dbfs(x):
    peak = float(np.max(np.abs(x)) + 1e-12)
    return 20 * math.log10(peak)

def normalize_peak(x, target_dbfs=-1.0):
    cur = peak_dbfs(x)
    need = target_dbfs - cur
    return apply_gain_db(x, need)

def rms_dbfs(x):
    rms = float(np.sqrt(np.mean(x**2)) + 1e-12)
    return 20 * math.log10(rms)

try:
    import pyloudnorm as pyln
    HAVE_LUFS = True
except Exception:
    HAVE_LUFS = False

def normalize_lufs(x, sr, target_lufs=-20.0):
    if not HAVE_LUFS:
        # LUFS 모듈 없으면 RMS 기준 근사
        cur = rms_dbfs(x)
        need = target_lufs - cur
        return apply_gain_db(x, need)
    meter = pyln.Meter(sr)
    loud = meter.integrated_loudness(x)
    need = target_lufs - loud
    y = apply_gain_db(x, need)
    return normalize_peak(y, -1.0)

# ----------------- 입출력 -----------------
def load_audio_mono_16k(path: str):
    y, sr = sf.read(path, dtype="float32", always_2d=False)
    if y.ndim > 1:
        y = np.mean(y, axis=1)  # 스테레오 -> 모노
    if sr != TARGET_SR:
        y = librosa.resample(y, orig_sr=sr, target_sr=TARGET_SR)
    return y, TARGET_SR

def save_wav(path: str, y: np.ndarray, sr: int = TARGET_SR):
    y = np.asarray(y)
    y = np.squeeze(y)
    if y.ndim == 2:
        if y.shape[0] <= 8:
            y = y.mean(axis=0)
        else:
            y = y.mean(axis=1)
    elif y.ndim > 2:
        axes = tuple(range(y.ndim - 1))
        y = y.mean(axis=axes)
    peak = np.max(np.abs(y)) + 1e-12
    if peak > 1.0:
        y = y / peak
    sf.write(path, y.astype(np.float32), sr)

# ----------------- 핵심 처리 -----------------
def enhance_file(model: Mayamodel, in_wav: str, out_wav: str, args):
    # 입력 읽기 + 프리게인
    y, sr = load_audio_mono_16k(in_wav)
    if args.pre_gain_db and abs(args.pre_gain_db) > 0.01:
        y = apply_gain_db(y, args.pre_gain_db)

    # 추론
    try:
        denoised = model.enhance(y, sr)
    except Exception:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp_in = tmp.name
        save_wav(tmp_in, y, sr)
        denoised = model.enhance(tmp_in)
        try:
            os.remove(tmp_in)
        except Exception:
            pass

    # 타입 정리
    if isinstance(denoised, torch.Tensor):
        denoised = denoised.squeeze().detach().cpu().numpy()
    denoised = np.squeeze(np.asarray(denoised))

    # 드라이/웨트 믹스
    mix = float(np.clip(args.mix, 0.0, 1.0))
    minlen = min(len(denoised), len(y))
    wet = denoised[:minlen]
    dry = y[:minlen]
    out = mix * wet + (1.0 - mix) * dry

    # 출력 노멀라이즈
    if not args.no_lufs:
        out = normalize_lufs(out, sr, args.target_lufs)
    if not args.no_peaknorm:
        out = normalize_peak(out, -1.0)

    save_wav(out_wav, out, TARGET_SR)

# ----------------- CLI -----------------
def main():
    ap = argparse.ArgumentParser(description="DCCRN noise suppression (mayavoz)")
    ap.add_argument("--in", dest="in_path", help="입력 WAV 파일 경로")
    ap.add_argument("--out", dest="out_path", help="출력 WAV 파일 경로")
    ap.add_argument("--in_dir", help="일괄 처리: 입력 폴더(*.wav)")
    ap.add_argument("--out_dir", help="일괄 처리: 출력 폴더")

    # 새 옵션들
    ap.add_argument("--pre_gain_db", type=float, default=12.0, help="모델 입력 전 프리게인(dB)")
    ap.add_argument("--mix", type=float, default=0.8, help="웨트(모델 출력) 비율 0~1")
    ap.add_argument("--target_lufs", type=float, default=-20.0, help="출력 목표 LUFS(없으면 RMS 근사)")
    ap.add_argument("--no_lufs", action="store_true", help="LUFS/RMS 노멀라이즈 끄기")
    ap.add_argument("--no_peaknorm", action="store_true", help="최종 피크 노멀라이즈 끄기")

    args = ap.parse_args()

    print("[+] 모델 로딩 중... (최초 1회는 다운로드)")
    model = Mayamodel.from_pretrained(MODEL_ID)

    if args.in_path and args.out_path:
        os.makedirs(os.path.dirname(args.out_path) or ".", exist_ok=True)
        enhance_file(model, args.in_path, args.out_path, args)
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
            enhance_file(model, in_wav, out_wav, args)
        print(f"[✓] 일괄 처리 완료: {len(wavs)}개 파일")
        return

    ap.print_help()

if __name__ == "__main__":
    main()
