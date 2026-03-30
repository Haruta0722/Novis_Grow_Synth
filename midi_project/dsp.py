"""
dsp.py  ―  DDSPシンセサイザーモジュール (モジュール分離版)

【設計思想】
  各DSPモジュールは独立しており、GUIから個別に操作・可視化できる。
  Decoderはパラメータを推論するだけで、音声生成はこのファイルが担う。

【DSPパラメータ構造】  DDSPParams (dataclass)
  ┌─ Oscillator ──────────────────────────────────────────────┐
  │  f0_hz            : float  基本周波数 [Hz]                 │
  │  harmonic_amps    : [NUM_HARMONICS]  各倍音の相対振幅       │
  └───────────────────────────────────────────────────────────┘
  ┌─ Envelope (ADSR) ─────────────────────────────────────────┐
  │  attack           : float 0〜1  アタック時間               │
  │  decay            : float 0〜1  ディケイ時間               │
  │  sustain          : float 0〜1  サステインレベル            │
  │  release          : float 0〜1  リリース時間               │
  └───────────────────────────────────────────────────────────┘
  ┌─ Filter ──────────────────────────────────────────────────┐
  │  cutoff           : float 0〜1  カットオフ周波数 (正規化)   │
  │  resonance        : float 0〜1  レゾナンス                 │
  └───────────────────────────────────────────────────────────┘
  ┌─ Noise ────────────────────────────────────────────────────┐
  │  noise_amount     : float 0〜1  ノイズ混合量               │
  └───────────────────────────────────────────────────────────┘

【ハード化時のデータフロー】
  VAEマイコン → DDSPParams (辞書) → DSPマイコン → synthesize_numpy()
  DSPマイコンは TensorFlow 不要。numpy だけで動作する。

【GUIでの操作】
  各パラメータを直接スライダー/ノブで上書きしてから synthesize_numpy() を呼ぶ。
  再現性が保たれる (乱数要素はノイズだけ)。
"""

from __future__ import annotations
from dataclasses import dataclass, field, asdict
import numpy as np

try:
    import tensorflow as tf

    HAS_TF = True
except ImportError:
    HAS_TF = False

from config import SR, TIME_LENGTH, NUM_HARMONICS


# ============================================================
# DDSPParams  ― パラメータをひとまとめにするデータクラス
# ============================================================
@dataclass
class DDSPParams:
    """
    DSPモジュール全体のパラメータ。
    Decoderの出力であり、GUIで直接操作できる単位。

    すべてのフィールドは 0〜1 に正規化されている
    (f0_hz だけ例外でHz値を直接持つ、
     unison_voices だけ例外で整数 1〜7 を直接持つ)。
    """

    # Oscillator
    f0_hz: float = 440.0
    harmonic_amps: list[float] = field(
        default_factory=lambda: [1.0] + [0.0] * (NUM_HARMONICS - 1)
    )

    # Unison
    unison_voices: int = 1  # ボイス数 1〜7 (1=OFF)
    detune_cents: float = 0.0  # デチューン幅 [セント] 0〜100
    unison_blend: float = (
        0.5  # ドライ(1ボイス)とウェット(全ボイス)のブレンド 0〜1
    )

    # Envelope (ADSR)
    attack: float = 0.1
    decay: float = 0.2
    sustain: float = 0.7
    release: float = 0.3

    # Filter
    cutoff: float = 1.0
    resonance: float = 0.0

    # Noise
    noise_amount: float = 0.0

    def to_dict(self) -> dict:
        """マイコン送信用辞書に変換"""
        d = asdict(self)
        if isinstance(d["harmonic_amps"], np.ndarray):
            d["harmonic_amps"] = d["harmonic_amps"].tolist()
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "DDSPParams":
        return cls(**d)

    def clamp(self) -> "DDSPParams":
        """全パラメータを有効範囲にクランプして返す"""
        self.unison_voices = int(np.clip(self.unison_voices, 1, 7))
        self.detune_cents = float(np.clip(self.detune_cents, 0.0, 100.0))
        self.unison_blend = float(np.clip(self.unison_blend, 0.0, 1.0))
        self.attack = float(np.clip(self.attack, 0.0, 1.0))
        self.decay = float(np.clip(self.decay, 0.0, 1.0))
        self.sustain = float(np.clip(self.sustain, 0.0, 1.0))
        self.release = float(np.clip(self.release, 0.0, 1.0))
        self.cutoff = float(np.clip(self.cutoff, 0.0, 1.0))
        self.resonance = float(np.clip(self.resonance, 0.0, 1.0))
        self.noise_amount = float(np.clip(self.noise_amount, 0.0, 1.0))
        self.harmonic_amps = list(np.clip(self.harmonic_amps, 0.0, 1.0))
        return self


# ============================================================
# パラメータ変換ユーティリティ
# ============================================================
def adsr_to_seconds(
    value: float, param: str, total: float = TIME_LENGTH / SR
) -> float:
    """
    正規化値 [0,1] → 秒数 (対数スケール)

    value=0.0 → 最短時間
    value=1.0 → 最長時間 (total の半分)

    対数スケールにすることで短い時間側の解像度を高める。
    """
    if param == "sustain":
        return float(value)  # sustain はレベルなのでそのまま返す
    min_sec = 0.005  # 最短 5ms
    max_sec = total * 0.5  # 最長 total の半分 (1.3s → 0.65s)
    # 対数補間: value=0 → min_sec, value=1 → max_sec
    return float(min_sec * (max_sec / min_sec) ** float(value))


def cutoff_to_hz(cutoff: float, sr: int = SR) -> float:
    """正規化カットオフ [0,1] → Hz (対数スケール: 20Hz〜SR/2)"""
    min_hz = 20.0
    max_hz = sr / 2.0
    return float(min_hz * (max_hz / min_hz) ** cutoff)


# ============================================================
# モジュール1: Oscillator  加算合成シンセ
# ============================================================
def oscillator_numpy(
    f0_hz: float,
    harmonic_amps: np.ndarray,  # [NUM_HARMONICS]  0〜1
    time_length: int = TIME_LENGTH,
    sr: int = SR,
) -> np.ndarray:
    """
    加算合成で倍音波形を生成する。

    harmonic_amps の値が小さい倍音はほぼ無音になるよう、
    振幅に対して二乗を適用して感度を高める。
    これにより GUIで倍音を 0 に近づけると実際に無音になる。

    Returns:
        audio: [time_length]  float32
    """
    harmonic_amps = np.array(harmonic_amps, dtype=np.float32)
    harmonic_amps = np.clip(harmonic_amps, 0.0, 1.0)
    num_harmonics = len(harmonic_amps)

    # 各倍音の周波数 [num_harmonics]
    harm_nums = np.arange(1, num_harmonics + 1, dtype=np.float32)
    harm_freqs = f0_hz * harm_nums
    harm_freqs = np.clip(harm_freqs, 0.0, sr / 2.0)

    # サンプルごとの位相増分 → 累積位相 [time, H]
    delta_phase = 2.0 * np.pi * harm_freqs[None, :] / sr
    phase = np.cumsum(np.tile(delta_phase, (time_length, 1)), axis=0)

    # 極小の倍音のみ無音とみなす
    # softmax均等分布では 1/32≈0.031 なので 0.05 は使えない
    harmonic_amps = np.where(harmonic_amps < 0.001, 0.0, harmonic_amps)

    # 合成
    audio = (harmonic_amps[None, :] * np.sin(phase)).sum(axis=1)  # [time]
    return audio.astype(np.float32)


# ============================================================
# モジュール1b: Unison  複数ボイスによるデチューン合成
# ============================================================
def unison_numpy(
    f0_hz: float,
    harmonic_amps: np.ndarray,
    unison_voices: int = 1,
    detune_cents: float = 0.0,
    unison_blend: float = 0.5,
    time_length: int = TIME_LENGTH,
    sr: int = SR,
) -> np.ndarray:
    """
    Unison合成: 複数のボイスをデチューンして加算することで音に厚みを出す。

    unison_blend でドライ(1ボイス)とウェット(全ボイス)をブレンドする。
    blend=0.0 → 1ボイスのみ、blend=1.0 → 全ボイスの平均
    """
    n = max(1, int(unison_voices))

    if n == 1 or detune_cents < 0.001:
        return oscillator_numpy(f0_hz, harmonic_amps, time_length, sr)

    dry = oscillator_numpy(f0_hz, harmonic_amps, time_length, sr)

    offsets_cents = np.linspace(
        -detune_cents / 2.0, detune_cents / 2.0, n
    ).tolist()

    wet = np.zeros(time_length, dtype=np.float32)
    for cents in offsets_cents:
        detuned_f0 = f0_hz * (2.0 ** (cents / 1200.0))
        wet += oscillator_numpy(detuned_f0, harmonic_amps, time_length, sr)
    wet /= n

    blend = float(np.clip(unison_blend, 0.0, 1.0))
    return ((1.0 - blend) * dry + blend * wet).astype(np.float32)


# ============================================================
# モジュール2: ADSR Envelope
# ============================================================
def adsr_envelope_numpy(
    attack: float,
    decay: float,
    sustain: float,
    release: float,
    time_length: int = TIME_LENGTH,
    sr: int = SR,
) -> np.ndarray:
    """
    ADSRエンベロープを生成する (ループ版・マイコン向け)

    Args:
        attack   : アタック時間 [秒]
        decay    : ディケイ時間 [秒]
        sustain  : サステインレベル [0〜1]
        release  : リリース時間 [秒]
        time_length: サンプル数
        sr       : サンプリングレート
    """
    t = np.linspace(
        0.0, time_length / sr, time_length, endpoint=False, dtype=np.float32
    )
    env = np.zeros(time_length, dtype=np.float32)
    total_t = time_length / sr

    # Attack → Decay → Sustain の区間
    a_end = attack
    d_end = attack + decay
    # Release は末尾から release 秒前に開始 (sustain 区間の後)
    r_start = max(d_end, total_t - release)

    for i, ti in enumerate(t):
        if ti < a_end:
            env[i] = ti / (a_end + 1e-6)
        elif ti < d_end:
            env[i] = 1.0 - (1.0 - sustain) * (ti - a_end) / (decay + 1e-6)
        elif ti < r_start:
            env[i] = sustain
        else:
            env[i] = sustain * max(0.0, 1.0 - (ti - r_start) / (release + 1e-6))

    return np.clip(env, 0.0, 1.0)


def adsr_envelope_numpy_fast(
    attack: float,
    decay: float,
    sustain: float,
    release: float,
    time_length: int = TIME_LENGTH,
    sr: int = SR,
) -> np.ndarray:
    """adsr_envelope_numpy のベクトル化版 (高速)"""
    t = np.linspace(
        0.0, time_length / sr, time_length, endpoint=False, dtype=np.float32
    )
    total_t = time_length / sr

    a_end = float(attack)
    d_end = float(attack + decay)
    # Release は末尾から release 秒前に開始
    # sustain区間 (d_end〜r_start) が必ず存在するよう保証する
    r_start = float(max(d_end, total_t - release))

    env = np.where(
        t < a_end,
        t / (a_end + 1e-6),
        np.where(
            t < d_end,
            1.0 - (1.0 - sustain) * (t - a_end) / (decay + 1e-6),
            np.where(
                t < r_start,
                np.full_like(t, sustain),
                sustain
                * np.clip(1.0 - (t - r_start) / (release + 1e-6), 0.0, 1.0),
            ),
        ),
    )
    return np.clip(env, 0.0, 1.0).astype(np.float32)


# ============================================================
# モジュール3: State Variable Filter (SVF)
# ============================================================
def svf_filter_numpy(
    audio: np.ndarray,  # [time_length]
    cutoff: float,  # 0〜1 (正規化)
    resonance: float,  # 0〜1
    mode: str = "lowpass",  # "lowpass" | "highpass" | "bandpass"
    sr: int = SR,
) -> np.ndarray:
    """
    State Variable Filter (SVF) のNumPy実装。

    サンプルごとにループするため低速だが、マイコンでも動作する。
    cutoff=1.0 でほぼ全域通過、cutoff=0.0 でほぼ全域遮断。
    resonance=0.0 でフラット、resonance=1.0 でセルフオシレーション寸前。

    Args:
        audio     : 入力波形 [time_length]
        cutoff    : 正規化カットオフ [0〜1]
        resonance : レゾナンス [0〜1]
        mode      : フィルタモード
        sr        : サンプリングレート

    Returns:
        filtered: [time_length]  float32
    """
    cutoff_hz = cutoff_to_hz(cutoff, sr)
    # SVFの係数
    f = 2.0 * np.sin(np.pi * cutoff_hz / sr)
    f = np.clip(f, 0.0, 1.0)
    q = 1.0 - resonance * 0.99  # Q: 1.0(フラット)→0.01(高共振)

    n = len(audio)
    out = np.zeros(n, dtype=np.float32)
    lp = 0.0  # ローパス状態
    bp = 0.0  # バンドパス状態

    for i in range(n):
        x = float(audio[i])
        hp = x - lp - q * bp
        bp = f * hp + bp
        lp = f * bp + lp

        if mode == "lowpass":
            out[i] = lp
        elif mode == "highpass":
            out[i] = hp
        elif mode == "bandpass":
            out[i] = bp

    return np.clip(out, -2.0, 2.0).astype(np.float32)


def svf_filter_numpy_fast(
    audio: np.ndarray,
    cutoff: float,
    resonance: float,
    mode: str = "lowpass",
    sr: int = SR,
) -> np.ndarray:
    """
    SVFのベクトル化近似版。
    完全なサンプル単位ループの代わりにチャンク処理で高速化。
    マイコン用途には svf_filter_numpy() を使うこと。
    """
    from scipy.signal import butter, sosfilt

    cutoff_hz = float(np.clip(cutoff_to_hz(cutoff, sr), 20.0, sr / 2.0 - 1.0))
    q_factor = float(np.clip(0.5 + resonance * 9.5, 0.5, 10.0))

    nyq = sr / 2.0
    norm = cutoff_hz / nyq

    try:
        if mode == "lowpass":
            sos = butter(2, norm, btype="low", output="sos")
        elif mode == "highpass":
            sos = butter(2, norm, btype="high", output="sos")
        else:
            bw = norm / q_factor
            low = max(1e-4, norm - bw / 2)
            high = min(0.9999, norm + bw / 2)
            sos = butter(2, [low, high], btype="band", output="sos")
        return sosfilt(sos, audio).astype(np.float32)
    except Exception:
        return audio.astype(np.float32)


# ============================================================
# モジュール4: Noise Generator
# ============================================================
def noise_generator_numpy(
    noise_amount: float,
    time_length: int = TIME_LENGTH,
    seed: int = None,
) -> np.ndarray:
    """
    ホワイトノイズを生成して noise_amount でスケールする。

    Returns:
        noise: [time_length]  float32
    """
    if seed is not None:
        np.random.seed(seed)
    noise = np.random.randn(time_length).astype(np.float32)
    return noise * float(np.clip(noise_amount, 0.0, 1.0))


# ============================================================
# メイン合成関数  synthesize_numpy()
# ============================================================
def synthesize_numpy(
    params: DDSPParams,
    sr: int = SR,
    time_length: int = TIME_LENGTH,
    fast_filter: bool = True,
    seed: int = None,
) -> np.ndarray:
    """
    DDSPParams から音声波形を合成する。

    処理フロー:
      1. Oscillator + Unison → 倍音波形 (複数ボイスのデチューン合成)
      2. ADSR        → 振幅エンベロープ
      3. Filter      → カットオフ / レゾナンス
      4. Noise       → ノイズ混合
      5. tanh        → クリッピング

    Args:
        params      : DDSPParams インスタンス
        sr          : サンプリングレート
        time_length : 出力サンプル数
        fast_filter : True = scipy版フィルタ(高速), False = SVFループ版(マイコン向け)
        seed        : 乱数シード

    Returns:
        audio: float32 [time_length]  (-1〜1)
    """
    params = params.clamp()

    # --- 1. Oscillator + Unison ---
    audio = unison_numpy(
        f0_hz=params.f0_hz,
        harmonic_amps=np.array(params.harmonic_amps, dtype=np.float32),
        unison_voices=params.unison_voices,
        detune_cents=params.detune_cents,
        unison_blend=params.unison_blend,
        time_length=time_length,
        sr=sr,
    )

    # --- 2. ADSR Envelope ---
    a_sec = adsr_to_seconds(params.attack, "attack", time_length / sr)
    d_sec = adsr_to_seconds(params.decay, "decay", time_length / sr)
    r_sec = adsr_to_seconds(params.release, "release", time_length / sr)
    s_lv = params.sustain  # sustain はレベルなので変換不要

    envelope = adsr_envelope_numpy_fast(
        a_sec, d_sec, s_lv, r_sec, time_length, sr
    )
    audio = audio * envelope

    # --- 3. Filter ---
    if params.cutoff < 0.999:  # cutoff=1.0 はフルオープン→スキップ
        if fast_filter:
            audio = svf_filter_numpy_fast(
                audio, params.cutoff, params.resonance, mode="lowpass", sr=sr
            )
        else:
            audio = svf_filter_numpy(
                audio, params.cutoff, params.resonance, mode="lowpass", sr=sr
            )

    # --- 4. Noise ---
    if params.noise_amount > 1e-4:
        noise = noise_generator_numpy(params.noise_amount, time_length, seed)
        audio = audio + noise * 0.1  # ノイズ実効量を1/10にスケール

    # --- 5. tanh クリッピング ---
    audio = np.tanh(audio).astype(np.float32)
    return audio


# ============================================================
# TFレイヤー版  (学習時に cvae.py から使用)
# ============================================================
if HAS_TF:

    def upsample_frames(x, target_length):
        """[batch, frames, ch] → [batch, target_length, ch]"""
        x = tf.expand_dims(x, axis=2)
        x = tf.image.resize(x, [target_length, 1], method="bilinear")
        x = tf.squeeze(x, axis=2)
        return x

    class OscillatorLayer(tf.keras.layers.Layer):
        """加算合成シンセ (TFレイヤー版・学習用)"""

        def __init__(self, sr=SR, time_length=TIME_LENGTH):
            super().__init__(name="oscillator")
            self.sr = float(sr)
            self.time_length = time_length

        def call(self, f0_hz, harmonic_amps):
            """
            Args:
                f0_hz        : [batch]
                harmonic_amps: [batch, NUM_HARMONICS]  (softmax済み)
            Returns:
                audio: [batch, time_length]
            """
            num_harmonics = tf.shape(harmonic_amps)[1]

            harm_nums = tf.cast(tf.range(1, num_harmonics + 1), tf.float32)
            harm_freqs = tf.reshape(f0_hz, [-1, 1]) * tf.reshape(
                harm_nums, [1, -1]
            )
            harm_freqs = tf.clip_by_value(harm_freqs, 0.0, self.sr / 2.0)

            # [batch, time, H]
            delta = (
                2.0
                * np.pi
                * tf.reshape(harm_freqs, [-1, 1, num_harmonics])
                / self.sr
            )
            delta = tf.tile(delta, [1, self.time_length, 1])
            phase = tf.cumsum(delta, axis=1)

            amps = tf.reshape(harmonic_amps, [-1, 1, num_harmonics])
            audio = tf.reduce_sum(amps * tf.sin(phase), axis=-1)
            return tf.clip_by_value(audio, -10.0, 10.0)

    class ADSRLayer(tf.keras.layers.Layer):
        """ADSRエンベロープ生成 (TFレイヤー版・学習用)"""

        def __init__(self, sr=SR, time_length=TIME_LENGTH):
            super().__init__(name="adsr")
            self.sr = float(sr)
            self.time_length = time_length
            self.total_t = time_length / sr

        def call(self, attack, decay, sustain, release):
            """
            Args: 各パラメータ [batch]  0〜1 の正規化値
            Returns: envelope [batch, time_length]
            """
            # 秒数に変換 (対数スケール: value=0→min_sec, value=1→max_sec)
            min_sec = 0.005
            max_sec = self.total_t * 0.5

            def to_sec(v):
                return min_sec * tf.pow(max_sec / min_sec, v)

            a = to_sec(attack)  # [batch]
            d = to_sec(decay)
            r = to_sec(release)
            s = sustain

            t = tf.cast(
                tf.linspace(0.0, self.total_t, self.time_length), tf.float32
            )  # [time]

            # ブロードキャスト: [batch, time]
            a = tf.reshape(a, [-1, 1])
            d = tf.reshape(d, [-1, 1])
            s = tf.reshape(s, [-1, 1])
            r = tf.reshape(r, [-1, 1])
            t = tf.reshape(t, [1, -1])

            a_end = a
            d_end = a + d
            r_start = tf.maximum(d_end, self.total_t - r)

            env = tf.where(
                t < a_end,
                t / (a_end + 1e-6),
                tf.where(
                    t < d_end,
                    1.0 - (1.0 - s) * (t - a_end) / (d + 1e-6),
                    tf.where(
                        t < r_start,
                        s,
                        s
                        * tf.clip_by_value(
                            1.0 - (t - r_start) / (r + 1e-6), 0.0, 1.0
                        ),
                    ),
                ),
            )
            return tf.clip_by_value(env, 0.0, 1.0)  # [batch, time]

    class FilterLayer(tf.keras.layers.Layer):
        """
        時変LPF (TFレイヤー版・学習用)

        完全なIIRフィルタはTFグラフモードで実装困難なため、
        スペクトルマスキングで近似する。
        """

        def __init__(self, sr=SR, time_length=TIME_LENGTH, n_fft=2048):
            super().__init__(name="filter")
            self.sr = float(sr)
            self.time_length = time_length
            self.n_fft = n_fft

        def call(self, audio, cutoff, resonance):
            """
            Args:
                audio     : [batch, time_length]
                cutoff    : [batch]  0〜1
                resonance : [batch]  0〜1
            Returns:
                filtered: [batch, time_length]
            """
            # cutoff → Hz → 正規化周波数
            min_hz = 20.0
            max_hz = self.sr / 2.0
            cutoff_hz = min_hz * tf.pow(max_hz / min_hz, cutoff)  # [batch]

            # STFT
            stft = tf.signal.stft(
                audio, self.n_fft, self.n_fft // 4, self.n_fft
            )
            n_bins = tf.shape(stft)[-1]

            # 周波数軸 [n_bins]
            freqs = tf.cast(tf.range(n_bins), tf.float32) * (
                self.sr / self.n_fft
            )

            # ローパスマスク [batch, 1, n_bins]
            cutoff_bc = tf.reshape(cutoff_hz, [-1, 1, 1])
            res_bc = tf.reshape(resonance, [-1, 1, 1])

            # ソフトローパス + レゾナンスピーク
            mask = tf.sigmoid((cutoff_bc - freqs) / (cutoff_bc * 0.1 + 1.0))
            # レゾナンスによるカットオフ付近のブースト
            peak = res_bc * tf.exp(
                -tf.square((freqs - cutoff_bc) / (cutoff_bc * 0.05 + 1.0))
            )
            mask = mask + peak * (1.0 - mask)
            mask = tf.clip_by_value(mask, 0.0, 2.0)

            # フィルタリング
            filtered_stft = stft * tf.cast(mask, tf.complex64)
            filtered_audio = tf.signal.inverse_stft(
                filtered_stft, self.n_fft, self.n_fft // 4, self.n_fft
            )
            # 長さを合わせる
            filtered_audio = filtered_audio[:, : self.time_length]
            pad_len = self.time_length - tf.shape(filtered_audio)[1]
            filtered_audio = tf.pad(filtered_audio, [[0, 0], [0, pad_len]])

            return tf.clip_by_value(filtered_audio, -10.0, 10.0)
