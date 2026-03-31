import os
import numpy as np
import pandas as pd
import tensorflow as tf
import librosa
from model import SR

# 音色ごとのクリップ長 (サンプル数)
# Pluck は急速減衰のため短め。Acid/Screech は長めで特徴を捉える。
CLIP_LEN = {
    0: int(SR * 1.3),  # Screech: 1.3秒
    1: int(SR * 2.0),  # Acid: 2.0秒 (うねりのサイクルを複数含める)
    2: int(SR * 0.5),  # Pluck: 0.5秒 (急速減衰部分だけ)
}
DEFAULT_LEN = int(SR * 1.3)


def load_wav(path):
    y, _ = librosa.load(path, sr=SR, mono=True)
    # RMS正規化
    rms = np.sqrt(np.mean(y**2) + 1e-9)
    target_rms = 0.1
    y = y / rms * target_rms
    return y.astype(np.float32)


def trim_onset(wav, top_db=40):
    """先頭の無音をトリムしてonsetから始まるようにする"""
    _, (start, _) = librosa.effects.trim(wav, top_db=top_db)
    return wav[start:]


def crop_or_pad_fixed(wav, target_len):
    """
    先頭固定クロップ: 常に先頭から target_len サンプルを使う。
    Pluck のように先頭に音があるデータに適している。
    音が短い場合はゼロパディング。
    """
    length = len(wav)
    if length >= target_len:
        return wav[:target_len]
    pad_len = target_len - length
    return np.pad(wav, (0, pad_len), constant_values=0.0).astype(np.float32)


def make_dataset_from_synth_csv(
    csv_path,
    base_dir=".",
    batch_size=16,
    shuffle=True,
):
    df = pd.read_csv(csv_path)

    df["path"] = df["path"].apply(
        lambda p: os.path.join(base_dir, p) if not os.path.isabs(p) else p
    )

    # 全音色の最大クリップ長 (バッチを揃えるために必要)
    max_len = max(CLIP_LEN.values())

    def gen():
        for _, row in df.iterrows():
            wav = load_wav(row["path"])

            # onset前の無音をトリム
            wav = trim_onset(wav)

            # timbre_id: screech=0, acid=1, pluck=2
            timbre_id = int(np.argmax([
                row["screech"], row["acid"], row["pluck"]
            ]))

            # 音色別クリップ長で先頭固定クロップ
            target_len = CLIP_LEN.get(timbre_id, DEFAULT_LEN)
            wav = crop_or_pad_fixed(wav, target_len)

            # バッチを揃えるため max_len にゼロパディング
            if target_len < max_len:
                wav = np.pad(
                    wav, (0, max_len - target_len), constant_values=0.0
                ).astype(np.float32)

            # pitch: 36–71 → 0–1 正規化
            pitch = np.float32((row["pitch"] - 36.0) / (71.0 - 36.0))

            # autoencoder なので label は使わない
            yield wav, pitch, np.int32(timbre_id)

    output_signature = (
        tf.TensorSpec(shape=(max_len,), dtype=tf.float32),
        tf.TensorSpec(shape=(), dtype=tf.float32),
        tf.TensorSpec(shape=(), dtype=tf.int32),
    )

    ds = tf.data.Dataset.from_generator(gen, output_signature=output_signature)

    if shuffle:
        ds = ds.shuffle(256)

    # (audio [T,1], pitch, timbre_id) の形式に整形
    # train_step が期待する ((audio, pitch, timbre_id), dummy) にする
    ds = ds.map(
        lambda wav, pitch, tid: (
            (tf.expand_dims(wav, -1), pitch, tid),
            tf.constant(0.0),
        ),
        num_parallel_calls=tf.data.AUTOTUNE,
    )

    ds = ds.batch(batch_size)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds
