"""
cvae.py  ―  Conditional VAE (再設計版)

【設計の核心】
  条件ごとの違いを出すには「condをDecoderが必ず使う構造」にする必要がある。

  旧設計の問題:
    Encoder(audio, cond) → z → Decoder(z, cond)
    → 再構成損失を最小化するとき Decoder は z だけ見れば十分なので
      cond を無視することを学習してしまう (条件無視問題)

  新設計:
    Encoder(audio) → z         ← condを受け取らない
    Decoder(z, cond) → params  ← condはDecoderだけが受け取る

  この設計だと:
    - z は「condで説明できない音声固有の残差」だけを保持する
    - condは音色カテゴリ固有のパラメータ (倍音・ノイズ量) を決定する
    - 推論時に cond を変えると必ず出力が変わる (条件が効く)

  さらに Decoder を2経路に分ける:
    cond_path : cond だけから倍音・ノイズの「骨格」を推論
    z_path    : z から音声固有の「ずれ」(ビブラート・アタックの個性等) を推論
    → 加算して最終パラメータを出力

  テンプレート損失は廃止し、音源から直接学習する。
"""

import numpy as np
import tensorflow as tf
from loss import Loss

from config import (
    SR,
    TIME_LENGTH,
    LATENT_DIM,
    LATENT_STEPS,
    NUM_HARMONICS,
    COND_DIM,
    ENCODER_CHANNELS,
    TIMBRE_VOCAB,
    TIMBRE_EMBED_DIM,
)
from dsp import (
    DDSPParams,
    OscillatorLayer,
    ADSRLayer,
    FilterLayer,
    upsample_frames,
)


# ============================================================
# 損失関数ユーティリティ
# ============================================================


# ============================================================
# TimbreEmbedding
# ============================================================
class TimbreEmbedding(tf.keras.layers.Layer):
    def __init__(self, vocab_size=TIMBRE_VOCAB, embed_dim=TIMBRE_EMBED_DIM):
        super().__init__(name="timbre_embedding")
        self.embed_dim = embed_dim
        self.embedding = tf.keras.layers.Embedding(
            input_dim=vocab_size,
            output_dim=embed_dim,
            name="timbre_embed_table",
        )

    def call(self, pitch, timbre_id):
        emb = self.embedding(timbre_id)
        return tf.concat([pitch[:, None], emb], axis=-1)

    def blend(self, pitch, timbre_weights):
        all_emb = self.embedding(tf.range(self.embedding.input_dim))
        mixed = tf.matmul(timbre_weights, all_emb)
        return tf.concat([pitch[:, None], mixed], axis=-1)


# ============================================================
# Encoder  q(z | x)  ← condを受け取らない
# ============================================================
def build_encoder(latent_dim=LATENT_DIM, cond_dim=COND_DIM):
    """
    Encoder q(z | x, c)

    EncoderにもCondを渡す理由:
      学習時の目標は q(z|x,c) ≈ p(z|c) = N(0,I) にすること。
      c を入れることで z から「c が説明できる情報（音色の違い）」が
      追い出され、z には「c で説明できない残差（音源ごとの個性）」
      だけが残る。

    cond の注入方法:
      Conv1D 特徴量に cond を RepeatVector でブロードキャストして concat。
      ただしその後の Conv1D を1層だけにして、
      z への cond の混入を最小限に抑える。
    """
    x_in = tf.keras.Input(shape=(TIME_LENGTH, 1), name="enc_audio")
    cond = tf.keras.Input(shape=(cond_dim,), name="enc_cond")

    x = x_in
    for ch, k, s in ENCODER_CHANNELS:
        x = tf.keras.layers.Conv1D(ch, k, strides=s, padding="same")(x)
        x = tf.keras.layers.LeakyReLU(0.2)(x)
        x = tf.keras.layers.Dropout(0.1)(x)
    # x: [batch, LATENT_STEPS, 512]

    # cond をフレーム方向にブロードキャストして浅く融合
    cond_bc = tf.keras.layers.RepeatVector(LATENT_STEPS)(cond)
    x_cond = tf.keras.layers.Concatenate(axis=-1)([x, cond_bc])
    # 融合Conv を1層だけ (深くするとzにcondが混入しすぎる)
    x_cond = tf.keras.layers.Conv1D(512, 1, padding="same", activation="relu")(
        x_cond
    )

    z_mean = tf.keras.layers.Conv1D(
        latent_dim, 3, padding="same", name="z_mean"
    )(x_cond)
    z_logvar = tf.keras.layers.Conv1D(
        latent_dim,
        3,
        padding="same",
        bias_initializer=tf.keras.initializers.Constant(-2.0),
        name="z_logvar_raw",
    )(x_cond)
    z_logvar = tf.keras.layers.Lambda(
        lambda v: tf.clip_by_value(v, -8.0, 2.0), name="z_logvar"
    )(z_logvar)

    return tf.keras.Model([x_in, cond], [z_mean, z_logvar], name="encoder")


def sample_z(z_mean, z_logvar):
    return z_mean + tf.exp(0.5 * z_logvar) * tf.random.normal(tf.shape(z_mean))


# ============================================================
# DDSPParameterDecoder  (z, cond) → DDSPパラメータ
# ============================================================
class FiLM(tf.keras.layers.Layer):
    """
    Feature-wise Linear Modulation (FiLM)
    Perez et al. 2018 で提案された条件付け手法。

    cond から γ (スケール) と β (シフト) を生成し、
    z の特徴量 x に対して x * γ + β を適用する。

    効果:
      - cond が z の表現空間全体を変調する → cond が支配的
      - z の情報は γ, β を通じてすべてのパラメータに反映される
      - 加算より表現力が高く、condとzの自然な協調が生まれる
    """

    def __init__(self, units):
        super().__init__()
        self.gamma_layer = tf.keras.layers.Dense(units, name="film_gamma")
        self.beta_layer = tf.keras.layers.Dense(units, name="film_beta")

    def call(self, x, cond):
        """
        x    : [batch, units]  z から得た特徴量
        cond : [batch, cond_dim]
        """
        gamma = self.gamma_layer(cond)  # [batch, units]
        beta = self.beta_layer(cond)  # [batch, units]
        return x * (1.0 + gamma) + beta  # 1.0 + gamma でスケールが1付近に初期化


class DDSPParameterDecoder(tf.keras.layers.Layer):
    """
    FiLM-based Decoder。

    全パラメータをzとcondの両方で制御し、condが支配的になる。

    処理フロー:
      z → GRU → MLP → 中間特徴量 h
      h × FiLM(cond) → 各パラメータヘッド

    FiLMにより cond が h 全体をスケール・シフトするため、
      - condを変えると必ずすべてのパラメータが変わる (cond支配)
      - zも全パラメータに寄与できる (個体差の表現)
    """

    def __init__(self, num_harmonics=NUM_HARMONICS):
        super().__init__(name="ddsp_param_decoder")
        self.num_harmonics = num_harmonics

        # z の時系列を集約
        self.gru = tf.keras.layers.GRU(256, return_sequences=False, name="gru")

        # z から中間特徴量を抽出
        self.z_mlp = tf.keras.Sequential(
            [
                tf.keras.layers.Dense(256, activation="relu"),
                tf.keras.layers.Dense(256, activation="relu"),
            ],
            name="z_mlp",
        )

        # FiLM: cond で中間特徴量を変調
        self.film = FiLM(256)

        # FiLM後の統合MLP
        self.post_mlp = tf.keras.Sequential(
            [
                tf.keras.layers.Dense(256, activation="relu"),
                tf.keras.layers.Dense(128, activation="relu"),
            ],
            name="post_mlp",
        )

        # 各パラメータへの独立ヘッド (すべて同じ 128次元の特徴量から生成)
        # harm_head のバイアス初期値: 基音(1倍音)が強く高次が弱い自然な初期分布
        # sigmoid(-k*(n-1)) で n=1→1.0, n=32→小さい値 になるよう設定
        harm_bias_init = np.array(
            [2.0 - i * 0.18 for i in range(num_harmonics)], dtype=np.float32
        )  # sigmoid 適用後: 1倍音≈0.88, 8倍音≈0.5, 32倍音≈0.07
        self.harm_head = tf.keras.layers.Dense(
            num_harmonics,
            bias_initializer=tf.keras.initializers.Constant(harm_bias_init),
            name="harm_head",
        )
        self.attack_head = tf.keras.layers.Dense(
            1, activation="sigmoid", name="attack_head"
        )
        self.decay_head = tf.keras.layers.Dense(
            1, activation="sigmoid", name="decay_head"
        )
        self.sustain_head = tf.keras.layers.Dense(
            1, activation="sigmoid", name="sustain_head"
        )
        self.release_head = tf.keras.layers.Dense(
            1, activation="sigmoid", name="release_head"
        )
        self.cutoff_head = tf.keras.layers.Dense(
            1, activation="sigmoid", name="cutoff_head"
        )
        self.resonance_head = tf.keras.layers.Dense(
            1, activation="sigmoid", name="resonance_head"
        )
        self.noise_head = tf.keras.layers.Dense(
            1, activation="sigmoid", name="noise_head"
        )

        # Unison ヘッド
        # unison_voices_raw: 0〜1 → 呼び出し側で 1〜7 の整数に変換
        self.unison_voices_head = tf.keras.layers.Dense(
            1, activation="sigmoid", name="unison_voices_head"
        )
        self.detune_cents_head = tf.keras.layers.Dense(
            1, activation="sigmoid", name="detune_cents_head"
        )
        self.unison_blend_head = tf.keras.layers.Dense(
            1, activation="sigmoid", name="unison_blend_head"
        )

    def call(self, z, cond):
        """
        z    : [batch, LATENT_STEPS, latent_dim]
        cond : [batch, COND_DIM]
        """
        # z → GRU で時系列集約 → MLP
        h = self.gru(z)  # [batch, 256]
        h = self.z_mlp(h)  # [batch, 256]

        # FiLM: cond が h を変調 (condが支配的)
        h = self.film(h, cond)  # [batch, 256]
        h = tf.nn.relu(h)

        # 統合後のMLP
        feat = self.post_mlp(h)  # [batch, 128]

        # 各パラメータ (全パラメータがzとcondの両方から影響を受ける)
        # softmax: 倍音を確率分布として扱い総エネルギーを正規化する
        # sigmoid では全倍音が0.5付近→合計振幅が過大→tanh飽和→音色差が消える
        harmonic_amps = tf.nn.softmax(self.harm_head(feat))
        attack = tf.squeeze(self.attack_head(feat), -1)
        decay = tf.squeeze(self.decay_head(feat), -1)
        sustain = tf.squeeze(self.sustain_head(feat), -1)
        release = tf.squeeze(self.release_head(feat), -1)
        cutoff = tf.squeeze(self.cutoff_head(feat), -1)
        resonance = tf.squeeze(self.resonance_head(feat), -1)
        noise_amount = tf.squeeze(self.noise_head(feat), -1)

        # Unison (0〜1 の連続値として出力)
        unison_voices_raw = tf.squeeze(self.unison_voices_head(feat), -1)
        detune_cents_raw = tf.squeeze(self.detune_cents_head(feat), -1)
        unison_blend = tf.squeeze(self.unison_blend_head(feat), -1)

        return {
            "harmonic_amps": harmonic_amps,
            "attack": attack,
            "decay": decay,
            "sustain": sustain,
            "release": release,
            "cutoff": cutoff,
            "resonance": resonance,
            "noise_amount": noise_amount,
            "unison_voices_raw": unison_voices_raw,  # 0〜1 → 呼び出し側で1〜7に変換
            "detune_cents_raw": detune_cents_raw,  # 0〜1 → 呼び出し側で0〜100centに変換
            "unison_blend": unison_blend,
        }


def build_decoder(cond_dim=COND_DIM, latent_dim=LATENT_DIM):
    z_in = tf.keras.Input(shape=(LATENT_STEPS, latent_dim), name="dec_z")
    cond = tf.keras.Input(shape=(cond_dim,), name="dec_cond")
    params = DDSPParameterDecoder()(z_in, cond)
    return tf.keras.Model([z_in, cond], params, name="decoder")


# ============================================================
# TimeWiseCVAE  VAE本体
# ============================================================
class TimeWiseCVAE(tf.keras.Model):
    def __init__(
        self, cond_dim=COND_DIM, latent_dim=LATENT_DIM, steps_per_epoch=87
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.cond_dim = cond_dim

        self.timbre_embedding = TimbreEmbedding()
        self.encoder = build_encoder(latent_dim)  # condなし
        self.decoder = build_decoder(cond_dim, latent_dim)
        self.oscillator = OscillatorLayer()
        self.adsr = ADSRLayer()
        self.filter_l = FilterLayer()

        # 補助分類器: 生成パラメータから音色IDを予測できるか強制する
        # これにより Decoder が timbre_cond を無視できなくなる
        self.timbre_classifier = tf.keras.layers.Dense(
            TIMBRE_VOCAB, name="timbre_cls"
        )

        self.steps_per_epoch = steps_per_epoch
        self.kl_warmup_epochs = 20
        self.kl_rampup_epochs = 80
        self.kl_warmup_steps = self.kl_warmup_epochs * steps_per_epoch
        self.kl_rampup_steps = self.kl_rampup_epochs * steps_per_epoch
        self.kl_target = 1.0
        self.free_bits = 0.5

        self.z_std_ema = tf.Variable(1.0, trainable=False, dtype=tf.float32)
        self.best_recon = tf.Variable(
            float("inf"), trainable=False, dtype=tf.float32
        )

    # ----------------------------------------------------------
    # 内部ヘルパー
    # ----------------------------------------------------------
    def _make_cond(self, pitch, timbre_id):
        return self.timbre_embedding(pitch, timbre_id)

    def _pitch_to_f0(self, pitch):
        midi = pitch * 35.0 + 36.0
        return 440.0 * tf.pow(2.0, (midi - 69.0) / 12.0)

    def _synthesize_from_params(self, p, f0_hz, training=False):
        audio = self.oscillator(f0_hz, p["harmonic_amps"])
        env = self.adsr(p["attack"], p["decay"], p["sustain"], p["release"])
        audio = audio * env
        # FilterLayer は学習パスから除外（cutoff低値へのショートカットを防ぐ）
        batch_size = tf.shape(audio)[0]
        noise = tf.random.normal([batch_size, TIME_LENGTH]) * tf.reshape(
            p["noise_amount"], [-1, 1]
        )
        # ノイズにもエンベロープを適用: 発音時のみノイズが乗るようにする
        audio = audio + noise * env
        return tf.keras.activations.tanh(audio)

    @staticmethod
    def _safe(val, fallback=1.0):
        val = tf.where(tf.math.is_nan(val), tf.cast(fallback, val.dtype), val)
        val = tf.where(tf.math.is_inf(val), tf.cast(fallback, val.dtype), val)
        return val

    # ----------------------------------------------------------
    # call
    # ----------------------------------------------------------
    def call(self, inputs, training=None):
        if isinstance(inputs, (list, tuple)) and len(inputs) == 3:
            audio, pitch, timbre_id = inputs
        else:
            audio = inputs
            pitch = tf.zeros([tf.shape(audio)[0]], dtype=tf.float32)
            timbre_id = tf.zeros([tf.shape(audio)[0]], dtype=tf.int32)

        cond = self._make_cond(pitch, timbre_id)
        z_mean, z_logvar = self.encoder(
            [audio, cond], training=training
        )  # condあり
        z = sample_z(z_mean, z_logvar)
        p = self.decoder([z, cond], training=training)
        x_hat = self._synthesize_from_params(
            p, self._pitch_to_f0(pitch), training
        )
        return x_hat[:, :, None], z_mean, z_logvar

    # ----------------------------------------------------------
    # generate / generate_blend
    # ----------------------------------------------------------
    def generate(self, pitch, timbre_id, temperature=1.0):
        """N(0,I) サンプリングで生成。condが変わると必ず出力が変わる。"""
        cond = self._make_cond(pitch, timbre_id)
        z = (
            tf.random.normal(
                [tf.shape(pitch)[0], LATENT_STEPS, self.latent_dim]
            )
            * temperature
        )
        p = self.decoder([z, cond], training=False)
        audio = self._synthesize_from_params(p, self._pitch_to_f0(pitch))
        return audio[:, :, None]

    def generate_blend(self, pitch, timbre_weights, temperature=1.0):
        """音色ブレンドで生成。"""
        cond = self.timbre_embedding.blend(pitch, timbre_weights)
        z = (
            tf.random.normal(
                [tf.shape(pitch)[0], LATENT_STEPS, self.latent_dim]
            )
            * temperature
        )
        p = self.decoder([z, cond], training=False)
        audio = self._synthesize_from_params(p, self._pitch_to_f0(pitch))
        return audio[:, :, None]

    # ----------------------------------------------------------
    # infer_ddsp_params  ← ハード化時のメインAPI
    # ----------------------------------------------------------
    def infer_ddsp_params(
        self, pitch, timbre_id=None, timbre_weights=None, temperature=1.0
    ) -> DDSPParams:
        if timbre_weights is not None:
            cond = self.timbre_embedding.blend(pitch, timbre_weights)
        elif timbre_id is not None:
            cond = self._make_cond(pitch, timbre_id)
        else:
            raise ValueError("timbre_id か timbre_weights を指定してください")

        z = (
            tf.random.normal(
                [tf.shape(pitch)[0], LATENT_STEPS, self.latent_dim]
            )
            * temperature
        )
        p = self.decoder([z, cond], training=False)

        midi = float(pitch[0]) * 35.0 + 36.0
        f0_hz = float(440.0 * (2.0 ** ((midi - 69.0) / 12.0)))

        return DDSPParams(
            f0_hz=f0_hz,
            harmonic_amps=p["harmonic_amps"][0].numpy().tolist(),
            attack=float(p["attack"][0]),
            decay=float(p["decay"][0]),
            sustain=float(p["sustain"][0]),
            release=float(p["release"][0]),
            cutoff=float(p["cutoff"][0]),
            resonance=float(p["resonance"][0]),
            noise_amount=float(p["noise_amount"][0]),
            # Unison: 0〜1 の連続値を実際の範囲に変換
            unison_voices=int(
                round(float(p["unison_voices_raw"][0]) * 6 + 1)
            ),  # 0〜1 → 1〜7
            detune_cents=float(p["detune_cents_raw"][0])
            * 100.0,  # 0〜1 → 0〜100cent
            unison_blend=float(p["unison_blend"][0]),
        )

    # ----------------------------------------------------------
    # encode / reconstruct
    # ----------------------------------------------------------
    def encode(self, audio, pitch, timbre_id):
        cond = self._make_cond(pitch, timbre_id)
        return self.encoder([audio, cond], training=False)

    def reconstruct(self, audio, pitch, timbre_id):
        cond = self._make_cond(pitch, timbre_id)
        z_mean, z_logvar = self.encoder([audio, cond], training=False)
        z = sample_z(z_mean, z_logvar)
        p = self.decoder([z, cond], training=False)
        audio_out = self._synthesize_from_params(p, self._pitch_to_f0(pitch))
        return audio_out[:, :, None]

    # ----------------------------------------------------------
    # 音色別パラメータ補助損失
    # ----------------------------------------------------------
    def _timbre_param_loss(self, p, timbre_id, x_hat_sq):
        """
        各音色らしいパラメータ範囲・振幅形状に誘導するソフトペナルティ損失。

        モデルが3音色を共通の「平均的なパラメータ」に収束させる問題を防ぐ。
        ターゲット範囲を外れた分だけペナルティを与える (hinge損失)。

        時間的エネルギー比率 (decay_ratio = 前半1/4のRMS / 後半1/2のRMS):
            Pluck   (id=2): ratio > 5   急速減衰 = 前半にエネルギー集中
            Screech (id=0): ratio < 2   持続的ノイズ = 後半も音量を維持
            Acid    (id=1): ratio < 2.5 持続的うねり = 後半も音量を維持
        """
        screech_mask = tf.cast(tf.equal(timbre_id, 0), tf.float32)  # [batch]
        acid_mask    = tf.cast(tf.equal(timbre_id, 1), tf.float32)
        pluck_mask   = tf.cast(tf.equal(timbre_id, 2), tf.float32)

        # 時間的エネルギー比率 [batch]
        q    = TIME_LENGTH // 4
        half = TIME_LENGTH // 2
        early_rms = tf.sqrt(
            tf.reduce_mean(tf.square(x_hat_sq[:, :q]),    axis=1) + 1e-8
        )
        late_rms = tf.sqrt(
            tf.reduce_mean(tf.square(x_hat_sq[:, half:]), axis=1) + 1e-8
        )
        decay_ratio = early_rms / (late_rms + 1e-8)  # 大きい = 前半集中

        # 高次倍音の平均 (上半分) [batch]
        high_harm = tf.reduce_mean(
            p["harmonic_amps"][:, NUM_HARMONICS // 2 :], axis=1
        )

        # --- Screech ---
        screech_noise_l = tf.reduce_mean(
            screech_mask * tf.maximum(0.0, 0.3 - p["noise_amount"])
        )
        screech_harm_l = tf.reduce_mean(
            screech_mask * tf.maximum(0.0, 0.1 - high_harm)
        )
        # 前半集中しすぎ = 持続性がない → ペナルティ (ratio < 2 を促す)
        screech_temporal_l = tf.reduce_mean(
            screech_mask * tf.maximum(0.0, decay_ratio - 2.0)
        )

        # --- Acid ---
        acid_res_l = tf.reduce_mean(
            acid_mask * tf.maximum(0.0, 0.4 - p["resonance"])
        )
        # 前半集中しすぎ = うねりが途切れる → ペナルティ (ratio < 2.5 を促す)
        acid_temporal_l = tf.reduce_mean(
            acid_mask * tf.maximum(0.0, decay_ratio - 2.5)
        )

        # --- Pluck ---
        pluck_decay_l = tf.reduce_mean(
            pluck_mask * tf.maximum(0.0, p["decay"] - 0.35)
        )
        pluck_sustain_l = tf.reduce_mean(
            pluck_mask * tf.maximum(0.0, p["sustain"] - 0.3)
        )
        pluck_harm_l = tf.reduce_mean(
            pluck_mask * tf.maximum(0.0, high_harm - 0.05)
        )
        # 急速減衰が弱い = 後半も音が残る → ペナルティ (ratio > 5 を促す)
        pluck_temporal_l = tf.reduce_mean(
            pluck_mask * tf.maximum(0.0, 5.0 - decay_ratio)
        )

        return (
            screech_noise_l
            + screech_harm_l
            + screech_temporal_l
            + acid_res_l
            + acid_temporal_l
            + pluck_decay_l
            + pluck_sustain_l
            + pluck_harm_l
            + pluck_temporal_l
        )

    # ----------------------------------------------------------
    # KLスケジュール
    # ----------------------------------------------------------
    def compute_kl_weight(self):
        step = tf.cast(self.optimizer.iterations, tf.float32)
        done = tf.cast(step >= self.kl_warmup_steps, tf.float32)
        prog = tf.clip_by_value(
            (step - self.kl_warmup_steps) / self.kl_rampup_steps, 0.0, 1.0
        )
        return self.kl_target * prog * done

    def _compute_losses(self, audio, pitch, timbre_id, training):
        """train_step / test_step 共通の損失計算"""
        cond = self._make_cond(pitch, timbre_id)

        z_mean, z_logvar = self.encoder([audio, cond], training=training)
        z_logvar = tf.clip_by_value(z_logvar, -8.0, 2.0)
        z = tf.clip_by_value(sample_z(z_mean, z_logvar), -10.0, 10.0)

        p = self.decoder([z, cond], training=training)

        f0_hz = self._pitch_to_f0(pitch)
        x_hat_audio = self._synthesize_from_params(p, f0_hz, training)[
            :, :TIME_LENGTH
        ]

        x_target = tf.clip_by_value(tf.squeeze(audio, axis=-1), -1.0, 1.0)
        x_hat_sq = tf.clip_by_value(x_hat_audio, -1.0, 1.0)

        s = self._safe
        recon = s(tf.reduce_mean(tf.square(x_target - x_hat_sq)))
        stft_l, mel_l, _ = Loss(x_target, x_hat_sq, fft_size=2048, hop_size=512)
        stft_l = s(stft_l)
        mel_l = s(mel_l)

        kl_per_dim = -0.5 * (
            1.0 + z_logvar - tf.square(z_mean) - tf.exp(z_logvar)
        )
        kl_mean = tf.reduce_mean(kl_per_dim, axis=[0, 1])
        kl = s(
            tf.clip_by_value(
                tf.reduce_mean(tf.maximum(kl_mean, self.free_bits)), 0.0, 100.0
            ),
            0.5,
        )

        # 振幅損失: サンプル単位のRMSを計算し音色別ターゲットを設定
        # Pluck は急速減衰が特徴なのでターゲットを低めにする
        audio_rms_per = tf.sqrt(
            tf.reduce_mean(tf.square(x_hat_sq), axis=1) + 1e-8
        )  # [batch]
        rms_target = tf.where(
            tf.equal(timbre_id, 2),          # Pluck
            tf.constant(0.05, dtype=tf.float32),
            tf.constant(0.2, dtype=tf.float32),   # Screech / Acid
        )
        amp_l = s(tf.reduce_mean(tf.maximum(0.0, rms_target - audio_rms_per)))

        # 音色別パラメータ補助損失
        timbre_l = s(self._timbre_param_loss(p, timbre_id, x_hat_sq))

        # 補助分類損失: 生成パラメータから音色IDを正しく予測できるか
        # Decoder が timbre_cond を無視すると全音色で同じパラメータが出て
        # この損失が大きくなるため、cond を活用するよう強制する
        param_feat = tf.concat(
            [
                p["harmonic_amps"],          # [batch, NUM_HARMONICS] 倍音分布
                p["noise_amount"][:, None],  # [batch, 1]
                p["resonance"][:, None],     # [batch, 1]
                p["decay"][:, None],         # [batch, 1]
                p["sustain"][:, None],       # [batch, 1]
            ],
            axis=-1,
        )
        cls_logits = self.timbre_classifier(param_feat)  # [batch, TIMBRE_VOCAB]
        cls_l = s(
            tf.reduce_mean(
                tf.keras.losses.sparse_categorical_crossentropy(
                    timbre_id, cls_logits, from_logits=True
                )
            )
        )

        kl_w = self.compute_kl_weight()
        loss = (
            recon * 5.0
            + stft_l * 3.0
            + mel_l * 4.0
            + amp_l * 5.0
            + timbre_l * 3.0
            + cls_l * 2.0
            + kl * kl_w
        )
        loss = s(loss, 1000.0)

        return loss, recon, stft_l, mel_l, amp_l, timbre_l, cls_l, kl, kl_w, z_mean

    # ----------------------------------------------------------
    # train_step
    # ----------------------------------------------------------
    def train_step(self, data):
        (audio, pitch, timbre_id), _ = data

        with tf.GradientTape() as tape:
            loss, recon, stft_l, mel_l, amp_l, timbre_l, cls_l, kl, kl_w, z_mean = (
                self._compute_losses(audio, pitch, timbre_id, training=True)
            )

        grads = tape.gradient(loss, self.trainable_variables)
        grads = [
            (
                tf.where(tf.math.is_nan(g), tf.zeros_like(g), g)
                if g is not None
                else None
            )
            for g in grads
        ]
        grads = [
            (
                tf.where(tf.math.is_inf(g), tf.zeros_like(g), g)
                if g is not None
                else None
            )
            for g in grads
        ]
        grads, grad_norm = tf.clip_by_global_norm(grads, 5.0)
        grad_norm = tf.where(tf.math.is_nan(grad_norm), 0.0, grad_norm)
        grad_norm = tf.where(tf.math.is_inf(grad_norm), 0.0, grad_norm)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))

        s = self._safe
        z_std = s(tf.reduce_mean(tf.math.reduce_std(z_mean, axis=1)), 1.0)
        self.z_std_ema.assign(0.99 * self.z_std_ema + 0.01 * z_std)
        self.best_recon.assign(tf.minimum(self.best_recon, recon))

        return {
            "loss": loss,
            "recon": recon,
            "best_recon": self.best_recon,
            "stft": stft_l,
            "mel": mel_l,
            "amp": amp_l,
            "timbre": timbre_l,
            "cls": cls_l,
            "kl": kl,
            "kl_weight": kl_w,
            "z_std_ema": self.z_std_ema,
            "grad_norm": grad_norm,
        }

    # ----------------------------------------------------------
    # test_step
    # ----------------------------------------------------------
    def test_step(self, data):
        (audio, pitch, timbre_id), _ = data
        loss, recon, stft_l, mel_l, amp_l, timbre_l, cls_l, kl, kl_w, _ = (
            self._compute_losses(audio, pitch, timbre_id, training=False)
        )
        return {
            "loss": loss,
            "recon": recon,
            "stft": stft_l,
            "mel": mel_l,
            "amp": amp_l,
            "timbre": timbre_l,
            "cls": cls_l,
            "kl": kl,
            "kl_weight": kl_w,
        }
