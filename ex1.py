"""Спектральный анализ, эквализация и подавление тональной помехи в WAV."""

import argparse
from pathlib import Path

import matplotlib
import numpy as np
from scipy import signal
from scipy.io import wavfile


def read_audio(path):
    sample_rate, samples = wavfile.read(path)
    if len(samples) < 2:
        raise ValueError("WAV должен содержать хотя бы два отсчёта.")

    dtype = samples.dtype
    audio = samples.astype(np.float64)
    if np.issubdtype(dtype, np.unsignedinteger):
        midpoint = (np.iinfo(dtype).max + 1) / 2
        audio = (audio - midpoint) / midpoint
    elif np.issubdtype(dtype, np.signedinteger):
        audio /= -np.iinfo(dtype).min

    if not np.isfinite(audio).all():
        raise ValueError("В WAV есть бесконечные или неопределённые отсчёты.")
    return sample_rate, audio


def equalize(audio, sample_rate, bands):
    if not bands:
        return audio

    frame = min(4096, len(audio))
    settings = dict(fs=sample_rate, window="hann", nperseg=frame, noverlap=frame // 2)
    frequencies, _, spectrum = signal.stft(audio.T, **settings)
    total_gain_db = np.zeros_like(frequencies)
    bin_width = sample_rate / frame

    for low, high, gain_db in bands:
        # Плавные переходы на границах полос уменьшают звон фильтра.
        transition = min(2 * bin_width, (high - low) / 2)
        weight = np.ones_like(frequencies)
        if low > 0:
            rising = np.clip((frequencies - low) / transition, 0, 1)
            weight *= 0.5 - 0.5 * np.cos(np.pi * rising)
        if high < sample_rate / 2:
            falling = np.clip((high - frequencies) / transition, 0, 1)
            weight *= 0.5 - 0.5 * np.cos(np.pi * falling)
        if not np.any(weight > 0):
            raise ValueError(f"Полоса {low:g}–{high:g} Гц слишком узкая для этого WAV.")
        total_gain_db += gain_db * weight

    spectrum *= 10 ** (total_gain_db[:, None] / 20)
    _, reconstructed = signal.istft(spectrum, **settings)
    return reconstructed[..., :len(audio)].T


def remove_tone(audio, sample_rate, frequency, quality):
    # Два прохода режекторного фильтра дают нулевой фазовый сдвиг.
    numerator, denominator = signal.iirnotch(frequency, quality, fs=sample_rate)
    sos = signal.tf2sos(numerator, denominator)
    pad_length = min(len(audio) - 1, 9)
    return signal.sosfiltfilt(sos, audio, axis=0, padlen=pad_length)


def save_audio(path, sample_rate, audio):
    peak = np.max(np.abs(audio))
    if peak > 1:
        audio = audio / peak
        print(f"Чтобы избежать клиппинга, общий уровень снижен на {20 * np.log10(peak):.2f} дБ.")

    scaled = np.clip(np.rint(audio * 32768), -32768, 32767)
    pcm = scaled.astype(np.int16)
    path.parent.mkdir(parents=True, exist_ok=True)
    wavfile.write(path, sample_rate, pcm)
    return pcm.astype(np.float64) / 32768


def power_db(power):
    return 10 * np.log10(np.maximum(power, 1e-16))


def channel_mean(power):
    # Усредняем мощности, чтобы каналы с разной фазой не погасили друг друга.
    return power.mean(axis=0)

def save_plots(original, processed, sample_rate, output, tone, show):
    if not show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    frame = min(4096, len(original))
    rows = (
        ("Исходный сигнал", np.atleast_2d(original.T)),
        ("Обработанный сигнал", np.atleast_2d(processed.T)),
    )
    spectrum_figure, spectrum_axes = plt.subplots(figsize=(10, 4), layout="constrained")
    for label, audio in rows:
        frequencies, density = signal.welch(audio, fs=sample_rate, nperseg=frame)
        spectrum_axes.plot(frequencies / 1000, power_db(channel_mean(density)), label=label)
    spectrum_axes.axvline(tone / 1000, color="0.4", linestyle="--", label=f"Помеха {tone:g} Гц")
    spectrum_axes.set(xlabel="Частота, кГц", ylabel="СПМ, дБ/Гц", title="Спектр сигнала",
                      xlim=(0, sample_rate / 2000))
    spectrum_axes.grid(alpha=0.25)
    spectrum_axes.legend()
    spectrum_figure.savefig(output.with_name(output.stem + "_spectrum.png"), dpi=160)

    panels = []
    for label, audio in rows:
        frequencies, times, density = signal.spectrogram(
            audio, fs=sample_rate, window="hann", nperseg=frame, noverlap=frame // 2,
        )
        panels.append((label, times, frequencies, power_db(channel_mean(density))))

    vmax = max(-70, max(panel[3].max() for panel in panels))
    figure, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True, layout="constrained")
    hop_time = (frame - frame // 2) / sample_rate
    half_bin_khz = sample_rate / frame / 2000
    for ax, (label, times, frequencies, density_db) in zip(axes, panels):
        extent = (times[0] - hop_time / 2, times[-1] + hop_time / 2,
                  -half_bin_khz, frequencies[-1] / 1000 + half_bin_khz)
        mesh = ax.imshow(density_db, origin="lower", aspect="auto", extent=extent,
                         interpolation="nearest", cmap="magma", vmin=vmax - 85, vmax=vmax)
        ax.set(title=label, ylabel="Частота, кГц", ylim=(0, sample_rate / 2000))
    axes[-1].set_xlabel("Время, с")
    figure.colorbar(mesh, ax=axes, label="СПМ, дБ/Гц")
    figure.savefig(output.with_name(output.stem + "_spectrogram.png"), dpi=160)

    if show:
        plt.show()
    plt.close("all")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="Исходный WAV")
    parser.add_argument("-o", "--output", type=Path, default=Path("processed.wav"),
                        help="Результирующий WAV (по умолчанию processed.wav)")
    parser.add_argument("--tone", type=float, default=14000,
                        help="Частота тональной помехи в Гц (по умолчанию 14000)")
    parser.add_argument("--quality", type=float, default=70,
                        help="Добротность режекторного фильтра (по умолчанию 70)")
    parser.add_argument("--band", type=float, nargs=3, action="append", default=[],
                        metavar=("LOW_HZ", "HIGH_HZ", "GAIN_DB"),
                        help="Границы полосы в Гц и усиление в дБ. Можно повторять.")
    parser.add_argument("--show", action="store_true", help="Открыть окна с графиками")
    args = parser.parse_args()

    if args.input.resolve() == args.output.resolve():
        parser.error("Имена исходного и результирующего WAV должны различаться.")
    if args.output.suffix.lower() != ".wav":
        parser.error("Имя выходного файла должно заканчиваться на .wav.")

    try:
        sample_rate, original = read_audio(args.input)
        nyquist = sample_rate / 2
        if not 0 < args.tone < nyquist:
            raise ValueError(f"Частота помехи должна быть между 0 и {nyquist:g} Гц.")
        if not np.isfinite(args.quality) or args.quality < 1:
            raise ValueError("Добротность должна быть конечным числом не меньше 1.")
        for low, high, gain in args.band:
            if not 0 <= low < high <= nyquist or not -100 <= gain <= 100:
                raise ValueError(f"Нужно 0 ≤ LOW_HZ < HIGH_HZ ≤ {nyquist:g} и усиление от −100 до 100 дБ.")

        equalized = equalize(original, sample_rate, args.band)
        filtered = remove_tone(equalized, sample_rate, args.tone, args.quality)
        processed = save_audio(args.output, sample_rate, filtered)
        save_plots(original, processed, sample_rate, args.output, args.tone, args.show)
    except (OSError, ValueError) as error:
        parser.error(str(error))

    print(f"WAV и два графика сохранены. Имя WAV — {args.output}")


if __name__ == "__main__":
    main()