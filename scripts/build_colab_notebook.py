"""Generate the baseline notebook with incremental local browser downloads."""

import json
from pathlib import Path


def markdown(text):
    return {"cell_type": "markdown", "metadata": {}, "source": text.strip().splitlines(keepends=True)}


def code(text):
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
            "source": text.strip().splitlines(keepends=True)}


def main():
    cells = [
        markdown("""
# Baseline DDIM на T4 с постепенным скачиванием результатов

Используем замороженную U-Net google/ddpm-cifar10-32. Получаем 5000 изображений,
100 NFE на изображение, время и FID относительно 50000 CIFAR-10 train.
Это первый ноутбук для РК 1. Собственные сэмплеры — в ноутбуке 02.

В Colab выберите **Среда выполнения → Сменить среду выполнения → T4 GPU**.
Используйте новую сессию; после смены версии исходников перезапустите среду.
Выполняйте ячейки по порядку. Сначала pilot и один полный батч, затем длинный запуск.

Архивы будут скачиваться через браузер на ваш компьютер, обычно в папку «Загрузки».
Разрешите несколько скачиваний для Colab, если браузер запросит это. Python может
запросить скачивание. Проверяйте сохранение через список загрузок браузера.
Рабочая папка /content временная, поэтому сохраняйте промежуточные ZIP на компьютер.
В конце есть ячейка повторного скачивания выбранного архива.
"""),
        code("""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import torch
print('PyTorch:', torch.__version__)
assert torch.cuda.is_available(), 'Выберите среду выполнения с GPU T4'
GPU_NAME = torch.cuda.get_device_name(0)
print('GPU:', GPU_NAME)
print('VRAM, GiB:', torch.cuda.get_device_properties(0).total_memory / 2**30)
assert 'T4' in GPU_NAME, 'Для этого эксперимента выберите T4, чтобы не смешивать устройства'
"""),
        markdown("""
## Загрузить архив с компьютера в Colab

Нажмите «Выбрать файлы» и выберите архив на компьютере. Его исходный путь на Mac
`/Users/maria.ignatova/mipt-hw/lm-in-cv-mipt/dist/diffusion-sampling-colab.zip`.
Если вы скачали его из чата, выберите файл из папки «Загрузки».

В Colab архив сохраняется как /content/diffusion-sampling-colab.zip, проект
распаковывается в /content/diffusion-sampling. Рабочие данные и ZIP хранятся
в этой папке до завершения сессии. Сохраняйте скачанные архивы на компьютере.
"""),
        code("""
from google.colab import files
from pathlib import Path
from zipfile import ZipFile
uploaded = files.upload()
assert len(uploaded) == 1, 'Выберите один ZIP с исходниками проекта'
uploaded_name, archive_bytes = next(iter(uploaded.items()))
SOURCE_BUNDLE = Path('/content/diffusion-sampling-colab.zip')
SOURCE_BUNDLE.write_bytes(archive_bytes)
with ZipFile(SOURCE_BUNDLE) as bundle:
    for member in bundle.infolist():
        assert (Path('/content') / member.filename).resolve().is_relative_to(Path('/content'))
    bundle.extractall('/content')
PROJECT = Path('/content/diffusion-sampling')
assert (PROJECT / 'pyproject.toml').is_file(), 'В ZIP должна быть папка diffusion-sampling с pyproject.toml'
for name in ['data', 'results', '.cache']:
    (PROJECT / name).mkdir(exist_ok=True)
print('Выбран файл', uploaded_name)
print('Архив в Colab', SOURCE_BUNDLE)
print('Рабочая папка', PROJECT)
"""),
        code("""
%pip install -q -e /content/diffusion-sampling
%cd /content/diffusion-sampling
import sys
if str(PROJECT / 'src') not in sys.path:
    sys.path.insert(0, str(PROJECT / 'src'))
"""),
        markdown("""
## Параметры запуска

Оставьте batch size 64, float32, eta=0, native leading и clipping=True.
Seed первого изображения — 50000, дальше отдельный seed на изображение.
Скачаем pilot на 16 изображениях, первый батч baseline на 64 изображениях,
затем новые батчи на отметках 1024, 2048, 3072, 4096 и 5000 изображений.

Если GPU-памяти не хватает, до нового запуска измените BATCH_SIZE на 32 и RUN_TAG,
например добавьте _bs32. Уже начатый запуск не продолжайте с другими параметрами.
После обновления исходников тоже используйте новый RUN_TAG. Для продолжения
выполнения в той же сессии сохраните те же исходники, RUN_TAG и параметры.
"""),
        code("""
from dataclasses import replace
import json
from diffusion_sampling.config import load_config
from diffusion_sampling.storage import write_json
from diffusion_sampling.exports import CheckpointExporter, export_noise_banks

RUN_TAG = 't4_ddim100_baseline_v1'
BATCH_SIZE = 64
DOWNLOAD_EVERY_BATCHES = 16
AUTO_DOWNLOAD = True

baseline_config = replace(load_config('configs/baseline_ddim_100.json'),
    device='cuda', dtype='float32', batch_size=BATCH_SIZE,
    output_dir=f'results/{RUN_TAG}')
pilot_config = replace(load_config('configs/pilot.json'), device='cuda',
    output_dir=f'results/{RUN_TAG}_pilot')
BASELINE_DIR = PROJECT / baseline_config.output_dir
PILOT_DIR = PROJECT / pilot_config.output_dir
EXPORT_DIR = PROJECT / 'results' / 'exports' / RUN_TAG
EXPORT_DIR.mkdir(parents=True, exist_ok=True)
write_json(PROJECT / 'configs/t4_baseline_runtime.json', baseline_config.to_dict())
write_json(PROJECT / 'configs/t4_pilot_runtime.json', pilot_config.to_dict())
exporter = CheckpointExporter(PROJECT, EXPORT_DIR, source_bundle=SOURCE_BUNDLE)
print(json.dumps(baseline_config.to_dict(), indent=2, ensure_ascii=False))
print('Параметр eta зафиксирован в реализации:', 0)
"""),
        code("""
import subprocess, sys, shutil, gc
from contextlib import redirect_stdout, redirect_stderr
from IPython.display import Image, display
from uuid import uuid4
from diffusion_sampling.experiment import run_experiment

def cli(*args):
    command = [sys.executable, '-m', 'diffusion_sampling', *args]
    with (EXPORT_DIR / 'setup.log').open('a') as log:
        with subprocess.Popen(command, text=True, stdout=subprocess.PIPE,
                              stderr=subprocess.STDOUT, bufsize=1) as process:
            for line in process.stdout:
                print(line, end='', flush=True)
                log.write(line)
                log.flush()
            returncode = process.wait()
    if returncode:
        raise subprocess.CalledProcessError(returncode, command)

def download_archive(path):
    if path is None:
        return
    path = Path(path)
    print(f'Архив готов: {path.name} ({path.stat().st_size / 2**20:.1f} MiB)')
    print('Архив в рабочей папке Colab', path)
    if AUTO_DOWNLOAD:
        try:
            files.download(str(path))
            print('Запрос скачивания отправлен браузеру. Проверьте папку загрузок.')
        except Exception as error:
            print('Не удалось запросить скачивание:', error)
            print('Генерация продолжится; архив можно скачать повторно в последней ячейке.')

class Tee:
    def __init__(self, first, second):
        self.first, self.second = first, second
    def write(self, text):
        self.first.write(text)
        self.second.write(text)
        return len(text)
    def flush(self):
        self.first.flush()
        self.second.flush()
    def __getattr__(self, name):
        return getattr(self.first, name)

def copy_generation_log(run_dir):
    path = EXPORT_DIR / f'{Path(run_dir).name}.log'
    if path.exists() and (Path(run_dir) / 'manifest.json').exists():
        shutil.copyfile(path, Path(run_dir) / 'generation.log')

def checkpoint(run_dir, manifest, shards, summary):
    copy_generation_log(run_dir)
    download_archive(exporter.delta(run_dir, manifest, shards, summary))

def run_logged(config, **kwargs):
    run_dir = PROJECT / config.output_dir
    with (EXPORT_DIR / f'{run_dir.name}.log').open('a') as log:
        try:
            with redirect_stdout(Tee(sys.stdout, log)), redirect_stderr(Tee(sys.stderr, log)):
                return run_experiment(config, root=PROJECT, **kwargs)
        finally:
            log.flush()
            copy_generation_log(run_dir)

cli('doctor')
tests = subprocess.run([sys.executable, '-m', 'unittest', 'discover', '-s', 'tests', '-v'],
                       text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
(EXPORT_DIR / 'tests.log').write_text(tests.stdout)
print(tests.stdout)
tests.check_returncode()
"""),
        markdown("""
## Сохранить начальные шумы на компьютер

Шумы создаются на CPU и сохраняются как тензоры. Перед длинным запуском скачиваем
точные банки debug и fid вместе с исходниками. Это нужно для воспроизводимости,
и последующего воспроизведения. Архив содержит выбранные начальные шумы и исходники.
"""),
        code("""
cli('prepare-noise', '--fid-count', '5000')
noise_archive = export_noise_banks(PROJECT,
    EXPORT_DIR / f'initial_noises_{RUN_TAG}_{uuid4().hex[:8]}.zip',
    [pilot_config.noise_path, baseline_config.noise_path], source_bundle=SOURCE_BUNDLE)
download_archive(noise_archive)
"""),
        markdown("""
## Pilot и его первый локальный архив

16 изображений, 20 NFE, batch size 8. Посмотрите сетку: это должна быть генерация
объектов CIFAR-10, а не сплошной шум или одинаковые пустые картинки.
Pilot проверяет загрузку модели, генерацию и экспорт. Основной baseline вычисляем дальше.
"""),
        code("""
pilot_summary = run_logged(pilot_config)
display(Image(filename=str(PILOT_DIR / 'preview.png')))
download_archive(exporter.full(PILOT_DIR, stage='pilot'))
"""),
        markdown("""
## Первый полный батч baseline

Запускаем только 64 изображения при 100 NFE. Скачиваем их, метрики и настройки
сразу после батча. Оценка оставшегося времени учитывает только денойзинг;
загрузка, FID, архивирование и скачивание добавят время отдельно.
"""),
        code("""
summary = run_logged(baseline_config, max_batches=1)
summary_path = BASELINE_DIR / 'summary.json'
print(json.dumps(summary, indent=2, ensure_ascii=False))
remaining = summary['requested_images'] - summary['generated_images']
print('Примерно осталось часов денойзинга:', remaining * summary['seconds_per_image'] / 3600)
display(Image(filename=str(BASELINE_DIR / 'preview.png')))
download_archive(exporter.delta(BASELINE_DIR))
"""),
        markdown("""
## Продолжить до 5000 с промежуточными скачиваниями

При batch size 64 скачивания запрашиваются на отметках 1024, 2048, 3072, 4096,
5000. Каждый delta ZIP содержит новые батчи, PNG, актуальные метрики и preview;
начальные шумы уже скачаны отдельно. Индекс позволяет продолжать архивирование
с последнего сохранённого диапазона в текущей сессии. Через последнюю ячейку
можно повторно скачать ZIP из рабочей папки Colab.

Архивация выполняется между батчами и не включается во время денойзинга.
Если экспорт завершится ошибкой, уже сгенерированные батчи останутся сохранены.
"""),
        code("""
summary = run_logged(baseline_config, on_checkpoint=checkpoint,
                     checkpoint_every_batches=DOWNLOAD_EVERY_BATCHES)
assert summary['generation_complete'] and summary['generated_images'] == 5000
assert summary['nfe_per_image'] == 100
print('Генерация завершена. Следующий шаг — FID.')
"""),
        markdown("""
## FID и финальные результаты

FID: torch-fidelity 0.3.0, inception-v3-compat, 2048 признаков. Сравниваем
5000 generated с 50000 CIFAR-10 train по протоколу FID-5k/real-50k.
Сохраняем измеренное значение FID в отчёт.

Освобождаем ненужную GPU-память перед отдельным процессом оценки.
"""),
        code("""
gc.collect()
torch.cuda.empty_cache()
cli('evaluate', '--run', str(BASELINE_DIR), '--device', 'cuda', '--download')
summary = json.loads(summary_path.read_text())
fid_report = json.loads((BASELINE_DIR / 'fid.json').read_text())
assert summary['generation_complete'] and summary['generated_images'] == 5000
assert summary['nfe_per_image'] == 100
assert summary['sample_equivalent_model_evaluations'] == 500000
assert summary['baseline_metrics_complete']
print('FID:', summary['fid'])
print('NFE на изображение:', summary['nfe_per_image'])
print('Время денойзинга, с:', summary['generation_seconds'])
print('Секунд на изображение:', summary['seconds_per_image'])
print('Размеры выборок:', fid_report['generated_count'], fid_report['reference']['count'])
display(Image(filename=str(BASELINE_DIR / 'preview.png')))
"""),
        markdown("""
## Скачать полный финальный архив

Он содержит все 5000 PNG, сырые тензоры, точный начальный шум, FID, время/NFE,
конфигурацию, версии, ревизию модели, preview, логи и исходники. Для анализа достаточно
этого ZIP. Промежуточные архивы служат резервной копией по ходу выполнения.

Можно записать наблюдения о картинках в VISUAL_NOTES. Пустая строка означает,
что визуальное описание ещё предстоит сделать; скачивание от этого не блокируется.
"""),
        code("""
VISUAL_NOTES = ''  # Например: какие объекты видны, есть ли размытие и артефакты.
if VISUAL_NOTES.strip():
    (BASELINE_DIR / 'visual_review.md').write_text(VISUAL_NOTES.strip() + '\\n')
logs = BASELINE_DIR / 'logs'
logs.mkdir(exist_ok=True)
for name in ['setup.log', 'tests.log']:
    if (EXPORT_DIR / name).exists():
        shutil.copyfile(EXPORT_DIR / name, logs / name)
final_archive = exporter.full(BASELINE_DIR, stage='final_fid')
download_archive(final_archive)
print('Финальный архив:', final_archive.name)
"""),
        markdown("""
## Повторно скачать любой архив

Поставьте REPEAT_DOWNLOAD=True, если скачивание было заблокировано или прервалось.
По умолчанию выбирается последний созданный ZIP. Для другого файла впишите
его имя в ARCHIVE_NAME. Сохраните промежуточные ZIP до проверки финального.
"""),
        code("""
archives = sorted(EXPORT_DIR.glob('*.zip'), key=lambda path: path.stat().st_mtime)
for path in archives:
    print(path.name, f'{path.stat().st_size / 2**20:.1f} MiB')
ARCHIVE_NAME = ''  # Имя из списка; пустая строка — последний архив.
REPEAT_DOWNLOAD = False  # Измените на True только для повторного скачивания.
if REPEAT_DOWNLOAD and archives:
    chosen = EXPORT_DIR / ARCHIVE_NAME if ARCHIVE_NAME else archives[-1]
    files.download(str(chosen))
"""),
        markdown("""
## Что получено для РК 1

Численные FID, NFE=100 и время для 5000 изображений, воспроизводимые настройки,
исходные шумы, сырые результаты и визуальная сетка. Опишите качество по preview.
Ключевой итог — *_full_final_fid_*.zip в локальной папке загрузок.

Следующим запускаем 02_samplers_colab.ipynb для сравнения собственных методов
при равном NFE. Для сопоставимости сохраняем общий протокол clipping и расписания.
"""),
    ]
    for index, cell in enumerate(cells):
        cell["id"] = f"baseline-{index:02d}"
    notebook = {"nbformat": 4, "nbformat_minor": 5, "cells": cells, "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"}, "colab": {"name": "01_baseline_colab.ipynb"},
        "accelerator": "GPU",
    }}
    output = Path(__file__).resolve().parents[1] / "notebooks" / "01_baseline_colab.ipynb"
    output.parent.mkdir(exist_ok=True)
    output.write_text(json.dumps(notebook, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
