"""Build the second Colab notebook for explicit equal-NFE comparisons."""

import json
from pathlib import Path

from build_colab_notebook import code, markdown


def main():
    cells = [
        markdown("""
# Собственные сэмплеры и одинаковый бюджет вычислений

Модель одна: google/ddpm-cifar10-32, веса заморожены. Реализованы DDIM,
DPM-Solver++ 2M второго порядка и Heun в EDM-координатах. Здесь проверяем их,
получаем короткое сравнение и готовим конфигурации для FID на T4.

**Problem statement.** Как меняется качество при одинаковом числе вызовов сети,
если менять численный метод и расположение уровней шума?

Ноутбук 01 отдельно воспроизводит библиотечный baseline. Здесь общий диапазон
t=999…0 и clipping отключён. Не объединяйте эти результаты с другим протоколом.
16 изображений достаточно для проверки, но недостаточно для выводов о FID.
"""),
        code("""
import torch
assert torch.cuda.is_available(), 'Выберите среду выполнения T4 GPU'
print(torch.cuda.get_device_name(0), torch.__version__)
"""),
        markdown("""
## Подготовить исходники

Загрузите свежий diffusion-sampling-colab.zip. Шумы, веса и результаты сохраняются
в Drive. Если исходники обновлены, старые измерения не продолжаем: задаём новые
output_dir в конфигурациях. Исходный архив собирается scripts/build_colab_bundle.py.
"""),
        code("""
from google.colab import drive, files
from pathlib import Path
from zipfile import ZipFile
drive.mount('/content/drive')
uploaded = files.upload()
assert 'diffusion-sampling-colab.zip' in uploaded
with ZipFile('diffusion-sampling-colab.zip') as bundle:
    for member in bundle.infolist():
        assert (Path('/content') / member.filename).resolve().is_relative_to(Path('/content'))
    bundle.extractall('/content')
PROJECT = Path('/content/diffusion-sampling')
PERSISTENT = Path('/content/drive/MyDrive/llm-in-cv/diffusion-sampling')
PERSISTENT.mkdir(parents=True, exist_ok=True)
for name in ['data', 'results', '.cache']:
    target = PERSISTENT / name
    target.mkdir(exist_ok=True)
    if not (PROJECT / name).exists():
        (PROJECT / name).symlink_to(target, target_is_directory=True)
"""),
        code("""
%pip install -q -e /content/diffusion-sampling
%cd /content/diffusion-sampling
import subprocess, sys, json
def cli(*args):
    subprocess.run([sys.executable, '-m', 'diffusion_sampling', *args], check=True)
cli('doctor')
subprocess.run([sys.executable, '-m', 'unittest', 'discover', '-s', 'tests', '-v'], check=True)
cli('prepare-noise')
"""),
        markdown("""
## Три метода при 19 NFE

DDIM и DPM-Solver++: 19 точек. Heun: 10 точек, 19 forward.
Все используют одни шумы и одинаковый диапазон уровней. Начальное VE-состояние
Heun специально соответствует тому же VP-состоянию z, что и у остальных.
Модель скачается при первом запуске; повторные команды используют кэш.
"""),
        code("""
for config in ['pilot_ddim_19.json', 'pilot_dpmpp_19.json', 'pilot_heun_19.json']:
    cli('run', '--config', f'configs/{config}')
"""),
        markdown("""
## Проверить совпадение с библиотекой на настоящей модели

Сравниваются все завершённые шаги и фактический NFE. Это не замер скорости:
сохранение промежуточных состояний и проверки добавляют синхронизацию.
"""),
        code("""
subprocess.run([sys.executable, 'scripts/verify_samplers.py', '--device', 'cuda', '--local-files-only'], check=True)
"""),
        markdown("""
## Сравнить время и картинки

Команда откажется объединять разные веса, шумы, точность, clipping, диапазоны,
окружения и batch size. FID пока отсутствует. В колонках сетки — одинаковый шум.
Не выбирайте лучший метод по 16 картинкам: это только pilot.
"""),
        code("""
runs = [f'results/sampler_pilots/{name}_uniform_t_nfe19' for name in ['ddim', 'dpmpp_2m', 'heun']]
cli('compare', '--runs', *runs, '--output', 'results/sampler_pilots/comparison.json')
from IPython.display import Image, display
display(Image(filename='results/sampler_pilots/comparison.png'))
"""),
        markdown("""
## Получить длинный численный эталон и MSE

500-шаговый собственный DDIM с теми же шумами и крайними точками. Считать его
достаточно точным можно только после проверки сходимости с большим бюджетом.
MSE считается по float32-выходам до округления в PNG. Эталон не является ground truth.
"""),
        code("""
cli('run', '--config', 'configs/reference_ddim_500_debug.json')
cli('compare', '--runs', *runs, '--reference', 'results/sampler_pilots/reference_ddim_500',
    '--output', 'results/sampler_pilots/comparison_with_reference.json')
report = json.loads(Path('results/sampler_pilots/comparison_with_reference.json').read_text())
for row in report['rows']:
    print(row['sampler'], 'NFE=', row['nfe'], 'MSE=', row['mse_to_reference'], 'FID=', row['fid'])
"""),
        markdown("""
## Дополнительная проверка точности эталона

Эта генерация дороже предыдущей: сравниваем 500 и 1000 NFE на тех же 16 шумах.
Небольшая ошибка 500 относительно 1000 поддерживает выбор эталона для этого
pilot, но не доказывает сходимость на всей выборке. Для следующего исследования
потребуется проверить независимые шумы.
"""),
        code("""
cli('run', '--config', 'configs/reference_ddim_1000_debug.json')
cli('compare', '--runs', 'results/sampler_pilots/reference_ddim_500',
    '--reference', 'results/sampler_pilots/reference_ddim_1000',
    '--output', 'results/sampler_pilots/reference_convergence.json')
cli('compare', '--runs', *runs, '--reference', 'results/sampler_pilots/reference_ddim_1000',
    '--output', 'results/sampler_pilots/comparison_with_reference.json')
"""),
        markdown("""
## Подготовить полный эксперимент с FID

Следующая ячейка только создаёт конфигурации: три метода, один бюджет 39 NFE,
одно uniform_t-расписание, по 5000 изображений. Сначала измерьте один батч
и оцените оставшееся время. Для 27 конфигураций задайте три бюджета и три расписания,
но полный перебор стоит запускать после оценки ресурсов.
"""),
        code("""
cli('make-sweep', '--output', 'configs/fid_study', '--name', 'fid_study',
    '--budgets', '39', '--schedules', 'uniform_t', '--purpose', 'baseline',
    '--num-samples', '5000', '--batch-size', '64', '--device', 'cuda')
"""),
        markdown("""
## Один выбранный полный запуск

Выполните последовательно для каждого выбранного метода. Имя config_to_run
изменяется явно. evaluate скачает CIFAR-10 и Inception, если их ещё нет.
Сравнивайте FID только после завершения одинаковых выборок для всех методов.
"""),
        code("""
config_to_run = 'configs/fid_study/ddim_uniform_t_nfe39.json'
cli('run', '--config', config_to_run, '--max-batches', '1')
config = json.loads(Path(config_to_run).read_text())
summary = json.loads((Path(config['output_dir']) / 'summary.json').read_text())
print('Осталось часов денойзинга примерно:',
      (summary['requested_images'] - summary['generated_images']) * summary['seconds_per_image'] / 3600)
"""),
        code("""
cli('run', '--config', config_to_run)
cli('evaluate', '--run', config['output_dir'], '--device', 'cuda', '--download')
"""),
        markdown("""
Результат РК 1: baseline из первого ноутбука с FID/NFE/временем, проверенная обвязка,
теоретическое описание трёх методов и конкретный план сравнения. Исследовательский
этап: оси NFE/расписание, длинный эталон, подбор опорных точек на calibration,
проверка на validation и независимых шумах, затем FID и перенос на другой сэмплер.
Подбор расписания ещё предстоит реализовать; этот ноутбук не выдаёт готовую оптимизацию.
"""),
    ]
    for index, cell in enumerate(cells):
        cell["id"] = f"samplers-{index:02d}"
    notebook = {"nbformat": 4, "nbformat_minor": 5, "cells": cells, "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"}, "colab": {"name": "02_samplers_colab.ipynb"}, "accelerator": "GPU",
    }}
    output = Path(__file__).resolve().parents[1] / "notebooks" / "02_samplers_colab.ipynb"
    output.write_text(json.dumps(notebook, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
