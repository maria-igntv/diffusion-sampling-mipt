"""Build a standalone Colab study that starts from a downloaded baseline ZIP."""

import json
from pathlib import Path

from build_colab_notebook import code, markdown


def main():
    root = Path(__file__).resolve().parents[1]
    cells = [
        markdown("""
# Сравнение сэмплеров после восстановления baseline

## Тема и problem statement

Исследуем DDIM, DPM-Solver++ 2M и Heun на замороженной U-Net
google/ddpm-cifar10-32. Как различаются FID, время и изображения при одинаковом
бюджете 39 NFE и одном семействе расписания uniform_t?

## Что выполняет ноутбук

1. Загружает полный финальный ZIP baseline с компьютера.
2. Проверяет SHA256 и восстанавливает исходники, результаты и точный начальный шум.
3. Проверяет сохранённый baseline, запускает тесты и численную сверку с Diffusers.
4. Выполняет короткий pilot трёх методов.
5. Генерирует по 5000 изображений для каждого метода при 39 NFE.
6. Сохраняет полные резервные ZIP между блоками генерации.
7. Вычисляет FID, строит таблицу и сравнительную сетку, упаковывает отчёт.

Выполненный baseline DDIM-100 восстанавливается из файлов. Его генерация и FID
уже сохранены. Новые вызовы модели относятся к проверкам и новым экспериментам.
В новой сессии потребуются установка зависимостей, загрузка весов, CIFAR-10 и Inception.

## Что загрузить

Один файл с компьютера, около 120 MiB.

`t4_ddim100_baseline_v1_full_final_fid_000000-005000_0270fdbb.zip`

Копии с суффиксом (1) или (2) также подходят. Исходники находятся внутри source.zip,
точные шумы находятся внутри initial_noise.pt. Дополнительный ZIP проекта не требуется.

## Как выполнять

Выберите **Среда выполнения → Сменить среду выполнения → T4 GPU**.
Запускайте по одной ячейке. После каждой ячейки скачивания дождитесь появления
готового файла в загрузках браузера, затем запускайте следующую генерацию.
Рабочая папка /content временная. Последний полный ZIP каждого метода позволяет
восстановить сохранённый прогресс при совпадающем окружении.

Библиотечный baseline имеет clipping=True и начало t=990. Новая группа имеет
clipping=False и диапазон t=999…0. Ранжируем три собственных метода внутри новой
группы. FID baseline остаётся отдельным результатом с явно указанными условиями.
"""),
        code("""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
from pathlib import Path
import torch

assert torch.cuda.is_available(), 'Выберите GPU T4'
assert 'T4' in torch.cuda.get_device_name(0), 'Для сравнения выберите T4'
print('GPU', torch.cuda.get_device_name(0))
print('PyTorch', torch.__version__)
PROJECT = Path('/content/diffusion-sampling')
SOURCE_BUNDLE = Path('/content/restored-source.zip')
"""),
        markdown("""
## 1. Восстановить baseline из полного ZIP

Следующая ячейка определяет функции восстановления на стандартной библиотеке Python.
Затем выберите один полный финальный архив baseline на компьютере.
Проверяются контрольные суммы, последовательности PNG и батчей, соответствие исходников.
Конфликтующие существующие файлы вызывают остановку с объяснением.
"""),
        code((root / 'scripts/colab_restore.py').read_text()),
        code("""
from google.colab import files

uploaded = files.upload()
assert len(uploaded) == 1, 'Выберите один полный ZIP baseline'
uploaded_name, archive_bytes = next(iter(uploaded.items()))
BASELINE_ZIP = Path('/content/baseline_full.zip')
BASELINE_ZIP.write_bytes(archive_bytes)
del uploaded, archive_bytes
baseline = restore_full_archive(BASELINE_ZIP, PROJECT, restore_source=True)
assert baseline['manifest']['config']['sampler'] == 'diffusers_ddim'
assert baseline['summary']['generation_complete']
assert baseline['summary']['generated_images'] == 5000
assert baseline['summary']['baseline_metrics_complete']
SOURCE_BUNDLE.write_bytes(baseline['source_bytes'])
BASELINE_DIR = baseline['run_dir']
print('Загружен', uploaded_name)
print('Восстановлена папка', BASELINE_DIR)
print('Сохранённый FID', baseline['summary']['fid'])
"""),
        markdown("""
## 2. Установить зависимости и проверить восстановленные данные

FID ниже читается из сохранённого отчёта Colab. Проверяем его связь с точным набором
PNG, согласованность метрик с батчами и хеш банка шумов. Полный численный пересчёт
FID относится к новым методам в следующих разделах.
"""),
        code("""
%pip install -q -e /content/diffusion-sampling
%cd /content/diffusion-sampling
import sys, json, math, shutil, subprocess
from io import BytesIO
from zipfile import ZIP_DEFLATED, ZipFile
from uuid import uuid4
from IPython.display import Image, display

os.environ.setdefault('TORCH_HOME', str(PROJECT / '.cache/torch'))
os.environ.setdefault('HF_HOME', str(PROJECT / '.cache/huggingface-home'))
sys.path.insert(0, str(PROJECT / 'src'))
from diffusion_sampling.config import load_config
from diffusion_sampling.experiment import load_shards
from diffusion_sampling.exports import CheckpointExporter
from diffusion_sampling.noise import load_noise
from diffusion_sampling.runtime import configure_runtime, environment
from diffusion_sampling.storage import file_sha256, object_sha256, tensor_sha256, write_json
from diffusion_sampling.study import create_sweep

configure_runtime(4)
manifest = baseline['manifest']
summary = baseline['summary']
shards = load_shards(BASELINE_DIR, manifest)
assert sum(s['end'] - s['start'] for s in shards) == 5000
assert math.isclose(sum(s['seconds'] for s in shards), summary['generation_seconds'], rel_tol=1e-12)
assert all(s['nfe'] == 100 for s in shards)
assert summary['sample_equivalent_model_evaluations'] == 500000
bank = load_noise(baseline['noise_path'])
assert bank['count'] == 5000 and bank['seed'] == 50000
assert bank['sha256'] == manifest['noise_sha256']
fid = json.loads((BASELINE_DIR / 'fid.json').read_text())
assert fid['fid'] == summary['fid'] and math.isfinite(fid['fid'])
assert fid['generated_count'] == 5000
assert fid['reference']['split'] == 'train' and fid['reference']['count'] == 50000
assert fid['experiment_id'] == manifest['experiment_id']
images = sorted((BASELINE_DIR / 'images').glob('*.png'))
assert object_sha256([(p.name, file_sha256(p)) for p in images]) == fid['generated_images_sha256']
print('Восстановление проверено. FID baseline', fid['fid'])
display(Image(filename=str(BASELINE_DIR / 'preview.png')))
del shards
"""),
        markdown("""
## 3. Зафиксировать настройки исследования

Общий бюджет 39 NFE. DDIM и DPM-Solver++ используют 39 узлов, Heun использует 20.
Все методы получают один сохранённый банк из 5000 шумов. Batch size 64, float32,
uniform_t, clipping=False. Для всех новых запусков используем одно окружение T4.

STUDY_NAME обозначает отдельную группу. При изменении настроек или окружения
задайте новое имя. Полные резервные ZIP включают точный шум, поэтому их размер
больше delta-архивов. Для восстановления каждого метода достаточно его последнего ZIP.
"""),
        code("""
STUDY_NAME = 'study_uniform39_v1'
NFE_BUDGET = 39
BATCH_SIZE = 64
NUM_IMAGES = 5000
BLOCK_BATCHES = 16
DOWNLOAD_RESULTS = True
RESTORE_STUDY_BACKUPS = False
RUN_LONG_REFERENCE = False

STUDY_DIR = PROJECT / 'results' / STUDY_NAME
EXPORT_DIR = PROJECT / 'results/exports' / STUDY_NAME
EXPORT_DIR.mkdir(parents=True, exist_ok=True)
SESSION_LOG = EXPORT_DIR / 'execution.log'
CURRENT_ENVIRONMENT = environment(torch.device('cuda'))
exporter = CheckpointExporter(PROJECT, EXPORT_DIR, source_bundle=SOURCE_BUNDLE)
source_noise_hash = bank['sha256']

def cli(*args):
    command = [sys.executable, '-m', 'diffusion_sampling', *map(str, args)]
    with SESSION_LOG.open('a') as log:
        with subprocess.Popen(command, cwd=PROJECT, text=True, stdout=subprocess.PIPE,
                              stderr=subprocess.STDOUT, bufsize=1) as process:
            for line in process.stdout:
                print(line, end='', flush=True)
                log.write(line)
                log.flush()
            returncode = process.wait()
    if returncode:
        raise subprocess.CalledProcessError(returncode, command)

def download_one(path):
    if path is None:
        return
    print('Готовый ZIP', path, f'{path.stat().st_size / 2**20:.1f} MiB')
    if DOWNLOAD_RESULTS:
        files.download(str(path))
    print('Проверьте завершение скачивания в браузере перед следующей генерацией.')

def snapshot(config, stage):
    run_dir = PROJECT / config.output_dir
    logs = run_dir / 'logs'
    logs.mkdir(exist_ok=True)
    for name in ['tests.log', 'execution.log']:
        if (EXPORT_DIR / name).exists():
            shutil.copyfile(EXPORT_DIR / name, logs / name)
    return exporter.full(run_dir, stage=stage)

cli('doctor')
"""),
        markdown("""
## 4. Продолжение ранее начатого исследования

При первом запуске оставьте RESTORE_STUDY_BACKUPS=False.
После обрыва сессии сначала снова загрузите baseline, затем поставьте True и выберите
последний полный ZIP каждого начатого метода этой группы. Можно выбрать до трёх файлов.
У законченных методов последующие ячейки проверят сохранённые данные.
При изменении окружения восстановленные измерения сохраняются отдельно. Для новой
сопоставимой группы потребуется новое STUDY_NAME и новые прогоны её методов.
"""),
        code("""
if RESTORE_STUDY_BACKUPS:
    uploaded_backups = files.upload()
    assert 1 <= len(uploaded_backups) <= 3, 'Выберите последний полный ZIP каждого метода'
    for name, data in uploaded_backups.items():
        with ZipFile(BytesIO(data)) as archive:
            saved_manifest = json.loads(archive.read('run/manifest.json'))
        assert saved_manifest['config']['output_dir'].startswith(f'results/{STUDY_NAME}/')
        assert saved_manifest['config']['num_samples'] == NUM_IMAGES
        assert saved_manifest['noise_sha256'] == source_noise_hash
        assert saved_manifest['environment'] == CURRENT_ENVIRONMENT, (
            'Окружение изменилось. Для новой группы задайте новое STUDY_NAME, '
            'отключите восстановление и выполните её методы заново.'
        )
        path = Path('/content') / Path(name).name
        path.write_bytes(data)
        restored = restore_full_archive(
            path, PROJECT,
            noise_equal=lambda left, right: (
                load_noise(left)['sha256'] == load_noise(right)['sha256'] == source_noise_hash
            ),
        )
        restored_shards = load_shards(restored['run_dir'], restored['manifest'])
        assert sum(s['end'] - s['start'] for s in restored_shards) == restored['summary']['generated_images']
        print('Восстановлено', restored['run_dir'], restored['summary']['generated_images'])
    del uploaded_backups, restored_shards
else:
    print('Новая группа экспериментов')
"""),
        markdown("""
## 5. Тесты и численная сверка с Diffusers

Сначала выполняются тесты сохранённой версии исходников. Затем два коротких примера
на настоящей модели проверяют промежуточные состояния и фактическое число вызовов.
Веса U-Net загрузятся при первом обращении. Дождитесь успешного завершения.
"""),
        code("""
tests = subprocess.run(
    [sys.executable, '-m', 'unittest', 'discover', '-s', 'tests', '-v'],
    cwd=PROJECT, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
)
(EXPORT_DIR / 'tests.log').write_text(tests.stdout)
print(tests.stdout)
tests.check_returncode()
verification = subprocess.run(
    [sys.executable, 'scripts/verify_samplers.py', '--device', 'cuda',
     '--output', f'results/{STUDY_NAME}/verification.json'],
    cwd=PROJECT, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
)
(EXPORT_DIR / 'verification.log').write_text(verification.stdout)
print(verification.stdout)
verification.check_returncode()
verification_report = json.loads((STUDY_DIR / 'verification.json').read_text())
assert verification_report['passed']
"""),
        markdown("""
## 6. Короткий pilot на 16 сохранённых шумах

Три метода при 19 NFE. Используем первые 16 шумов восстановленного банка fid.
Проверяем генерацию и соответствие колонок сетки одному начальному шуму.
Качество полной выборки оценивается в следующих разделах через FID.
"""),
        code("""
pilot_configs = PROJECT / 'configs' / f'{STUDY_NAME}_pilot'
create_sweep(pilot_configs, name=f'{STUDY_NAME}_pilot', budgets=(19,),
             schedules=('uniform_t',), num_samples=16, batch_size=8,
             purpose='pilot', device='cuda',
             noise_path=manifest['config']['noise_path'], noise_seed=50000)
pilot_runs = []
for method in ['ddim', 'dpmpp_2m', 'heun']:
    config_path = pilot_configs / f'{method}_uniform_t_nfe19.json'
    cli('run', '--config', config_path)
    pilot_runs.append(PROJECT / load_config(config_path).output_dir)
cli('compare', '--runs', *pilot_runs, '--output', STUDY_DIR / 'pilot_comparison.json')
display(Image(filename=str(STUDY_DIR / 'pilot_comparison.png')))
"""),
        markdown("""
## 7. Подготовить три полных запуска

Одна группа содержит три метода при 39 NFE, uniform_t, по 5000 изображений.
Ниже для каждого метода есть первый батч, пять блоков продолжения и отдельный FID.
После каждого блока создаётся полный ZIP. Скачивание выполняется следующей короткой
ячейкой. Дождитесь сохранения файла перед продолжением.
"""),
        code("""
CONFIG_DIR = PROJECT / 'configs' / STUDY_NAME
create_sweep(CONFIG_DIR, name=STUDY_NAME, budgets=(NFE_BUDGET,),
             schedules=('uniform_t',), num_samples=NUM_IMAGES, batch_size=BATCH_SIZE,
             purpose='baseline', device='cuda',
             noise_path=manifest['config']['noise_path'], noise_seed=50000)

def run_block(config, max_batches, stage):
    config_path = CONFIG_DIR / f'{config.sampler}_uniform_t_nfe{NFE_BUDGET}.json'
    run_dir = PROJECT / config.output_dir
    if (run_dir / 'summary.json').exists():
        previous = json.loads((run_dir / 'summary.json').read_text())
        if previous['generation_complete']:
            saved_manifest = json.loads((run_dir / 'manifest.json').read_text())
            assert saved_manifest['config'] == config.to_dict()
            assert saved_manifest['environment'] == CURRENT_ENVIRONMENT
            assert saved_manifest['noise_sha256'] == source_noise_hash
            saved_shards = load_shards(run_dir, saved_manifest)
            assert sum(s['end'] - s['start'] for s in saved_shards) == NUM_IMAGES
            print('Генерация уже завершена', config.sampler)
            return None
    cli('run', '--config', config_path, '--max-batches', max_batches)
    saved_manifest = json.loads((run_dir / 'manifest.json').read_text())
    result = json.loads((run_dir / 'summary.json').read_text())
    assert saved_manifest['noise_sha256'] == source_noise_hash
    assert result['nfe_per_image'] == NFE_BUDGET
    assert saved_manifest['environment'] == CURRENT_ENVIRONMENT
    print('Изображений', result['generated_images'], '/', NUM_IMAGES)
    print('Секунд на изображение', result['seconds_per_image'])
    print('Осталось минут денойзинга примерно',
          (NUM_IMAGES - result['generated_images']) * result['seconds_per_image'] / 60)
    return snapshot(config, stage)

def finish_method(config, visual_notes):
    run_dir = PROJECT / config.output_dir
    result = json.loads((run_dir / 'summary.json').read_text())
    assert result['generation_complete'] and result['generated_images'] == NUM_IMAGES
    if not result['baseline_metrics_complete']:
        cli('evaluate', '--run', run_dir, '--device', 'cuda', '--download')
    result = json.loads((run_dir / 'summary.json').read_text())
    assert result['baseline_metrics_complete'] and math.isfinite(result['fid'])
    if visual_notes.strip():
        (run_dir / 'visual_review.md').write_text(visual_notes.strip() + '\\n')
    print('FID', result['fid'])
    print('NFE', result['nfe_per_image'])
    print('Время денойзинга', result['generation_seconds'])
    return snapshot(config, 'final_fid')
"""),
    ]
    for method, label in [('ddim', 'DDIM'), ('dpmpp_2m', 'DPM-Solver++ 2M'), ('heun', 'Heun')]:
        cells.extend([
            markdown(f"""
## {label}. Первый батч и оценка времени

При восстановленном запуске читаются уже сохранённые батчи и добавляется максимум один.
Первый батч нового запуска содержит 64 изображения.
"""),
            code(f"""
config = load_config(CONFIG_DIR / f'{method}_uniform_t_nfe{{NFE_BUDGET}}.json')
latest_archive = run_block(config, 1, 'first_batch')
display(Image(filename=str(PROJECT / config.output_dir / 'preview.png')))
"""),
            code('download_one(latest_archive)'),
        ])
        for block in range(1, 6):
            cells.extend([
                markdown(f"""
### {label}. Блок {block} из 5

Добавляем до 16 батчей. Если генерация уже завершена, проверяются сохранённые результаты.
"""),
                code(f"latest_archive = run_block(config, BLOCK_BATCHES, 'block_{block}')"),
                code('download_one(latest_archive)'),
            ])
        cells.extend([
            markdown(f"""
### {label}. FID, визуальные наблюдения и полный финальный ZIP

Посмотрите сетку и запишите наблюдения в VISUAL_NOTES. При пустой строке текстовый
файл пропускается. FID сравнивает 5000 generated с 50000 CIFAR-10 train.
Уже сохранённый завершённый FID читается из отчёта. Для нового расчёта нужны CIFAR-10
и веса Inception. После выполнения скачайте полный ZIP следующей ячейкой.
"""),
            code("""
VISUAL_NOTES = ''
latest_archive = finish_method(config, VISUAL_NOTES)
display(Image(filename=str(PROJECT / config.output_dir / 'preview.png')))
"""),
            code('download_one(latest_archive)'),
        ])
    cells.extend([
        markdown("""
## 8. Сравнительная таблица и изображения

Проверяются одинаковые веса, шумы, точность, clipping, крайние точки, окружение,
batch size и протокол FID. Все три запуска должны быть завершены.
В строках сетки расположены методы, в колонках используется одинаковый шум.
Меньшее значение FID означает лучший результат по выбранной метрике и протоколу.
Один общий банк даёт первичное сравнение. Для оценки устойчивости нужны независимые банки.
"""),
        code("""
runs = [PROJECT / load_config(CONFIG_DIR / f'{method}_uniform_t_nfe{NFE_BUDGET}.json').output_dir
        for method in ['ddim', 'dpmpp_2m', 'heun']]
cli('compare', '--runs', *runs, '--output', STUDY_DIR / 'comparison.json')
report = json.loads((STUDY_DIR / 'comparison.json').read_text())
assert len(report['rows']) == 3 and report['budgets'] == [NFE_BUDGET]
for row in report['rows']:
    assert row['fid'] is not None
    print(row['sampler'], 'NFE', row['nfe'], 'FID', row['fid'],
          'сек/изображение', row['seconds_per_image'])
display(Image(filename=str(STUDY_DIR / 'comparison.png')))
"""),
        markdown("""
## 9. Дополнительный длинный эталон и MSE

По умолчанию RUN_LONG_REFERENCE=False. При True выполняются DDIM-500 и DDIM-1000
на тех же 16 шумах pilot, проверяется разница между эталонами, вычисляется MSE
коротких методов к DDIM-1000. MSE считается по float32 до преобразования в PNG.
Это дополнительная ось оценки численной точности. Связь с FID проверяется отдельно.
Подбор расписания и проверка на новых независимых шумах относятся к дальнейшему этапу.
"""),
        code("""
if RUN_LONG_REFERENCE:
    from dataclasses import replace
    references = []
    for budget in [500, 1000]:
        template = load_config(PROJECT / f'configs/reference_ddim_{budget}_debug.json')
        reference_config = replace(
            template, device='cuda', noise_path=manifest['config']['noise_path'],
            noise_seed=50000, output_dir=f'results/{STUDY_NAME}/reference_ddim_{budget}',
        )
        path = CONFIG_DIR / f'reference_ddim_{budget}.json'
        write_json(path, reference_config.to_dict())
        cli('run', '--config', path)
        references.append(PROJECT / reference_config.output_dir)
    cli('compare', '--runs', references[0], '--reference', references[1],
        '--output', STUDY_DIR / 'reference_convergence.json')
    cli('compare', '--runs', *pilot_runs, '--reference', references[1],
        '--output', STUDY_DIR / 'pilot_mse.json')
else:
    print('Длинный эталон отключён. Основное сравнение FID доступно выше.')
"""),
        markdown("""
## 10. Сохранить итоговый отчёт

Архив отчёта содержит comparison.json, CSV, сравнительную сетку, проверку реализаций,
pilot, конфигурации и логи, а также отдельную карточку библиотечного baseline.
Полные результаты генерации находятся в трёх скачанных *_full_final_fid_*.zip.
Скачайте также выполненный ноутбук через **Файл → Скачать → Скачать IPYNB**.
"""),
        code("""
write_json(STUDY_DIR / 'baseline_context.json', {
    'summary': baseline['summary'], 'config': baseline['manifest']['config'],
    'note': 'Separate library baseline protocol. Rank the three common-protocol runs in comparison.json.',
})
report_archive = EXPORT_DIR / f'{STUDY_NAME}_report_{uuid4().hex[:8]}.zip'
with ZipFile(report_archive, 'w', compression=ZIP_DEFLATED) as archive:
    for path in sorted(STUDY_DIR.glob('*')):
        if path.is_file():
            archive.write(path, Path('report') / path.name)
    for path in sorted(CONFIG_DIR.glob('*.json')):
        archive.write(path, Path('configs') / path.name)
    for path in sorted(EXPORT_DIR.glob('*.log')):
        archive.write(path, Path('logs') / path.name)
    archive.write(SOURCE_BUNDLE, 'source.zip')
print('Отчёт готов', report_archive)
"""),
        code('download_one(report_archive)'),
        markdown("""
## Ожидаемый итог и дальнейшее исследование

Для каждого метода получены 5000 изображений, NFE=39, конечный FID, время,
точные шумы, тензоры и полный ZIP. Есть сравнительная таблица и сетка одинаковых шумов.
Численные значения новых методов появятся после фактического запуска на T4.

Далее добавляем бюджеты 19/79, сравниваем uniform_t/log_sigma/karras, подбираем
опорные уровни на calibration и проверяем их на независимых шумах и другом сэмплере.
Алгоритм поиска расписания требует отдельной реализации.
"""),
    ])
    for index, cell in enumerate(cells):
        cell['id'] = f'restored-study-{index:03d}'
    notebook = {'nbformat': 4, 'nbformat_minor': 5, 'cells': cells, 'metadata': {
        'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
        'language_info': {'name': 'python'}, 'accelerator': 'GPU',
        'colab': {'name': '03_sampler_study_from_backup.ipynb'},
    }}
    output = root / 'notebooks/03_sampler_study_from_backup.ipynb'
    output.write_text(json.dumps(notebook, ensure_ascii=False, indent=2) + '\n')
    print(output)


if __name__ == '__main__':
    main()
