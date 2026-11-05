"""Build notebook 04 with resumable references, MSE and a 27-config study."""

import json
from pathlib import Path

from build_colab_notebook import code, markdown


def main():
    root = Path(__file__).resolve().parents[1]
    cells = [
        markdown("""
# Длинный эталон, MSE, бюджеты и независимые шумы

Исследуем DDIM, DPM-Solver++ 2M и Heun на google/ddpm-cifar10-32.
Как зависят качество и ошибка относительно длинной генерации от NFE и расписания?
Повторяется ли преимущество выбранных конфигураций на новых шумах?

Начинайте в новой сессии Colab. Восстановление данных доступно в CPU-сессии.
Для вычислений выберите
**Среда выполнения → Сменить среду выполнения → T4 GPU**.
При ограничении квоты GPU вычислительную часть можно выполнить после появления T4.

## Какие файлы загрузить

Три полных финальных ZIP из выполненного ноутбука 03. Подходят версии с визуальными
описаниями, названия которых заканчиваются на _reviewed.zip.

```text
ddim_uniform_t_nfe39_full_final_fid_000000-005000_8ce6f444.zip
dpmpp_2m_uniform_t_nfe39_full_final_fid_000000-005000_3376f091.zip
heun_uniform_t_nfe39_full_final_fid_000000-005000_a018e253.zip
```

Исходники и точный банк из 5000 шумов находятся внутри ZIP. После восстановления
устанавливаются зависимости. Для новых генераций загрузятся веса модели и данные FID.

## Фазы и ожидаемые результаты

| Фаза | Вычисления | Выход |
|---|---|---|
| convergence | DDIM-500/1000 и три метода при 39 NFE на 16 шумах | Проверка чувствительности к эталону |
| mse64 | Длинный эталон и 27 конфигураций на 64 шумах | MSE, ошибки отдельных изображений, графики |
| grid | 3 метода × 3 расписания × 3 бюджета, по 5000 изображений | FID, время, NFE, графики |
| mse1024 | Эталон и выбранные конфигурации на 1024 новых шумах | Подтверждение MSE |
| independent | Выбранные конфигурации на двух новых банках по 5000 шумов | Повторные FID и разброс |

Три готовых результата uniform_t/39 NFE используются при совпадении окружения.
При другом окружении создаётся новая сопоставимая группа. Длинный численный эталон
оценивается через различие DDIM-500 и DDIM-1000. FID и MSE интерпретируются отдельно.

## Как работать с блоками

Запускайте по одной ячейке. Один блок продвигает одну конфигурацию максимум на 16
батчей. Для длинного эталона блок ограничен одним батчем. После блока сохраняется
полный ZIP текущего запуска, скачивание выполняется следующей короткой ячейкой.
Дождитесь сохранения файла в загрузках браузера перед следующим блоком.

Сначала скачайте support ZIP с планом и всеми точными банками шумов. Обновите эту
копию после фиксации выбранных конфигураций. При потере сессии загрузите три исходных
ZIP, support ZIP и последний полный checkpoint каждого выполненного или начатого
запуска, который хотите восстановить. Завершённые задачи пропускаются.

План требует нескольких сессий T4. Ориентир полного расширения составляет 7–9 часов
вычислений, фактическое время уточняется по завершённым блокам и доступным ресурсам.
"""),
        code("""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
from pathlib import Path
from io import BytesIO
from zipfile import ZipFile
import json
import torch

PROJECT = Path('/content/diffusion-sampling')
SOURCE_BUNDLE = Path('/content/restored-source.zip')
print('CUDA доступна', torch.cuda.is_available())
if torch.cuda.is_available():
    print('GPU', torch.cuda.get_device_name(0))
print('Восстановление файлов доступно в текущей сессии')
"""),
        markdown("""
## 1. Проверить и восстановить три архива

Сначала определяются функции восстановления. Затем выберите три полных финальных
ZIP методов. Контрольные суммы проверяются перед восстановлением. Конфликтующие
существующие файлы вызывают ошибку с объяснением.
"""),
        code((root / 'scripts/colab_restore.py').read_text()),
        code("""
from google.colab import files

uploaded = files.upload()
assert len(uploaded) == 3, 'Выберите три полных ZIP методов из ноутбука 03'
archived_runs = []
seen_methods = set()
for index, (name, data) in enumerate(uploaded.items()):
    path = Path('/content') / f'input_method_{index}.zip'
    path.write_bytes(data)
    restored = restore_full_archive(
        path, PROJECT, restore_source=(index == 0),
        noise_equal=lambda left, right: torch.equal(
            torch.load(left, map_location='cpu', weights_only=True)['noise'],
            torch.load(right, map_location='cpu', weights_only=True)['noise'],
        ),
    )
    config = restored['manifest']['config']
    assert config['sampler'] in {'ddim', 'dpmpp_2m', 'heun'}
    assert config['schedule'] == 'uniform_t' and not config['clip_sample']
    assert restored['summary']['nfe_per_image'] == 39
    assert restored['summary']['generated_images'] == 5000
    assert restored['summary']['baseline_metrics_complete']
    seen_methods.add(config['sampler'])
    archived_runs.append(restored['run_dir'])
    if index == 0:
        SOURCE_BUNDLE.write_bytes(restored['source_bytes'])
    print(config['sampler'], 'FID', restored['summary']['fid'])
assert seen_methods == {'ddim', 'dpmpp_2m', 'heun'}
del uploaded, data
"""),
        markdown("""
## 2. Установить зависимости и определить функции исследования

Устанавливаем сохранённый пакет эксперимента и matplotlib для научных графиков.
Функции следующей ячейки встроены в ноутбук. Все формулы сэмплеров, проверки
запусков, FID и экспорт используются из восстановленной версии пакета.
"""),
        code("""
%pip install -q -e /content/diffusion-sampling matplotlib
%cd /content/diffusion-sampling
import sys
sys.path.insert(0, str(PROJECT / 'src'))
os.environ.setdefault('TORCH_HOME', str(PROJECT / '.cache/torch'))
os.environ.setdefault('HF_HOME', str(PROJECT / '.cache/huggingface-home'))
from IPython.display import Image, display
"""),
        code((root / 'scripts/extended_study.py').read_text()),
        markdown("""
## 3. Параметры и восстановление прогресса

Для первоначального запуска оставьте RESTORE_PROGRESS=False. Для продолжения
задайте True. STUDY_NAME должен соответствовать сохранённому support ZIP.
При смене окружения используйте новое имя и создайте новую группу измерений.

Следующая ячейка требует T4. Все задачи имеют float32, clipping=False, batch size 64,
одинаковые крайние уровни шума и один сохранённый банк внутри каждого сравнения.
"""),
        code("""
STUDY_NAME = 'extended_study_v1'
RESTORE_PROGRESS = False
DOWNLOAD_RESULTS = True
MAX_NEW_BATCHES = 16

assert torch.cuda.is_available(), 'Для вычислительных фаз подключите GPU T4'
assert 'T4' in torch.cuda.get_device_name(0), 'Выберите T4 для этой группы измерений'
configure_runtime(4)
current_environment = environment(torch.device('cuda'))

def same_saved_noise(left, right):
    existing, incoming = load_noise(left), load_noise(right)
    count = incoming['count']
    return existing['count'] >= count and torch.equal(existing['noise'][:count], incoming['noise'])

if RESTORE_PROGRESS:
    progress_files = files.upload()
    support_files = []
    run_files = []
    for name, data in progress_files.items():
        path = Path('/content') / Path(name).name
        path.write_bytes(data)
        with ZipFile(path) as archive:
            if 'support_manifest.json' in archive.namelist():
                support_files.append(path)
            else:
                run_files.append(path)
    assert len(support_files) == 1, 'Выберите один последний support ZIP и checkpoint ZIP запусков'
    saved_plan = restore_support(support_files[0], PROJECT, current_environment)
    assert saved_plan['name'] == STUDY_NAME
    allowed_configs = [task['config'] for tasks in saved_plan['phases'].values() for task in tasks]
    for path in run_files:
        with ZipFile(path) as archive:
            saved_manifest = json.loads(archive.read('run/manifest.json'))
        assert saved_manifest['config'] in allowed_configs, 'ZIP относится к другой группе'
        assert saved_manifest['environment'] == current_environment
        restored = restore_full_archive(path, PROJECT, noise_equal=same_saved_noise)
        if restored['summary']['generation_complete']:
            _load_run(restored['run_dir'])
        print('Восстановлено', restored['run_dir'], restored['summary']['generated_images'])
    del progress_files

plan = prepare_plan(PROJECT, STUDY_NAME, archived_runs=archived_runs)
for phase in plan['phases']:
    rows = phase_progress(PROJECT, plan, phase)
    print(phase, 'готово', sum(row['done'] for row in rows), '/', len(rows))
support_zip = support_backup(PROJECT, plan, SOURCE_BUNDLE)
print('Support ZIP', support_zip)
"""),
        code("""
def download_one(path):
    if path is None:
        return
    print('Архив готов', path, f'{path.stat().st_size / 2**20:.1f} MiB')
    if DOWNLOAD_RESULTS:
        files.download(str(path))
    print('Дождитесь завершения скачивания в браузере')

download_one(support_zip)
"""),
        markdown("""
## 4. Тесты и численная сверка реализаций

Ожидается успешное завершение тестов. Затем выполняется короткая численная сверка
с Diffusers на настоящей модели в текущем окружении T4. Сравниваются промежуточные
состояния и фактический NFE. Веса загрузятся при первом обращении к модели.
"""),
        code("""
import subprocess
tests = subprocess.run([sys.executable, '-m', 'unittest', 'discover', '-s', 'tests', '-v'],
                       cwd=PROJECT, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
print(tests.stdout)
tests.check_returncode()
test_log = PROJECT / f'results/{STUDY_NAME}/reports/tests.log'
test_log.parent.mkdir(parents=True, exist_ok=True)
test_log.write_text(tests.stdout)
verification = subprocess.run(
    [sys.executable, 'scripts/verify_samplers.py', '--device', 'cuda',
     '--output', f'results/{STUDY_NAME}/reports/verification.json'],
    cwd=PROJECT, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
)
print(verification.stdout)
(test_log.parent / 'verification.log').write_text(verification.stdout)
verification.check_returncode()
assert json.loads((test_log.parent / 'verification.json').read_text())['passed']
"""),
        markdown("""
## 5. Выполнение выбранной фазы

Сначала PHASE='convergence'. Затем последовательно mse64 и grid.
После grid сформируйте отчёт и зафиксируйте выбор в разделе 7.
После фиксации доступны mse1024 и independent.

Для каждой фазы повторяйте две ячейки блока и скачивания, пока все строки
не получат done=True. Строки прогресса показывают число изображений каждой задачи.
Следующий блок автоматически выбирает первую незавершённую задачу.

Для Heun бюджеты 19/39/79 соответствуют 10/20/40 узлам. Для DDIM и DPM-Solver++
число узлов равно NFE. При завершении генерации FID-задачи автоматически вычисляется
FID по протоколу generated-5k / CIFAR-10 train-50k. Время FID учитывается отдельно.
"""),
        code("""
PHASE = 'convergence'  # Затем mse64, grid, mse1024, independent.
for row in phase_progress(PROJECT, plan, PHASE):
    print(row)
"""),
        code("""
latest_checkpoint = run_next_block(
    PROJECT, plan, PHASE, SOURCE_BUNDLE, max_batches=MAX_NEW_BATCHES,
)
for row in phase_progress(PROJECT, plan, PHASE):
    print(row)
"""),
        code('download_one(latest_checkpoint)'),
        markdown("""
## 6. Отчёт завершённой фазы

После done=True у всех задач выполните следующую ячейку. Она проверяет условия
сравнения, сохраняет JSON/CSV/сетки и графики. Для MSE сохраняются ошибки отдельных
изображений. Для convergence выводится также MSE DDIM-500 к DDIM-1000.

Посмотрите величину ошибки между эталонами и порядок коротких методов при обоих
эталонах. Заметная чувствительность к эталону должна быть отражена в выводах.
"""),
        code("""
phase_report = write_phase_report(PROJECT, plan, PHASE)
print(json.dumps(phase_report, ensure_ascii=False, indent=2))
reports_dir = PROJECT / f'results/{STUDY_NAME}/reports'
curve = reports_dir / f'{PHASE}_curves.png'
if curve.exists():
    display(Image(filename=str(curve)))
if PHASE == 'convergence':
    print('Ошибка между эталонами')
    print((reports_dir / 'reference_convergence.json').read_text())
reports_zip = report_backup(PROJECT, plan, SOURCE_BUNDLE)
"""),
        code('download_one(reports_zip)'),
        markdown("""
## 7. Зафиксировать конфигурации перед независимой проверкой

Выполняйте после полного grid. Для каждого метода выбирается лучшее расписание
при 39 NFE по основному банку 50000–54999. Добавляется uniform_t каждого метода.
Совпадения объединяются. Получается от трёх до шести фиксированных конфигураций.

После фиксации выполняются mse1024 на шумах 40000–41023 и independent на двух
банках 60000–64999 и 70000–74999. Повторно скачайте support ZIP, чтобы сохранить
выбранные конфигурации и точные банки перед подтверждающими экспериментами.
"""),
        code("""
plan = lock_selection(PROJECT, plan)
print('Зафиксированы конфигурации', plan['selection'])
support_zip = support_backup(PROJECT, plan, SOURCE_BUNDLE)
for phase in ['mse1024', 'independent']:
    print(phase, 'задач', len(plan['phases'][phase]))
"""),
        code('download_one(support_zip)'),
        markdown("""
Вернитесь в раздел 5 и выполните сначала mse1024, затем independent.
После каждой завершённой фазы скачайте отчёт из раздела 6.

## Какие результаты сохранить

1. Последний support ZIP после фиксации конфигураций.
2. Последний полный checkpoint каждого запуска для восстановления данных.
3. Последний reports ZIP с JSON, CSV, сетками и графиками.
4. Выполненный ноутбук через **Файл → Скачать → Скачать IPYNB**.

В independent.csv записаны средний FID и минимальное/максимальное значения по двум
новым банкам. Каждый подтверждающий прогон сохранён отдельно. Основной банк служит
для выбора конфигураций. Устойчивость оценивается по новым банкам.

Визуальные наблюдения и проверенные результаты uniform_t/39 NFE доступны в репозитории
в docs/study_uniform39_results.md и docs/visual_reviews. Подбор собственных опорных
точек расписания требует следующей реализации после сравнения стандартных сеток.
"""),
    ]
    for index, cell in enumerate(cells):
        cell['id'] = f'extended-study-{index:03d}'
    notebook = {'nbformat': 4, 'nbformat_minor': 5, 'cells': cells, 'metadata': {
        'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
        'language_info': {'name': 'python'}, 'accelerator': 'GPU',
        'colab': {'name': '04_extended_study_colab.ipynb'},
    }}
    output = root / 'notebooks/04_extended_study_colab.ipynb'
    output.write_text(json.dumps(notebook, ensure_ascii=False, indent=2) + '\n')
    print(output)


if __name__ == '__main__':
    main()
