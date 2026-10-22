# Background Bias in Image Classification — рабочий план

CSCI 494: Deep Learning · Amina Abdikalyk, Islam Sanatbek, Tomiris Rakymbay, Assylkhan Tleubayev

**Исследовательский вопрос.** Модель определяет породу по самому животному или использует фон как подсказку? Уменьшает ли случайная замена фона при обучении эту зависимость, и какой ценой для обычной точности?

**Гипотезы (записать в отчёт заранее):**
- H1. У ResNet-18 A заметно падает accuracy на background-swapped по сравнению с original.
- H2. У ResNet-18 B gap меньше, чем у A, при сопоставимой accuracy на original.
- H3. На foreground-only обе модели теряют меньше, чем на swapped (нейтральный фон «не врёт», а чужой фон может подсказывать неверно).

Если гипотезы не подтвердятся, это тоже результат — его объясняем в отчёте.

---

## 0. Старт: что сделать в первые 2–3 дня

Порядок важен: сначала решения и инфраструктура, потом код.

| # | Шаг | Кто | Готово, когда |
|---|---|---|---|
| 0.1 | Узнать дедлайн курса и формат сдачи (отчёт? презентация? код?) | Amina | Дата записана в этом файле, недели пересчитаны |
| 0.2 | Создать GitHub-репозиторий, добавить всех четверых | Amina | Все могут сделать push |
| 0.3 | Создать общую папку `bgbias/` в Google Drive (структура в разделе 2) | Islam | Папка расшарена всем |
| 0.4 | Зафиксировать решения из раздела 1 (фон, маски, гиперпараметры) | Все, 30 мин созвон | Раздел 1 заполнен, вопросы закрыты |
| 0.5 | Ноутбук `00_setup.ipynb`: clone репо, установка зависимостей, загрузка датасета в `/content`, проверка GPU | Islam | Ноутбук проходит «Run all» у каждого участника |
| 0.6 | Ноутбук `01_data.ipynb`: splits + замена фона + 20–30 примеров | Islam + Amina | Примеры просмотрены всей командой, splits закоммичены |

**После 0.6 команда смотрит примеры вместе и только потом запускает baseline.**

---

## 1. Решения, которые нужно зафиксировать до кода

### 1.1 Данные и разбиение
- Датасет: **Oxford-IIIT Pet** через `torchvision.datasets.OxfordIIITPet(split=..., target_types=["category", "segmentation"], download=True)`.
  - 37 пород, около 7.4k изображений; официальные `trainval` (~3680) и `test` (~3669).
  - Маска — **trimap**: `1` = животное, `2` = фон, `3` = граница/неопределённо.
- **Test = официальный `test`**, не трогаем до финальной оценки.
- **Validation = 15–20% от `trainval`**, стратифицированно по породам (`sklearn.model_selection.train_test_split(..., stratify=labels, random_state=42)`).
- Splits сохраняем как CSV (`splits/train.csv`, `val.csv`, `test.csv`: `image_id, label, breed, species`) и **коммитим в git**. Все модели используют только их.
- Сплит делается по исходным изображениям **до** создания любых вариантов. Все версии одной картинки остаются в одном разделе.
- Проверить и записать: битые файлы, grayscale/CMYK изображения (`.convert("RGB")`), размеры, баланс классов.

### 1.2 Маски и вставка на новый фон
- Foreground = `trimap == 1` **плюс** граница `trimap == 3` (иначе срезается шерсть по контуру). Сравнить оба варианта на примерах, выбрать один и зафиксировать.
- Мягкие края: alpha-маска с небольшим Gaussian blur (σ ≈ 1–2 px) → `out = alpha * img + (1 - alpha) * bg`.
- Фон ресайзится/кропается под размер исходного изображения, затем общий pipeline (resize → crop 224).
- **Foreground-only:** нейтральный фон — серый цвет среднего ImageNet `(124, 116, 104)`. Зафиксировать один вариант.

### 1.3 Откуда брать фоны
Рекомендация: **Places365 (validation, 256×256)** — сцены без привязки к породам.
- Отобрать ~3–5k изображений и разделить на **три непересекающихся пула**: `bg_train`, `bg_val`, `bg_test`.
- Лучше делить по категориям сцен (например, разные scene classes в train и test), чтобы тест проверял действительно новые фоны.
- Вручную/быстро отфильтровать сцены, где часто бывают животные (пастбище, зоопарк и т.п.).

Дополнительный вариант для анализа (по времени): **фон из Pet-изображения другой породы** (как Mixed-Rand в Xiao et al., 2020). Это самая прямая проверка «фон подсказывает породу». Минус — на месте вырезанного животного остаётся дыра, её надо заливать (inpainting) или брать картинки, где животное занимает мало площади.

### 1.4 Тестовые варианты фиксированы
- Для каждого val/test изображения заранее выбираем фон из своего пула с фиксированным seed и сохраняем manifest: `manifests/val_swapped.csv`, `manifests/test_swapped.csv` (`image_id, bg_path, bg_category`).
- Варианты генерируются на лету, детерминированно (фон из manifest, центр-кроп фона). Поэтому все модели видят **одни и те же** входы, и хранить готовые картинки не нужно.
- `val_swapped` — только для мониторинга, **не** для выбора модели.

### 1.5 Протокол обучения ResNet-18 (A и B идентичны, кроме фона)
| Параметр | Значение (стартовое) |
|---|---|
| Архитектура | ResNet-18, `weights=IMAGENET1K_V1`, новая голова на 37 классов |
| Вход | 224×224, ImageNet-нормализация |
| Аугментации (общие) | RandomResizedCrop(224, scale=(0.7, 1.0)), HorizontalFlip |
| Замена фона | A: p = 0; B: p = 0.5, фон из `bg_train`, до остальных аугментаций |
| Optimizer | AdamW, lr 1e-3 голова / 1e-4 backbone, weight decay 1e-4 |
| Расписание | cosine, 15 эпох (проверить, что сходится; при необходимости 20) |
| Batch / precision | 64, AMP |
| Выбор checkpoint | лучший по **accuracy на original validation** — одинаково для A и B |
| Seeds | 0, 1, 2 (фиксируем `torch`, `numpy`, `random`, DataLoader generator) |

Важно: настройки подбираются только по validation и потом **одни и те же** для A и B. Тест смотрим только в конце.

### 1.6 HOG + linear SVM
- Resize 128×128, grayscale, `skimage.feature.hog(orientations=9, pixels_per_cell=(8,8), cells_per_block=(2,2))`.
- `StandardScaler` + `LinearSVC`, C из `{0.01, 0.1, 1, 10}` по validation.
- Один запуск (детерминированный). Оценка на тех же трёх тестовых условиях.
- Ожидаем низкую точность на 37 породах — это нормально, это нижняя планка.

### 1.7 Метрики
- **Accuracy** и **macro F1** на каждом условии: original, swapped, fg-only.
- `gap_swap = acc(original) − acc(swapped)`, `gap_fg = acc(original) − acc(fg-only)`.
- Для ResNet: mean ± std по трём seed.
- Дополнительно: **flip rate** — доля изображений, где предсказание изменилось после замены фона; per-class accuracy; confusion matrix; кошки vs собаки.
- Опционально: **background-only** (животное закрашено серым) — насколько модель угадывает породу вообще без животного. Сильный диагностический аргумент.
- Опционально: парный bootstrap по тестовым изображениям для доверительного интервала разницы gap(A) − gap(B).

---

## 2. Структура проекта

```
dl_project/                     # github.com/mundmina/dl_project
├── PLAN.md · README.md · requirements.txt
├── configs/                    # resnet_A.yaml (p_swap 0.0), resnet_B.yaml (p_swap 0.5)   [готово]
├── splits/                     # train/val/test CSV — коммитим
├── manifests/                  # background_pools.csv, val/test_swapped.csv — коммитим
├── src/bgbias/
│   ├── data_setup.py           # загрузка датасетов + кэш архивов в Drive          [готово]
│   ├── splits.py               # splits, проверки, аудит изображений и масок       [готово]
│   ├── backgrounds.py          # пулы фонов Places365, manifests                    [готово]
│   ├── compose.py              # trimap → alpha, замена фона, fg-only, bg-only      [готово]
│   ├── datasets.py             # Dataset для всех условий + случайная замена        [готово]
│   ├── models.py               # ResNet-18 с новой головой                         [готово]
│   ├── train.py                # цикл обучения, checkpoints, логирование           [готово]
│   ├── evaluate.py             # условия → metrics.json                            [готово]
│   ├── hog_svm.py              # baseline                                          [готово]
│   ├── gradcam.py                                                                  [готово]
│   └── utils.py                # seed, пути                                         [готово]
├── notebooks/
│   ├── 00_setup.ipynb          # [готово]
│   ├── 01_data.ipynb           # [готово, запущен]
│   ├── 02_hog_svm.ipynb        # [готово, запущен]
│   ├── 03_train_resnet.ipynb   # [готово, запущен]
│   ├── 04_evaluate.ipynb       # [готово, запущен]
│   ├── 05_analysis.ipynb       # [готово, не запускался]
│   └── 06_demo.ipynb
├── results/                    # metrics.json, data_audit.csv — коммитим
└── report/                     # LaTeX, figures/
```

Google Drive `MyDrive/bgbias/` (большие файлы, не в git):
```
data/        # архивы датасета и фонов (копировать в /content перед обучением)
week1/       # копии splits / manifests / галереи из 01_data
checkpoints/ # resnet_A_seed0.pt ...
logs/
```

Правила:
- Код живёт в `src/`, ноутбуки только вызывают функции. Так все запускают одно и то же.
- Данные **копируются из Drive в `/content`** перед обучением — чтение прямо с Drive очень медленное.
- Каждый запуск сохраняет `results/<run_name>/metrics.json` + config + seed + git commit hash.
- Работа через ветки и pull request; в `main` — только проверенное.

---

## 3. План по неделям

Сроки ориентировочные — пересчитать после шага 0.1.

### Неделя 1 — данные и подготовка
| Задача | Кто | Результат / критерий готовности |
|---|---|---|
| Загрузка датасета, проверка классов, битых файлов, баланса | Islam | Таблица: число изображений по породам в train/val/test |
| Stratified split, сохранение CSV | Amina | `splits/*.csv` в git, пересечений нет (assert в коде) |
| Проверка масок: 50 случайных trimap поверх картинок | Islam | Список проблемных масок, решение по границе (класс 3) |
| Скачать и отобрать фоны, разделить на 3 пула | Islam | `bg_train / bg_val / bg_test`, без пересечений |
| `compose.py`: замена фона и нейтральный фон | Islam + Amina | Функции + unit-проверка на 1 картинке |
| Генерация фиксированных val/test вариантов + manifest | Islam | Файлы в Drive, manifest в git |
| Галерея 20–30 примеров «оригинал → маска → новый фон → нейтральный фон» | Amina | Картинка-сетка в `report/figures/`, команда её посмотрела |
| Каркас `train.py` на маленьком подмножестве (smoke test) | Assylkhan | 1 эпоха на 200 картинках проходит без ошибок |

### Неделя 2 — базовые модели
| Задача | Кто | Результат |
|---|---|---|
| HOG + SVM, подбор C по validation | Tomiris | `results/hog_svm/metrics.json` на трёх условиях |
| `evaluate.py`: accuracy, macro F1, gap, flip rate | Tomiris | Одна функция для всех моделей |
| ResNet-18 A: подбор lr/эпох по validation (1 seed) | Assylkhan | Зафиксированный config, кривые обучения |
| ResNet-18 A: seeds 0, 1, 2 | Assylkhan | 3 checkpoints + metrics |
| Проверка: обучение воспроизводимо (тот же seed → близкий результат) | Amina | Запись в README |

### Неделя 3 — основной эксперимент
| Задача | Кто | Результат |
|---|---|---|
| ResNet-18 B (p = 0.5), seeds 0, 1, 2, тот же config | Assylkhan | 3 checkpoints + metrics |
| Оценка A, B, HOG на original / swapped / fg-only | Tomiris | Главная таблица (mean ± std) |
| (если есть время) ablation p ∈ {0, 0.25, 0.5, 1.0} — нужно только 25% и 100%, т.к. 0 = A, 0.5 = B | Assylkhan | График «gap и accuracy vs p» |
| (опц.) background-only тест и Mixed-Rand фон из Pet | Islam | Доп. таблица |
| Начать отчёт: введение, related work, метод | Amina | Черновик разделов 1–3 |

### Неделя 4 — анализ и защита
| Задача | Кто | Результат |
|---|---|---|
| Анализ ошибок: какие породы больше всего зависят от фона, примеры flip | Tomiris | Графики per-class, confusion matrix |
| Проверка артефактов вставки: падение на картинках с плохими масками vs хорошими | Islam | Абзац в отчёте |
| Grad-CAM для A и B на одних и тех же картинках | Assylkhan | Сетка иллюстраций |
| Отчёт: результаты, обсуждение, ограничения | Amina | Финальный PDF |
| Презентация + `06_demo.ipynb` | Все | Слайды, демо работает «Run all» |
| Финальная проверка воспроизводимости: чистый Colab → README → результаты | Amina | Числа совпадают |

---

## 4. Главная таблица результатов (шаблон)

| Модель | Acc original | Acc swapped | Acc fg-only | F1 original | F1 swapped | Gap swap | Flip rate |
|---|---|---|---|---|---|---|---|
| HOG + SVM | | | | | | | |
| ResNet-18 A | ± | ± | ± | ± | ± | ± | ± |
| ResNet-18 B (p=0.5) | ± | ± | ± | ± | ± | ± | ± |

Читать gap только вместе с абсолютной accuracy: маленький gap у слабой модели ничего не доказывает.

---

## 5. Бюджет вычислений (Colab, T4)

- Train ≈ 3k изображений → одна эпоха ResNet-18 при 224 px с AMP ≈ 0.5–1.5 мин (если данные в `/content`).
- Один запуск 15 эпох ≈ 15–25 мин. A + B × 3 seeds = 6 запусков ≈ 2–2.5 ч.
- Ablation (+ p=0.25, p=1.0) × 3 seeds = ещё 6 запусков.
- Бесплатный Colab может отключаться → checkpoint каждую эпоху в Drive и возможность продолжить.

---

## 6. Риски

| Риск | Что делаем |
|---|---|
| Артефакты на краях масок сами роняют accuracy | Мягкая alpha, учёт границы; сравнить падение на «хороших» и «плохих» масках; fg-only как контроль |
| Фоны в тесте похожи на train-фоны | Разные пулы, лучше разные категории сцен |
| Colab отключается / лимиты GPU | Checkpoints в Drive, resume; запуски распределить между аккаунтами |
| Разные результаты у разных участников | Один код в `src/`, configs, фиксированные seeds, splits в git |
| Подглядывание в test | Test только в `04_evaluate`, после фиксации всех настроек |
| Не успеваем | Обязательно: A, B, HOG, 3 условия, 3 seeds. Опционально: ablation, background-only, Mixed-Rand, bootstrap |

---

## 7. Related work для отчёта

- Xiao, Engstrom, Ilyas, Madry. *Noise or Signal: The Role of Image Backgrounds in Object Recognition*, ICLR 2021 (ImageNet-9: Only-FG, Mixed-Same, Mixed-Rand) — ближайшая работа, на неё опираемся.
- Parkhi, Vedaldi, Zisserman, Jawahar. *Cats and Dogs*, CVPR 2012 — Oxford-IIIT Pet.
- Geirhos et al. *Shortcut Learning in Deep Neural Networks*, Nature MI 2020.
- Beery, Van Horn, Perona. *Recognition in Terra Incognita*, ECCV 2018 — зависимость от контекста/фона.
- Selvaraju et al. *Grad-CAM*, ICCV 2017.
- Dalal, Triggs. *HOG*, CVPR 2005; He et al. *ResNet*, CVPR 2016.

---

## 8. Открытые вопросы
- [ ] Дедлайн курса и формат сдачи
- [ ] Источник фонов: Places365 (рекомендация) или другой
- [ ] Граница trimap: включать в foreground или нет (решаем по примерам)
- [ ] Цвет нейтрального фона
- [ ] Чей Colab/Drive — основной для хранения данных и checkpoints
