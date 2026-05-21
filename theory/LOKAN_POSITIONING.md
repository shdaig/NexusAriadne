# Позиционирование LOKAN в литературе: NAS, дифференцируемые программы, символьная регрессия

---

## 1. Карта смежных исследовательских областей

LOKAN находится на пересечении четырёх потоков исследований, каждый из которых отвечает на свой вопрос:

```
Дифференцируемый NAS          Нейро-арифметические сети
"какую операцию применять      "могут ли нейросети выучить
  на каждом ребре?"              арифметические операции?"
          \                          /
           \                        /
            ──────── LOKAN ─────────
           /                        \
          /                          \
Символьная регрессия           KAN / интерпретируемые сети
"найти формулу из данных"       "обучаемые активации на рёбрах"
```

Ключевое отличие LOKAN от каждого из потоков: он не просто **применяет** операцию и не просто **ищет** формулу — он **обучает вероятность того, какие входы должны перемножаться**, сохраняя при этом полную дифференцируемость.

---

## 2. Дифференцируемый поиск архитектур (NAS)

### 2.1 Что происходит в области

**DARTS** (Liu et al., ICLR 2019) — основополагающая работа дифференцируемого NAS. Для каждого ребра в ячейке вычислительного графа вводится softmax-смесь из K операций-кандидатов (conv3×3, skip, none, dilated conv, ...):

```
ō(i→j)(x) = Σ_k  exp(α_k) / Σ_k' exp(α_k')  ·  o_k(x)
```

Параметры архитектуры α и веса сети W оптимизируются попеременно. После обучения — дискретизация через argmax по α.

**Проблемы DARTS**, вызвавшие огромный поток follow-up работ:
- Collapse к skip-connections (операция без параметров выигрывает из-за меньшего шума градиента)
- Несоответствие между найденной архитектурой (смесью) и реализованной (argmax)
- Высокая стоимость поиска

**Решения**: PC-DARTS (частичные каналы, ICLR 2020), GDAS (Gumbel-based дискретизация, CVPR 2019), SNAS (стохастический NAS, ICLR 2019), β-DARTS (регуляризация по collapse, CVPR 2022).

**Одноэтапный (one-shot) NAS**: суперсеть содержит все возможные архитектуры; поиск — sampling архитектур из суперсети. Масштабируется на пространства миллионов архитектур (NAS-Bench-201, OFA).

**Hardware-aware NAS** (MobileNets, EfficientNets, FBNet): пространство поиска сужено до операций, эффективных на целевом железе.

**Ключевое наблюдение**: весь NAS в deep learning ищет между одними и теми же "черными ящиками" (convolutional kernels, attention heads). Никто не ищет между математически интерпретируемыми операциями (+, ×) с теоретическими гарантиями аппроксимации.

### 2.2 Как позиционировать LOKAN

**Тезис**: LOKAN — это *интерпретируемый intra-layer NAS* для математических операций.

Отличия от DARTS:
| | DARTS | LOKAN |
|---|---|---|
| Пространство поиска | Произвольные нейронные блоки | Математически мотивированные: Σ и Π |
| Уровень поиска | Тип слоя / тип ребра (между слоями) | Тип агрегации внутри слоя |
| Теоретическая база | Нет (эмпирически эффективно) | Теорема КА + аппроксимация B-сплайнами |
| Интерпретируемость найденной архитектуры | Низкая | Высокая (граф Σ/Π узлов) |
| Дискретизация | argmax после обучения | τ → 0 (температурный отжиг) |
| Цель | Точность на downstream задачах | Точность + **обнаружение структуры** |

**Формулировка для Abstract**: "We present LOKAN, a differentiable operation search framework within KAN layers, where the routing of spline-transformed inputs into summation and multiplication groups is learned end-to-end via temperature-annealed softmax, analogous to differentiable NAS but over a mathematically grounded operation vocabulary."

---

## 3. Непрерывная релаксация дискретных выборов

### 3.1 Что происходит в области

**Gumbel-Softmax / Concrete Distribution** (Jang et al., ICLR 2017; Maddison et al., ICLR 2017):

Для сэмплирования из категориального распределения при обучении с обратным распространением ошибки вводится мягкая аппроксимация:

```
y_k = exp((log π_k + g_k) / τ) / Σ_j exp((log π_j + g_j) / τ)
```

где g_k ~ Gumbel(0,1). При τ → 0 распределение концентрируется на argmax — дискретный сэмпл. При τ = 1 — равномерная смесь.

**Straight-Through Estimator** (Bengio et al., 2013): на forward pass — argmax (дискретно), на backward pass — softmax (дифференцируемо). Используется в VQ-VAE, дискретных bottleneck архитектурах.

**Differentiable tree structures** (Popel & Bojar, 2018; Choi et al., 2018): релаксация дискретных деревьев для парсинга, структурного предсказания.

**Discrete VAE / Categorical VAE**: latent переменные как дискретные категории, обучаемые через Gumbel-Softmax. Применяется в DALL-E, VQ-VAE-2.

### 3.2 Как позиционировать LOKAN

LOKAN использует **температурную softmax** (аналог Gumbel-Softmax без шума Gumbel):

```
p_{i,o,g} = exp(l_{i,o,g} / τ) / Σ_{g'} exp(l_{i,o,g'} / τ)
```

с детерминированным экспоненциальным отжигом τ(t) = τ_start · (τ_end / τ_start)^{t/T}.

**Тезис**: LOKAN применяет идею Gumbel-Softmax к задаче обнаружения операционной структуры в символьной регрессии, где пространство операций определено теоремой Колмогорова–Арнольда.

Отличие от Gumbel-Softmax в стандартном применении:
1. Нет шума Gumbel (детерминированный отжиг — более стабильно для малых датасетов)
2. Пространство категорий фиксировано и теоретически мотивировано (G групп)
3. Выбор не бинарный — каждый вход может частично принадлежать группе

**Формулировка**: "LOKAN's operation routing is a structured instance of the Gumbel-Softmax relaxation, applied to the domain of mathematical operations and guided by the theoretical completeness of the {sum, product} basis."

---

## 4. Нейро-арифметические сети

### 4.1 Что происходит в области — самое близкое к LOKAN

**NALU (Neural Arithmetic Logic Units)**, Trask et al., NeurIPS 2018:

Специализированная нейронная ячейка для изучения точных арифметических операций:
```
a = Wz,    W = tanh(Ŵ) ⊙ σ(M̂)   (NAC — для сложения/вычитания)
y = exp(W · log|z + ε|)            (для умножения/деления)
y = g · a + (1-g) · m,  g = σ(Gz) (выбор между + и ×)
```

Ограничения NALU: нестабильное обучение при экстраполяции, логарифм не определён для отрицательных чисел, гарантии только для линейных комбинаций.

**NMU / NAU (Neural Multiplication/Addition Units)**, Madsen & Johansen, NeurIPS 2020:

Исправленная версия: NAU — реализация сложения с регуляризацией на {-1, 0, 1} веса; NMU — умножение через `∏ (1 - w + w·x)` — **та же самая формула, что и в LOKAN!**:

```python
# NMU (Madsen & Johansen, 2020):
y = ∏_i (1 - W_i + W_i · x_i),   W_i ∈ [0, 1]

# LOKAN term (2024):
term_{i,g} = 1 - p_{i,g} + p_{i,g} · ỹ_{i}
```

Это **прямая теоретическая связь**! LOKAN использует ту же "мягкую маску" произведения, что и NMU, но:
1. Применяет её к B-сплайн трансформациям (а не к сырым входам)
2. Обучает G параллельных групп одновременно
3. Объединяет с аддитивной группой через softmax маршрутизацию

**iNALU** (Schlör et al., 2020), **Real NVP**, **RealNVU**: продолжения NALU с нормализующими потоками.

**Neural Power Units** (Kochergin & Markovich, 2021): обобщение NALU на степенные функции.

**Текущее состояние**: NALU-подобные сети показали хорошие результаты на синтетических арифметических задачах, но плохо масштабируются и имеют проблемы с экстраполяцией. Главная открытая проблема — **нет теоретических гарантий аппроксимации**.

### 4.2 Как позиционировать LOKAN

**Тезис**: LOKAN решает открытую проблему NALU — обеспечивает теоретические гарантии аппроксимации (через связь с теоремой КА и B-сплайнами) при сохранении дифференцируемого обучения структуры.

| | NALU / NMU | LOKAN |
|---|---|---|
| Активации | Линейные / тождественные | B-сплайны (произвольные гладкие функции) |
| Теор. гарантии аппроксимации | Нет | Есть (O(G^{-(k+1)}) ошибка) |
| Экстраполяция | Плохая | Определяется выбором сплайна |
| Количество операций | 2 (+ и ×) | G групп (mix of products + 1 sum) |
| Интерпретируемость | Средняя (веса W) | Высокая (граф с B-сплайновыми кривыми) |
| Символьная регрессия | Ограничена | Полный pipeline |

**Формулировка**: "LOKAN generalizes the Neural Multiplication Unit (Madsen & Johansen, 2020) by replacing linear inputs with B-spline activations, extending the operation vocabulary from binary {sum, product} to G soft groups, and providing approximation-theoretic guarantees through the connection to the Kolmogorov-Arnold representation theorem."

**Важно процитировать явно**: Madsen & Johansen, NeurIPS 2020 — идентичная базовая формула.

---

## 5. Уравнение символьной регрессии: дифференцируемые подходы

### 5.1 Что происходит в области

**EQL (Equation Learner)**, Martius & Lampert, ICLR 2017:

Нейросеть, где каждый нейрон — одна из операций {sin, cos, ×, +, ...}. L0-регуляризация обнуляет неиспользуемые связи. Находит символьные формулы из данных через обычный SGD.

Проблема: каждый нейрон должен быть **заранее назначен** на одну операцию — нет дифференцируемого поиска.

**Deep Symbolic Regression (DSR)**, Petersen et al., ICLR 2021:

Рекуррентная сеть генерирует выражение в виде дерева (токен за токеном). Обучается через REINFORCE: дерево → вычислить R² → обновить политику.

Преимущество: находит компактные формулы.
Проблема: нестабильное обучение, комбинаторный взрыв пространства поиска.

**NeSymReS** (Biggio et al., NeurIPS 2021): трансформер обучается предсказывать символьное выражение по набору точек (meta-learning). Быстрый на inference, но требует огромный датасет для предобучения.

**PySR** (Cranmer, 2023): эволюционный алгоритм, Pareto-оптимальный фронт сложность/точность. Практически лучший на Feynman датасете.

**AI Feynman 2.0** (Udrescu et al., NeurIPS 2020): физически мотивированный поиск — рекурсивно тестирует сепарабельность, симметрии (Hessian-based), разложение по полиномам.

**Текущие тренды (2024–2025)**:
- **LLM-based SR**: GPT-4 / Code LLaMA генерирует Python-функции как гипотезы формул
- **Foundation models for SR**: предобученные на синтетических датасетах трансформеры (E2E Symbolic Regression, Vastl et al. 2024)
- **Differentiable SR**: D-SR, Kamienny et al. — конец к концу дифференцируемый, без RL
- **Физически информированная SR**: ограничения из размерного анализа, инвариантностей

### 5.2 Как позиционировать LOKAN

**Тезис**: LOKAN занимает уникальную нишу между нейросетевым фиттингом и символьной регрессией — он обучается как нейросеть (градиентный спуск, GPU), но его архитектура после обучения непосредственно читается как символьное выражение.

Сравнение подходов к символьной регрессии:

| Подход | Метод поиска | SR pipeline | Дифференцируемость |
|---|---|---|---|
| PySR | Эволюционный | Прямой → формула | ❌ |
| DSR | RL (REINFORCE) | Прямой → формула | Частично |
| EQL | SGD + L0 | Прямой → формула | ✅ |
| KAN / MultKAN | SGD | Обучить → прунинг → symbolify | ✅ |
| **LOKAN** | SGD + temperature | Обучить → **logit decode** → prune → symbolify | ✅ |

LOKAN добавляет промежуточный слой интерпретации: **logit decode до symbolify** уже показывает, какие входы перемножаются — это принципиально новый тип intermediate representation.

**Формулировка**: "Unlike EQL which pre-assigns operations to neurons, or PySR which searches combinatorially, LOKAN learns a continuous probability distribution over sum/product assignments jointly with the spline activations, collapsing to a discrete symbolic structure as temperature decreases — a gradient-based analog of symbolic structure discovery."

---

## 6. Mixture of Experts и маршрутизация

### 6.1 Что происходит в области

**Mixture of Experts (MoE)**, Jacobs et al. (1991); Shazeer et al. (2017):

Каждый токен/пример маршрутизируется к одному из K "экспертов" (отдельных нейросетей) через gating function:

```
y = Σ_k g_k(x) · E_k(x),    Σ_k g_k = 1
```

Sparse MoE (Shazeer 2017) — top-k маршрутизация, только 1–2 эксперта активны. Масштабируется до 1.7T параметров (Switch Transformer, Mixtral).

**Ключевые проблемы**: load balancing (неравномерное использование экспертов), training instability.

**Soft MoE** (Google, 2023): мягкая маршрутизация без top-k, более стабильна.

**MoE для структурного обучения**: Routing Networks (Rosenbaum et al., 2018) — маршрутизация разных примеров через разные подсети.

### 6.2 Как позиционировать LOKAN

**Тезис**: LOKAN — это специализированный MoE для математических операций, где "эксперты" — это группы произведения, а "примеры" — это входные переменные, а не батч-элементы.

Принципиальное отличие от стандартного MoE:
- В MoE каждый **пример** маршрутизируется к эксперту (маршрутизация по оси батча)
- В LOKAN каждая **переменная входа** маршрутизируется в операционную группу (маршрутизация по оси признаков)
- Эксперт в LOKAN — не нейросеть, а математическая операция (произведение)

**Формулировка**: "LOKAN can be interpreted as a feature-wise Mixture of Experts where experts are algebraic operations — multiplication groups and a summation group — and routing probabilities are shared across the entire dataset, learning the structural properties of the target function rather than its instance-specific complexity."

---

## 7. Нейро-символьная интеграция (Neuro-Symbolic AI)

### 7.1 Что происходит в области

**Neural Module Networks (NMN)**, Andreas et al. (2016): разные части вопроса направляются к разным нейронным модулям (нашедший объект, ответ на вопрос о цвете, ...). Структура модулей задаётся парсером.

**Differentiable Inductive Logic Programming (dILP)**, Evans & Grefenstette (2018): мягкая логика поверх нейронных представлений.

**Neural Theorem Provers**, **AlphaGeometry** (2024): нейросеть + символьный решатель в связке.

**Program Synthesis через обучение**: DreamCoder (Ellis et al., 2021) — обучается библиотеке функций и правилам их комбинирования; μP/Poirot — нейросеть обнаруживает программные инварианты.

**Текущий тренд**: нейро-символьные системы начинают доминировать в научных задачах — AlphaFold + AlphaCode + AlphaGeometry показывают, что символьная структура + нейросетевое обучение сильнее, чем каждый подход по отдельности.

### 7.2 Как позиционировать LOKAN

**Тезис**: LOKAN реализует идею neuro-symbolic интеграции для математических законов: нейронная часть (B-сплайны) обнаруживает форму одномерных зависимостей, символьная часть (logit routing + temperature annealing) обнаруживает операционную структуру.

```
Нейронная часть:   ỹ_{i,o} = w_b · SiLU(x_i) + w_s · spline(x_i)
                                 ↕ градиент
Символьная часть:  p_{i,o,g} = softmax(l_{i,o,g} / τ)
                                 ↕ температурный отжиг
Результат:         Граф с явными Σ/Π узлами, B-сплайн кривые на рёбрах
```

---

## 8. Как преподнести LOKAN: три нарратива

В зависимости от целевой конференции/журнала LOKAN можно позиционировать тремя способами:

### Нарратив 1: "Дифференцируемый NAS для символьных операций"
**Целевая аудитория**: NeurIPS, ICML (Machine Learning трек)

Акцент на связи с DARTS, Gumbel-Softmax. Показать, что LOKAN — это NAS для математических операций с теоретическими гарантиями. Ключевые эксперименты: сравнение с DARTS на symbolic regression benchmarks, ablation на τ-schedule.

**Слабость этого нарратива**: NAS-аудитория не интересуется интерпретируемостью, и будет сравнивать LOKAN с современными NAS методами по скорости поиска, где LOKAN проиграет.

### Нарратив 2: "Learnable aggregation в KAN для обнаружения мультипликативных структур"
**Целевая аудитория**: ICLR, тематические воркшопы (AI4Science, KAN extensions)

Акцент на принципиальном расширении KAN / MultKAN. LOKAN не требует знания структуры до обучения. Ключевые эксперименты: staged SR benchmark, Feynman dataset, символьная регрессия с операционным графом.

**Сильная сторона**: прямое сравнение с MultKAN возможно; чёткое преимущество в автоматическом обнаружении структуры.

### Нарратив 3: "Gradient-based symbolic structure discovery с аппроксимационными гарантиями"
**Целевая аудитория**: Physical Review Letters (letters), Science Advances, Nature Machine Intelligence

Акцент на реальных научных открытиях. Один физический пример (закон сохранения, формула Фейнмана) + теоретический результат. Аудитория физиков и учёных, не NAS-инженеров.

**Сильная сторона**: высокий импакт если показать открытие чего-то нетривиального.

### Рекомендуемый нарратив для первой публикации

**Нарратив 2** — наименьший риск и наиболее конкретный технический вклад. Нарратив 3 — цель через 6–12 месяцев.

---

## 9. Структура Related Works для публикации

```
## Related Work

### Kolmogorov-Arnold Networks
[KAN 1.0 Liu 2024] ... [KAN 2.0 / MultKAN Liu 2024] ...
Наш вклад: LOKAN заменяет ручное задание mult-узлов на автоматически
обучаемую маршрутизацию.

### Differentiable Architecture Search
[DARTS Liu 2019] ... [SNAS Xie 2019] ... [PC-DARTS Xu 2020] ...
Наш вклад: LOKAN применяет идею непрерывной релаксации операций
к внутрислойной агрегации с математически мотивированным пространством поиска.

### Continuous Relaxations of Discrete Choices
[Gumbel-Softmax Jang 2017] [Concrete Maddison 2017] [STE Bengio 2013]
Наш вклад: детерминированный температурный отжиг более стабилен
для малых датасетов символьной регрессии.

### Neural Arithmetic Units
[NALU Trask 2018] [NMU Madsen 2020] [NAU Madsen 2020]
Наш вклад: LOKAN обобщает NMU на B-сплайновые активации
и предоставляет теоретические гарантии аппроксимации.

### Symbolic Regression
[EQL Martius 2016] [DSR Petersen 2021] [PySR Cranmer 2023] [NeSymReS Biggio 2021]
Наш вклад: LOKAN занимает промежуточное положение —
обучается как нейросеть, читается как символьное выражение.

### Mixture of Experts
[Shazeer 2017] [Soft MoE 2023]
Наш вклад: feature-wise MoE, где эксперты — алгебраические операции.
```

---

## 10. Конкретные цитируемые работы

### Обязательные (P0)

| Работа | Связь с LOKAN |
|---|---|
| Liu et al. 2024 (KAN, arXiv:2404.19756) | Базовая архитектура |
| Liu et al. 2024 (KAN 2.0, arXiv:2408.10205) | Прямой конкурент / дополнение |
| **Madsen & Johansen 2020 (NMU, NeurIPS)** | **Идентичная формула term = 1−W+W·x** |
| Trask et al. 2018 (NALU, arXiv:1808.00508) | Предшественник нейро-арифметики |
| Jang et al. 2017 (Gumbel-Softmax, ICLR) | Основа температурной релаксации |
| Maddison et al. 2017 (Concrete, ICLR) | То же |
| Liu et al. 2019 (DARTS, ICLR) | NAS контекст |

### Важные (P1)

| Работа | Связь с LOKAN |
|---|---|
| Martius & Lampert 2016 (EQL, ICLR) | Символьная регрессия через нейросеть |
| Petersen et al. 2021 (DSR, ICLR) | SR baseline |
| Cranmer 2023 (PySR) | SR baseline |
| Bengio et al. 2013 (STE) | Дискретизация через gradient |
| Girosi & Poggio 1989 (irrelevance of KA theorem) | Контекст для опровержения скептицизма |

### Желательные (P2)

| Работа | Связь |
|---|---|
| Shazeer et al. 2017 (Sparse MoE) | MoE контекст |
| Udrescu et al. 2020 (AI Feynman 2.0) | Конкурент в физическом SR |
| Biggio et al. 2021 (NeSymReS) | Meta-learning для SR |
| Kolmogorov 1957 (теорема) | Исторический контекст |
| Lorentz 1976 (о негладкости в теореме КА) | Обоснование перехода к B-сплайнам |

---

## 11. Резюме: как звучит LOKAN для каждой аудитории

**Для NAS-аудитории (NeurIPS/ICML)**:
> "We propose differentiable operation routing in KAN layers — a form of intra-layer NAS over the mathematically complete vocabulary {sum, product}. Unlike DARTS which searches over black-box neural blocks, LOKAN's search space is grounded in the Kolmogorov-Arnold theorem, yielding interpretable networks with approximation guarantees."

**Для символьной регрессии (GECCO/GPTP)**:
> "LOKAN bridges the gap between neural fitting and symbolic regression: it trains end-to-end like a neural network while learning a discrete sum/product routing that directly encodes the symbolic structure of the target function, without evolutionary search or reinforcement learning."

**Для KAN-аудитории (AI4Science воркшопы)**:
> "LOKAN extends MultKAN by replacing manual specification of multiplication nodes with automatic structure discovery: the routing of spline-transformed features into sum and product groups is learned jointly via temperature-annealed softmax, revealing multiplicative dependencies without prior knowledge."

**Для физиков (Physical Review / Nature)**:
> "We present a neural network that automatically discovers whether physical variables combine additively or multiplicatively — learning not just the shape of the dependence but its algebraic structure — providing a new tool for data-driven derivation of physical laws."
