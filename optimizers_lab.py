# -*- coding: utf-8 -*-
"""
Optimizers Lab — учебный симулятор: как оптимизаторы превращают градиент в шаг.

Проект 11 из серии учебных симуляторов по машинному обучению.

Один и тот же холм, одна и та же стартовая точка, одна и та же скорость
обучения — и пять разных способов обновить параметры:

    SGD → Momentum → AdaGrad → RMSProp → Adam

    Потери → градиент → оптимизатор → обновление параметров → новое состояние

Никакого машинного обучения, датасетов и сторонних библиотек: вся математика
написана вручную на стандартной библиотеке Python (tkinter, math). Каждый шаг
показывается с подстановками, так что любую строку можно проверить на бумаге.
Язык интерфейса переключается на ходу: русский (по умолчанию) или английский.

Автор: SteveWorks001

Куда смотреть:
  1. Настройки, пределы и палитра .... DEFAULT_*, *_MIN/_MAX, COL_*
  2. Обозначения и числа ............. sub(), sup(), f4(), fs(), fmt()
  3. Целевые функции ................. class Quadratic / class Rosenbrock
  4. Оптимизаторы .................... class Optimizer и пять наследников
  5. Движок прогона .................. class StepRecord / class Trace
  6. Локализация ..................... LANG / TEXTS / T()
  7. Справка и эксперименты .......... HELP / PRESETS / EXP_TEXTS
  8. Виджеты ......................... NumberControl, ContourView, LossChart
  9. Окно ............................ class OptimizersLabApp
 10. Точка входа ..................... main()
"""

import math
import tkinter as tk
from tkinter import ttk, font as tkfont

# ---------------------------------------------------------------
# 1. Настройки, пределы и цветовая палитра
# ---------------------------------------------------------------
# Пределы у всех полей маленькие и осмысленные: поле не должно позволять
# собрать эксперимент, который ничего не объясняет.
DEFAULT_ETA = 0.250
ETA_MIN, ETA_MAX = 0.002, 0.500

DEFAULT_BETA = 0.900           # β у Momentum, RMSProp и β₁ у Adam — один на всех
BETA_MIN, BETA_MAX = 0.000, 0.990

DEFAULT_BETA2_K = 3            # β₂ задаётся как 1 − 10⁻ᵏ, k = 1…4
BETA2_K_MIN, BETA2_K_MAX = 1, 4

DEFAULT_EPS_K = 8              # ε задаётся как 10⁻ᵏ, k = 1…12
EPS_K_MIN, EPS_K_MAX = 1, 12

DEFAULT_KAPPA = 2.0            # число обусловленности a:b для квадратичной функции
KAPPA_MIN, KAPPA_MAX = 1.0, 40.0
QUAD_A = 0.5                   # кривизна по x держим постоянной: меняется только b
DEFAULT_THETA0 = (3.0, 1.8)    # стартовая точка по умолчанию (общая для всех методов)

START_SPAN = 5.0               # стартовую точку холма можно выбрать в ±5 от его центра
MAX_STEPS = 600                # предел шагов на один прогон
BLOWUP_RADIUS = 1.0e6          # дальше этого координаты считались улетевшими

TOL_DIST = 5.0e-3              # ||θ − θ*|| — насколько близко к минимуму
TOL_GRAD = 5.0e-3              # ||∇L|| — насколько плоское место
TOL_STALL = 1.0e-12            # прирост потерь меньше этого — шаг ничего не дал

AUTO_DELAY_MS = 120            # пауза между итерациями в автоматическом режиме
FAST_BATCH = 40                # сколько шагов делает быстрый прогон за один заход

MINUS = "−"                    # настоящий математический минус (U+2212)
DOT = "·"                      # знак умножения в формулах

# Цвет оптимизаторов: пять хорошо различимых оттенков на светлом фоне.
COL_TEXT = "#242424"
COL_BG = "#fbfbf7"
COL_PAPER = "#fdfdf9"
COL_FRAME = "#c8c8c0"
COL_LINE = "#8d949b"
COL_GRID = "#ededdf"
COL_DIM = "#8a8a82"
COL_HEAD = "#31566f"
COL_KEY = "#b07d10"
COL_MIN = "#2f7d4f"
COL_START = "#4a4a4a"
COL_BAD_BG = "#fde8e6"
COL_BAD_FG = "#7a1a13"
COL_AXIS_TEXT = "#5a5a54"        # подписи делений: темнее сетки, светнее текста

# Подпись автора — одно место в программе: она же в углу окна, она же в README.
AUTHOR = "SteveWorks001"

OPT_COLORS = {
    "sgd": "#2f6f9f",
    "momentum": "#c8651b",
    # Оливковый, а не зелёный: крест минимума тоже зелёный, и на одном холсте
    # две одинаковые зелёные линии слились бы в одну.
    "adagrad": "#5f7d1a",
    "rmsprop": "#5b3a86",
    "adam": "#b0305f",
}
OPT_ORDER = ("sgd", "momentum", "adagrad", "rmsprop", "adam")

# ---------------------------------------------------------------
# 2. Обозначения и форматирование чисел
# ---------------------------------------------------------------
SUBS = "₀₁₂₃₄₅₆₇₈₉"
SUPS = "⁰¹²³⁴⁵⁶⁷⁸⁹"


def sub(v):
    """Подстрочные цифры: sub(1) -> «₁»."""
    return "".join(SUBS[int(c)] if c.isdigit() else c for c in str(v))


def sup(v):
    """Надстрочные цифры: sup(12) -> «¹²»."""
    return "".join(SUPS[int(c)] if c.isdigit() else c for c in str(v))


# Индексы в формулах. Символом «t» пользуемся постоянно, поэтому они вынесены
# в константы: в исходнике формула читается так же, как в учебнике.
TH = "ₜ"                 # ₜ
TH1 = "ₜ₊₁"        # ₜ₊₁
THM1 = "ₜ₋₁"       # ₜ₋₁


def exp_text_num(v):
    """«3.00 · 10⁻⁹»:mantissa и настоящая надстрочная степень вместо «e-09»."""
    mant, exp = f"{v:.2e}".split("e")
    power = int(exp)
    sign = "⁻" if power < 0 else ""
    return f"{mant.replace('-', MINUS)} · 10{sign}{sup(abs(power))}"


def f4(v):
    """Число для формул: четыре знака, настоящий минус, ноль без «−0.0000»."""
    if v is None or not math.isfinite(v):
        return "∞"
    if abs(v) < 5e-9:
        v = 0.0
    return f"{v:.4f}".replace("-", MINUS)


def fs(v):
    """Число с плавающей точкой для состояний оптимизатора: 3e−09 вместо 0.0000.

    AdaGrad накапливает G, а ε по умолчанию 10⁻⁸: обычная четырёхзначная
    печать превратила бы такие величины в «0.0000» и скрыла смысл шага.
    """
    if v is None or not math.isfinite(v):
        return "∞"
    a = abs(v)
    if a != 0.0 and (a < 1e-3 or a >= 1e6):
        return exp_text_num(v)
    if a < 5e-9:
        v = 0.0
    return f"{v:.4f}".replace("-", MINUS)


def fsg(v):
    """Шесть значащих знаков — для строк, где читатель делит напечатанные числа.

    Поправка Adam делит m на 1 − β₁ᵗ, а 1 − β₂ᵗ на первых шагах порядка 10⁻³:
    четырёх знаков после запятой не хватает, и строка «2.2500 = 0.0023 / 0.0010»
    перестала бы сходиться.
    """
    if v is None or not math.isfinite(v):
        return "∞"
    a = abs(v)
    if a != 0.0 and (a < 1e-4 or a >= 1e7):
        return exp_text_num(v)
    return f"{v:.6g}".replace("-", MINUS)


def fmt(v):
    """Число для поля ввода: его можно напечатать и прочитать обратно float()."""
    if abs(v) < 1e-12:
        return "0"
    if abs(v - round(v)) < 1e-10 and abs(v) < 1e15:
        return str(int(round(v)))
    return f"{v:.6g}"


def tick_text(v):
    """Подпись деления оси: настоящий минус, без лишних знаков."""
    if v != 0.0 and abs(v) < 1e-3:
        return exp_text_num(v)
    return fmt(v).replace("-", MINUS)


def pow10_text(k):
    """Степень десятки: 10⁻⁸ (для ε) и 1 − 10⁻ᵏ (для β₂)."""
    return "10" + ("⁻" if k > 0 else "") + sup(abs(k))


def level_text(v):
    """Подпись уровня изолинии: чем крупнее уровень, тем меньше знаков.

    Метки лежат близко друг к другу, и «1163.37» длиной в семь знаков загораживал
    соседнюю подпись.
    """
    a = abs(v)
    if a >= 1e5:
        return exp_text_num(v)
    if a >= 1000.0:
        t = f"{v:.0f}"
    elif a >= 10.0:
        t = f"{v:.1f}"
    elif a >= 0.01:
        t = f"{v:.2f}"
    else:
        return exp_text_num(v)
    return t.replace("-", MINUS)


def signed(v):
    """Отрицательное число в множителе берём в скобки, чтобы не слилось с минусом."""
    return f"({f4(v)})" if v < -5e-9 else f4(v)


def fs_signed(v):
    return f"({fs(v)})" if v < -5e-9 else fs(v)


def clamp(v, lo, hi):
    return lo if v < lo else hi if v > hi else v


def dist_to(theta, target):
    """Евклидово расстояние ||θ − θ*|| между двумя точками на плоскости."""
    dx = theta[0] - target[0]
    dy = theta[1] - target[1]
    return math.hypot(dx, dy)


def norm2(v):
    return math.hypot(v[0], v[1])


# ---------------------------------------------------------------
# 3. Целевые функции: одна задача для всех пяти оптимизаторов
#    Формулы записаны математическими символами, поэтому не переводятся.
# ---------------------------------------------------------------
class Objective:
    """Оптимизируемая функция: значения, аналитический градиент и минимум.

    Градиент задан формулой, а не разностным отношением: численные оценки
    добавили бы погрешность, и сравнивать оптимизаторы стало бы некорректно.
    """

    key = ""
    formula = ""
    deriv_text = ""
    state_note = ""
    center = (0.0, 0.0)
    radius = 5.0
    start_span = 5.0           # поле выбора стартовой точки: центр ± start_span

    def loss(self, x, y):
        raise NotImplementedError

    def grad(self, x, y):
        raise NotImplementedError

    @property
    def min_theta(self):
        return self.center

    @property
    def min_loss(self):
        return self.loss(*self.center)

    def loss_at(self, theta):
        return self.loss(theta[0], theta[1])

    def grad_at(self, theta):
        return self.grad(theta[0], theta[1])

    def levels(self, count, vmax):
        """Уровни изолиний: квадратичная сетка значений, чтобы эллипсы не слипались.

        Между соседними уровнями расстояние по значению растёт; для холма с
        квадратичным ростом это даёт примерно равные промежутки на плоскости.
        """
        vmin = self.min_loss
        if not math.isfinite(vmax) or vmax <= vmin:
            return []
        out = []
        for i in range(1, count + 1):
            frac = i / count
            out.append(vmin + (vmax - vmin) * frac * frac)
        return out


class Quadratic(Objective):
    """Анизотропный холм L(x, y) = ½·a·x² + ½·b·y² с минимумом в (0, 0).

    a держим постоянным, b = a·κ: при κ > 1 направление y «круче», и обычной
    спуск по нему перемахивает через долину, а вдоль пологого x идёт медленно.
    Именно эта растянутость и отличает оптимизаторы друг от друга.
    """

    key = "quad"

    def __init__(self, kappa=DEFAULT_KAPPA):
        self.kappa = kappa
        self.a = QUAD_A
        self.b = QUAD_A * kappa
        self.formula = f"L(x, y) = ½·{fmt(self.a)}·x² + ½·{fmt(self.b)}·y²"
        self.deriv_text = (f"∇L = ({fmt(self.a)}·x, {fmt(self.b)}·y)")
        self.state_note = f"κ = b/a = {fmt(kappa)}"

    def loss(self, x, y):
        return 0.5 * self.a * x * x + 0.5 * self.b * y * y

    def grad(self, x, y):
        return (self.a * x, self.b * y)

    def set_radius_for(self, theta):
        """Кадр по стартовой точке: траектория обязана влезать целиком."""
        reach = max(abs(theta[0]), abs(theta[1]), 1.0)
        self.radius = reach * 1.35 + 0.5


class Rosenbrock(Objective):
    """Долина Розенброка L(x, y) = (1 − x)² + c·(y − x²)², минимум в (1, 1).

    Узкая изогнутая долина: спуск вниз по склону быстрый, а движение вдоль
        дна — медленное, и направление градиента всё время врёт. Классическая
    проверка оптимизатора, и она же — причина, по которой Momentum и Adam
    не «лучше», а просто ведут себя по-другому.
    """

    key = "rosen"

    def __init__(self, c=2.0):
        self.c = c
        self.center = (1.0, 1.0)
        self.radius = 2.8
        self.start_span = 2.6
        self.formula = f"L(x, y) = (1 − x)² + {fmt(c)}·(y − x²)²"
        self.deriv_text = ("∇L = (−2(1 − x) − 4c·x(y − x²), 2c(y − x²))"
                           .replace("4c", fmt(4 * c)).replace("2c", fmt(2 * c)))

    def loss(self, x, y):
        t = y - x * x
        return (1.0 - x) ** 2 + self.c * t * t

    def grad(self, x, y):
        t = y - x * x
        return (-2.0 * (1.0 - x) - 4.0 * self.c * x * t, 2.0 * self.c * t)

    def set_radius_for(self, theta):
        need = max(dist_to(theta, self.center), 1.6)
        self.radius = need * 1.30 + 0.4


OBJECTIVES = {"quad": Quadratic, "rosen": Rosenbrock}

# ---------------------------------------------------------------
# 4. Оптимизаторы: пять способов превратить градиент в обновление
# ---------------------------------------------------------------
class Hyper:
    """Гиперпараметры прогона. η и β общие для всех пяти методов — иначе
    сравнение теряет смысл: сравнивать надо разницу оптимизаторов, а не
    разницу настроек."""

    def __init__(self, eta=DEFAULT_ETA, beta=DEFAULT_BETA,
                 beta2_k=DEFAULT_BETA2_K, eps_k=DEFAULT_EPS_K):
        self.eta = eta
        self.beta = beta
        self.beta2_k = beta2_k
        self.eps_k = eps_k

    @property
    def beta2(self):
        return 1.0 - 10.0 ** (-self.beta2_k)

    @property
    def eps(self):
        return 10.0 ** (-self.eps_k)

    def signature(self):
        return (round(self.eta, 12), round(self.beta, 12), self.beta2_k, self.eps_k)


class StepRecord:
    """Один полный шаг оптимизатора: все промежуточные числа остаются видимыми."""

    def __init__(self, t, theta, loss, grad, hp):
        self.t = t                        # номер итерации, начиная с 1
        self.theta = theta                # θₜ — параметры ДО шага
        self.loss = loss                  # L(θₜ)
        self.grad = grad                  # gₜ = ∇L(θₜ)
        self.hp = hp
        self.new_theta = theta            # θₜ₊₁ — заполняет оптимизатор
        self.new_loss = loss
        self.delta = (0.0, 0.0)           # Δθ = θₜ₊₁ − θₜ
        self.trace = []                   # [(kind, строка)]: f — формула, s — подстановка
        self.dist = 0.0
        self.dist_new = 0.0

    @property
    def grad_norm(self):
        return norm2(self.grad)


class Optimizer:
    """Общая схема оптимизатора.

    Каждый метод — это своя память о прошлых градиентах. Поэтому у класса есть
    reset(): состояние обязана обнуляться при любом перезапуске эксперимента, а
    не переезжать из одного оптимизатора в другой.
    """

    key = ""
    chain = ("chain_grad", "chain_update")
    state_keys = ()

    def __init__(self, objective, hp):
        self.obj = objective
        self.hp = hp
        self.t = 0
        self.reset()

    def reset(self):
        self.t = 0

    # ---------- вспомогательное для подстановок ----------
    def _update(self, rec, new_theta):
        """Оформить конец шага: Δθ, новые потери и расстояния до минимума."""
        rec.new_theta = new_theta
        rec.delta = (new_theta[0] - rec.theta[0], new_theta[1] - rec.theta[1])
        rec.new_loss = self.obj.loss_at(new_theta)
        target = self.obj.min_theta
        rec.dist = dist_to(rec.theta, target)
        rec.dist_new = dist_to(new_theta, target)

    def advance(self, rec):
        """Ровно одна итерация. Возвращает False, если считать дальше нельзя."""
        raise NotImplementedError

    def state_lines(self):
        """Внутреннее состояние после текущего шага: [(ключ TEXTS, вектор)]."""
        return []

    def bad(self, *values):
        """Числа вышли за пределы разумного: шаг нельзя показывать как настоящий."""
        for v in values:
            if not math.isfinite(v) or abs(v) > BLOWUP_RADIUS:
                return True
        return False

    # ---------- печать подстановок ----------
    def line(self, rec, kind, text):
        rec.trace.append((kind, text))

    def comp(self, rec, name, value, rhs, num=fs):
        """Строка вида «x: 1.7500 = 0.9000 · 0.0000 + 1.7500»."""
        self.line(rec, "s", f"     {name}: {num(value)} = {rhs}")

    def head(self, rec, formula):
        self.line(rec, "f", formula)


class Sgd(Optimizer):
    """SGD: θₜ₊₁ = θₜ − η·gₜ. Градиент используется напрямую, без памяти."""

    key = "sgd"
    chain = ("chain_grad", "chain_update")

    def advance(self, rec):
        h = self.hp
        gx, gy = rec.grad
        self.t += 1
        try:
            nx = rec.theta[0] - h.eta * gx
            ny = rec.theta[1] - h.eta * gy
        except OverflowError:
            return False
        if self.bad(nx, ny):
            return False
        self.head(rec, f"θ{TH1} = θ{TH} − η·g{TH}")
        self.line(rec, "s", f"     η = {f4(h.eta)}")
        self.comp(rec, "x", nx, f"{f4(rec.theta[0])} − {f4(h.eta)} · {signed(gx)}")
        self.comp(rec, "y", ny, f"{f4(rec.theta[1])} − {f4(h.eta)} · {signed(gy)}")
        self._update(rec, (nx, ny))
        return True


class Momentum(Optimizer):
    """Momentum: vₜ = β·vₜ₋₁ + gₜ, θₜ₊₁ = θₜ − η·vₜ.

    Соглашение записано явно: η стоит только в обновлении параметров, а не в
    накоплении скорости. От этого зависит эффективный размер шага, и в справке
    это единственный пункт, где методы расходятся в записи чаще всего.
    """

    key = "momentum"
    chain = ("chain_grad", "chain_velocity", "chain_update")
    state_keys = ("st_velocity",)

    def reset(self):
        super().reset()
        self.v = (0.0, 0.0)

    def advance(self, rec):
        h = self.hp
        gx, gy = rec.grad
        self.t += 1
        vx = h.beta * self.v[0] + gx
        vy = h.beta * self.v[1] + gy
        try:
            nx = rec.theta[0] - h.eta * vx
            ny = rec.theta[1] - h.eta * vy
        except OverflowError:
            return False
        if self.bad(vx, vy, nx, ny):
            return False
        self.head(rec, f"v{TH} = β·v{THM1} + g{TH}")
        self.line(rec, "s", f"     β = {f4(h.beta)}")
        self.comp(rec, "x", vx, f"{f4(h.beta)} · {fs(self.v[0])} + {fs(gx)}")
        self.comp(rec, "y", vy, f"{f4(h.beta)} · {fs(self.v[1])} + {fs(gy)}")
        self.head(rec, f"θ{TH1} = θ{TH} − η·v{TH}")
        self.comp(rec, "x", nx, f"{f4(rec.theta[0])} − {f4(h.eta)} · {signed(vx)}")
        self.comp(rec, "y", ny, f"{f4(rec.theta[1])} − {f4(h.eta)} · {signed(vy)}")
        self.v = (vx, vy)
        self._update(rec, (nx, ny))
        return True

    def state_lines(self):
        return [("st_velocity", self.v)]


class AdaGrad(Optimizer):
    """AdaGrad: Gₜ = Gₜ₋₁ + gₜ², θₜ₊₁ = θₜ − η·gₜ/(√Gₜ + ε).

    Накопленная сумма квадратов градиентов только растёт, поэтому шаг по
    часто обновляемому направлению уменьшается со временем — вплоть до остановки.
    """

    key = "adagrad"
    chain = ("chain_grad", "chain_accumulate", "chain_update")
    state_keys = ("st_accum",)

    def reset(self):
        super().reset()
        self.g = (0.0, 0.0)

    def advance(self, rec):
        h = self.hp
        gx, gy = rec.grad
        self.t += 1
        ax = self.g[0] + gx * gx
        ay = self.g[1] + gy * gy
        dx, dy = math.sqrt(ax) + h.eps, math.sqrt(ay) + h.eps
        try:
            nx = rec.theta[0] - h.eta * gx / dx
            ny = rec.theta[1] - h.eta * gy / dy
        except (OverflowError, ZeroDivisionError):
            return False
        if self.bad(ax, ay, nx, ny):
            return False
        self.head(rec, f"G{TH} = G{THM1} + (g{TH})²")
        self.comp(rec, "x", ax, f"{fs(self.g[0])} + ({fs(gx)})²")
        self.comp(rec, "y", ay, f"{fs(self.g[1])} + ({fs(gy)})²")
        self.head(rec, f"θ{TH1} = θ{TH} − η·g{TH}/(√G{TH} + ε)")
        self.line(rec, "s", f"     ε = {pow10_text(h.eps_k)}")
        self.comp(rec, "x", nx,
                  f"{f4(rec.theta[0])} − {f4(h.eta)} · {signed(gx)}"
                  f" / (√{fs(ax)} + {pow10_text(h.eps_k)})")
        self.comp(rec, "y", ny,
                  f"{f4(rec.theta[1])} − {f4(h.eta)} · {signed(gy)}"
                  f" / (√{fs(ay)} + {pow10_text(h.eps_k)})")
        self.g = (ax, ay)
        self._update(rec, (nx, ny))
        return True

    def state_lines(self):
        return [("st_accum", self.g)]


class RmsProp(Optimizer):
    """RMSProp: sₜ = β·sₜ₋₁ + (1 − β)·gₜ², θₜ₊₁ = θₜ − η·gₜ/(√sₜ + ε).

    Та же идея, что у AdaGrad, но среднее экспоненциальное: старые градиенты
    забываются, и шаг перестаёт необратимо ужаться.
    """

    key = "rmsprop"
    chain = ("chain_grad", "chain_average", "chain_update")
    state_keys = ("st_avg",)

    def reset(self):
        super().reset()
        self.s = (0.0, 0.0)

    def advance(self, rec):
        h = self.hp
        gx, gy = rec.grad
        self.t += 1
        b = h.beta
        sx = b * self.s[0] + (1.0 - b) * gx * gx
        sy = b * self.s[1] + (1.0 - b) * gy * gy
        dx, dy = math.sqrt(sx) + h.eps, math.sqrt(sy) + h.eps
        try:
            nx = rec.theta[0] - h.eta * gx / dx
            ny = rec.theta[1] - h.eta * gy / dy
        except OverflowError:
            return False
        if self.bad(sx, sy, nx, ny):
            return False
        self.head(rec, f"s{TH} = β·s{THM1} + (1 − β)·(g{TH})²")
        self.line(rec, "s", f"     β = {f4(b)}, 1 − β = {f4(1.0 - b)}")
        self.comp(rec, "x", sx, f"{f4(b)} · {fs(self.s[0])} + {f4(1.0 - b)} · ({fs(gx)})²")
        self.comp(rec, "y", sy, f"{f4(b)} · {fs(self.s[1])} + {f4(1.0 - b)} · ({fs(gy)})²")
        self.head(rec, f"θ{TH1} = θ{TH} − η·g{TH}/(√s{TH} + ε)")
        self.line(rec, "s", f"     ε = {pow10_text(h.eps_k)}")
        self.comp(rec, "x", nx,
                  f"{f4(rec.theta[0])} − {f4(h.eta)} · {signed(gx)}"
                  f" / (√{fs(sx)} + {pow10_text(h.eps_k)})")
        self.comp(rec, "y", ny,
                  f"{f4(rec.theta[1])} − {f4(h.eta)} · {signed(gy)}"
                  f" / (√{fs(sy)} + {pow10_text(h.eps_k)})")
        self.s = (sx, sy)
        self._update(rec, (nx, ny))
        return True

    def state_lines(self):
        return [("st_avg", self.s)]


class Adam(Optimizer):
    """Adam: два скользящих среднем + поправка на смещение.

    m̂ₜ и v̂ₜ нужны потому, что m₀ = v₀ = 0: без деления на (1 − βᵗ) первые
    шаги были бы искусственно маленькими, и на коротких запусках Adam
    выглядел бы медленнее, чем он является на самом деле.
    """

    key = "adam"
    chain = ("chain_grad", "chain_moment1", "chain_moment2",
             "chain_bias", "chain_update")
    state_keys = ("st_moment1", "st_moment2", "st_step")

    def reset(self):
        super().reset()
        self.m = (0.0, 0.0)
        self.v = (0.0, 0.0)

    def advance(self, rec):
        h = self.hp
        gx, gy = rec.grad
        self.t += 1
        b1, b2 = h.beta, h.beta2
        mx = b1 * self.m[0] + (1.0 - b1) * gx
        my = b1 * self.m[1] + (1.0 - b1) * gy
        vx = b2 * self.v[0] + (1.0 - b2) * gx * gx
        vy = b2 * self.v[1] + (1.0 - b2) * gy * gy
        bc1 = 1.0 - b1 ** self.t
        bc2 = 1.0 - b2 ** self.t
        if bc1 <= 0.0 or bc2 <= 0.0:
            return False
        mxh, myh = mx / bc1, my / bc1
        vxh, vyh = vx / bc2, vy / bc2
        dx, dy = math.sqrt(vxh) + h.eps, math.sqrt(vyh) + h.eps
        try:
            nx = rec.theta[0] - h.eta * mxh / dx
            ny = rec.theta[1] - h.eta * myh / dy
        except OverflowError:
            return False
        if self.bad(mx, my, vx, vy, mxh, myh, nx, ny):
            return False
        self.head(rec, f"m{TH} = β₁·m{THM1} + (1 − β₁)·g{TH}")
        self.line(rec, "s", f"     β₁ = {f4(b1)}, β₂ = {fmt(b2)}, t = {self.t}")
        self.comp(rec, "x", mx, f"{f4(b1)} · {fs(self.m[0])} + {f4(1 - b1)} · {fs(gx)}")
        self.comp(rec, "y", my, f"{f4(b1)} · {fs(self.m[1])} + {f4(1 - b1)} · {fs(gy)}")
        self.head(rec, f"v{TH} = β₂·v{THM1} + (1 − β₂)·(g{TH})²")
        self.comp(rec, "x", vx, f"{fmt(b2)} · {fs(self.v[0])} + {fs(1 - b2)} · ({fs(gx)})²")
        self.comp(rec, "y", vy, f"{fmt(b2)} · {fs(self.v[1])} + {fs(1 - b2)} · ({fs(gy)})²")
        self.head(rec, f"m̂{TH} = m{TH}/(1 − β₁ᵗ),  v̂{TH} = v{TH}/(1 − β₂ᵗ)")
        self.line(rec, "s", f"     1 − β₁ᵗ = 1 − {f4(b1)}{sup(self.t)} = {fsg(bc1)}")
        self.line(rec, "s", f"     1 − β₂ᵗ = 1 − {fmt(b2)}{sup(self.t)} = {fsg(bc2)}")
        self.comp(rec, "x", mxh, f"{fsg(mx)} / {fsg(bc1)}", fsg)
        self.comp(rec, "y", myh, f"{fsg(my)} / {fsg(bc1)}", fsg)
        self.comp(rec, "x", vxh, f"{fsg(vx)} / {fsg(bc2)}", fsg)
        self.comp(rec, "y", vyh, f"{fsg(vy)} / {fsg(bc2)}", fsg)
        self.head(rec, f"θ{TH1} = θ{TH} − η·m̂{TH}/(√v̂{TH} + ε)")
        self.line(rec, "s", f"     ε = {pow10_text(h.eps_k)}")
        self.comp(rec, "x", nx,
                  f"{f4(rec.theta[0])} − {f4(h.eta)} · {signed(mxh)}"
                  f" / (√{fs(vxh)} + {pow10_text(h.eps_k)})")
        self.comp(rec, "y", ny,
                  f"{f4(rec.theta[1])} − {f4(h.eta)} · {signed(myh)}"
                  f" / (√{fs(vyh)} + {pow10_text(h.eps_k)})")
        self.m = (mx, my)
        self.v = (vx, vy)
        self._update(rec, (nx, ny))
        return True

    def state_lines(self):
        return [("st_moment1", self.m), ("st_moment2", self.v),
                ("st_step", float(self.t))]


OPT_CLASSES = {cls.key: cls for cls in (Sgd, Momentum, AdaGrad, RmsProp, Adam)}

# ---------------------------------------------------------------
# 5. Движок прогона: состояние одного оптимизатора на выбранной задаче
# ---------------------------------------------------------------
# Итоги прогона. Статус — часть учебного текста: «сошёлся», «разошёлся» и
# «упёрся в предел шагов» это три разных наблюдения, и показывать их надо
# разными словами.
S_RUNNING = "running"
S_CONVERGED = "converged"
S_DIVERGED = "diverged"
S_LIMIT = "limit"
S_STALLED = "stalled"


class Trace:
    """Один прогон: стартовая точка, оптимизатор, история потерь и траектория."""

    def __init__(self, opt_key, objective, theta0, hp):
        self.opt_key = opt_key
        self.objective = objective
        self.theta0 = theta0
        self.hp = hp
        self.cls = OPT_CLASSES[opt_key]
        self.color = OPT_COLORS[opt_key]
        self.records = []
        self.losses = []
        self.trail = []
        self.status = S_RUNNING
        self.reset()

    def reset(self, theta0=None, hp=None, objective=None):
        """Полный сброс: параметры, состояние оптимизатора, счётчик, история."""
        if theta0 is not None:
            self.theta0 = theta0
        if hp is not None:
            self.hp = hp
        if objective is not None:
            self.objective = objective
        self.opt = self.cls(self.objective, self.hp)
        self.theta = self.theta0
        self.loss = self.objective.loss_at(self.theta)
        self.grad = self.objective.grad_at(self.theta)
        self.records = []
        self.losses = [self.loss]
        self.trail = [self.theta]
        self.status = S_RUNNING

    @property
    def iterations(self):
        return len(self.records)

    @property
    def dist(self):
        return dist_to(self.theta, self.objective.min_theta)

    @property
    def grad_norm(self):
        return norm2(self.grad)

    def converged(self):
        return self.dist < TOL_DIST and self.grad_norm < TOL_GRAD

    def finished(self):
        return self.status != S_RUNNING

    def step(self):
        """Ровно одна итерация выбранного оптимизатора.

        Возвращает StepRecord либо None, если шаг делать нечем: сошлись,
        уперлись в предел или числа вышли за разумные.
        """
        if self.status != S_RUNNING:
            return None
        if len(self.records) >= MAX_STEPS:
            self.status = S_LIMIT
            return None
        rec = StepRecord(self.opt.t + 1, self.theta, self.loss, self.grad, self.hp)
        ok = False
        try:
            ok = self.opt.advance(rec)
        except (OverflowError, ValueError, ZeroDivisionError):
            ok = False
        if not ok:
            # Шаг не применим: координаты или накопленные суммы вышли за предел.
            # Последние честные значения оставляем как есть — именно они и рисуют
            # растущие прыжки, и именно их показывает таблица сравнения.
            self.status = S_DIVERGED
            return None
        gain = self.loss - rec.new_loss
        self.records.append(rec)
        self.theta = rec.new_theta
        self.loss = rec.new_loss
        self.grad = self.objective.grad_at(self.theta)
        self.losses.append(self.loss)
        self.trail.append(self.theta)
        if self.converged():
            self.status = S_CONVERGED
        elif abs(gain) < TOL_STALL and self.iterations > 1:
            # Шаг уже не меняет потери: дальше только перебирать предел шагов.
            self.status = S_STALLED
        elif len(self.records) >= MAX_STEPS:
            self.status = S_LIMIT
        return rec

    def snapshot_state(self):
        return self.opt.state_lines()


# ---------------------------------------------------------------
# 6. Локализация: весь текст интерфейса живёт в TEXTS
#    (русский по умолчанию)
# ---------------------------------------------------------------
LANG = "ru"

# Названия алгоритмов не переводятся: их печатают так же в обеих языках.
# К ним даёт пояснение короткая подпись opt_*_gloss.
OPT_NAMES = {"sgd": "SGD", "momentum": "Momentum", "adagrad": "AdaGrad",
             "rmsprop": "RMSProp", "adam": "Adam"}

TEXTS = {
    "ru": {
        "app_title": "Оптимизаторы: сравнение методов",
        "subtitle": "учебный симулятор · один холм, один старт, пять способов обновить параметры",
        "lang_label": "Язык",

        # --- левая колонка: задача, старт, выбор методов, настройки ---
        "box_objective": "Что оптимизируем",
        "lbl_objective": "целевая функция",
        "obj_quad_name": "Анизотропный холм ½·a·x² + ½·b·y²",
        "obj_rosen_name": "Долина Розенброка (1 − x)² + c·(y − x²)²",
        "lbl_kappa": "κ = b/a — растянутость холма",
        "kappa_locked": "у долины Розенброка форма фиксированная: κ недоступен",
        "hint_min": "минимум: ({mx}, {my}), потери там {ml}",

        "box_start": "Старт (он же для всех методов)",
        "lbl_x0": "x₀",
        "lbl_y0": "y₀",

        "box_optimizers": "Оптимизаторы в эксперименте",
        "lbl_focus": "подробный расчёт показать для",
        "opt_sgd_gloss": "шаг строго против текущего градиента, памяти нет",
        "opt_momentum_gloss": "градиенты копятся в скорость: инерция гасит колебания",
        "opt_adagrad_gloss": "знаменатель — сумма всех квадратов градиентов: шаг только уменьшается",
        "opt_rmsprop_gloss": "знаменатель — среднее последних квадратов: шаг не застаивается",
        "opt_adam_gloss": "среднее градиента + среднее квадрата + поправка на смещение",

        "box_params": "Настройки (общие для всех выбранных)",
        "lbl_eta": "η — скорость обучения",
        "lbl_beta": "β — память: Momentum, RMSProp и β₁ у Adam",
        "lbl_beta2": "β₂ — память второго момента (Adam)",
        "lbl_eps": "ε — добавка к знаменателю",
        "beta2_text": "β₂ = 1 − {p} = {v}",
        "eps_text": "ε = {p}",
        "hint_shared": "η и β общие для всех методов: сравниваем разницу оптимизаторов, а не разницу настроек.",
        "hint_stability": "Для спуска без памяти холм устойчив при η < 2/b = {v}. Сейчас η {cmp} этого предела.",
        "cmp_below": "ниже",
        "cmp_above": "выше",
        "cmp_equal": "ровно на",

        # --- центр: график и запуск ---
        "graph_title": "Ландшафт потерь и траектории",
        "axis_x": "x",
        "axis_y": "y",
        "lbl_start": "старт",
        "lbl_offframe": "точек вне кадра: {n}",
        "chart_title": "Потери по номеру итерации",
        "chart_axis_iter": "итерация t",
        "chart_axis_loss": "L(θ)",
        "lbl_log_scale": "логарифмическая шкала потерь",
        "legend_title": "траектории",

        "btn_step": "Один шаг",
        "btn_auto": "Автопрогон",
        "btn_stop": "Остановить",
        "btn_fast": "До схождения",
        "btn_reset": "Сброс",
        "run_note": "Все выбранные методы получают один и тот же номер итерации: шаг двигает их одновременно.",

        # --- правая колонка: вкладки ---
        "tab_step": "Шаг и расчёт",
        "tab_compare": "Сравнение",
        "tab_exp": "Эксперименты",
        "tab_help": "Справка",

        # --- вкладка «Шаг и расчёт» ---
        "det_head": "Итерация {n} · {opt}",
        "det_head_next": "Итерация {n} · {opt} · шаг ещё не применён",
        "det_idle": "Нажмите «Один шаг»: здесь появится полная цепочка подстановок — градиент, внутреннее состояние оптимизатора и новые параметры.",
        "det_finished": "Прогон завершён ({status}). Нажмите «Сброс», чтобы начать заново.",
        "chain_grad": "Градиент",
        "chain_velocity": "Обновление скорости",
        "chain_accumulate": "Накопление квадратов градиентов",
        "chain_average": "Экспоненциальное среднее квадратов",
        "chain_moment1": "Первый момент",
        "chain_moment2": "Второй момент",
        "chain_bias": "Поправка на смещение",
        "chain_update": "Обновление параметров",
        "ln_state": "Состояние оптимизатора после шага",
        "ln_params": "θₜ → θₜ₊₁",
        "ln_status": "Состояние прогона",
        "tpl_theta_at": "θ{th} = (x, y) = ({x}, {y})",
        "tpl_loss_at": "L({x}, {y}) = {v}",
        "tpl_grad_at": "g{th} = ({x}, {y})   ‖g{th}‖ = {n}",
        "tpl_new_theta": "θ{th1} = ({x}, {y})",
        "tpl_delta": "Δθ = θ{th1} − θ{th} = ({x}, {y})",
        "tpl_new_loss": "L(θ{th1}) = {v}",
        "tpl_dist": "‖θ − θ*‖: {a} → {b}",
        "tpl_state_vec": "{name}: ({x}, {y})",
        "tpl_state_num": "{name}: {v}",
        "st_velocity": "скорость v",
        "st_accum": "сумма квадратов G",
        "st_avg": "среднее квадратов s",
        "st_moment1": "первый момент m",
        "st_moment2": "второй момент v",
        "st_step": "номер итерации t",
        "formula_hdr": "Формула метода",
        "opt_formulas": {
            "sgd": "θₜ₊₁ = θₜ − η·gₜ",
            "momentum": "vₜ = β·vₜ₋₁ + gₜ,  θₜ₊₁ = θₜ − η·vₜ",
            "adagrad": "Gₜ = Gₜ₋₁ + gₜ²,  θₜ₊₁ = θₜ − η·gₜ/(√Gₜ + ε)",
            "rmsprop": "sₜ = β·sₜ₋₁ + (1−β)·gₜ²,  θₜ₊₁ = θₜ − η·gₜ/(√sₜ + ε)",
            "adam": "mₜ = β₁mₜ₋₁ + (1−β₁)gₜ,  vₜ = β₂vₜ₋₁ + (1−β₂)gₜ²,\n"
                    "m̂ₜ = mₜ/(1−β₁ᵗ),  v̂ₜ = vₜ/(1−β₂ᵗ),\n"
                    "θₜ₊₁ = θₜ − η·m̂ₜ/(√v̂ₜ + ε)",
        },

        # --- вкладка «Сравнение» ---
        "head_opt": "Оптимизатор",
        "head_iter": "итер.",
        "head_loss": "L(θ)",
        "head_x": "x",
        "head_y": "y",
        "head_dist": "‖θ − θ*‖",
        "head_gnorm": "‖g‖",
        "head_state": "статус",
        "cmp_note": "Одна функция, один старт, одна η и β. Отличается только способ превратить градиент в шаг.",
        "cmp_none": "Не выбран ни один оптимизатор: сравнивать нечего.",
        "cmp_best": "На итерации {t} наименьшие потери у: {names}",
        "cmp_tied": "Похожий результат сразу у нескольких методов: {names}",
        "cmp_progress": "Сошлись: {done}. Продолжают: {rest}.",
        "cmp_caution": "Это наблюдение над одной задачей и одним набором настроек. Оно не означает, что метод лучше вообще.",

        # --- статусы прогонов ---
        "stt_running": "идёт",
        "stt_converged": "сошёлся",
        "stt_diverged": "разошёлся",
        "stt_limit": "предел {n} шагов",
        "stt_stalled": "шаг перестал уменьшать потери",
        # Короткие формы — только для колонки состояния в таблице: длинные
        # подписи выше в неё не помещаются и обрезались бы без смысла.
        "cell_limit": "предел",
        "cell_stalled": "нет прогресса",

        # --- нижняя строка состояния ---
        "status_ready": "Готово. Настройте холм и η, затем «Один шаг» или «До схождения».",
        "status_lang": "Язык переключён: задача, настройки, номер итерации, траектории и вкладка остались как были.",
        "status_step": "Итерация {n}: {opt} — потери {loss}, ‖g‖ = {gn}, ‖θ − θ*‖ = {dist}.",
        "status_step_done": "Все выбранные прогоны завершены ({detail}).",
        "status_nothing": "Шаг не сделан: {opt} уже завершился ({status}).",
        "status_reset": "Эксперимент перезапущен: θ₀, состояние каждого оптимизатора, счётчики, история потерь и траектории — как при запуске.",
        "status_cfg": "{what} изменена: прогон перезапущен, состояние оптимизаторов обнулено — иначе старые накопленные моменты относились бы к другим настройкам.",
        "status_obj": "Задача: {formula}. Минимум в ({mx}, {my}).",
        "status_preset": "Эксперимент «{name}» загружен: {goal}",
        "status_err": "Поле «{field}»: «{raw}» — не число. Осталось последнее правильное значение.",
        "status_range": "Поле «{field}»: {raw} вне предела [{lo}; {hi}]. Осталось последнее правильное значение.",
        "status_auto": "Автопрогон: одна итерация каждые {ms} мс. Кнопка «Остановить» прерывает его.",
        "status_fast": "Быстрый прогон: {n} итераций за один раз. Итог — {opt}: {status}.",
        "status_diverged": "{opt}: числа вышли за разумный предел — прогон остановлен, неправильный результат не рисуем.",
        "status_focus": "Подробный расчёт показывается для метода: {opt}.",
        "status_noopt": "Выберите хотя бы один оптимизатор — иначе эксперимент пустой.",

        # --- справка ---
        "help_nav_prev": "<",
        "help_nav_next": ">",
        "help_pick": "Раздел справки",
        "exp_pick": "Готовый эксперимент",
        "exp_load": "Загрузить эксперимент",
        "exp_note": "Каждый пункт задаёт не подпись, а реальный запуск: функция, старт, η, β, ε и набор методов.",
    },

    "en": {
        "app_title": "Optimizers: a comparison",
        "subtitle": "an educational simulator · one landscape, one start, five ways to update parameters",
        "lang_label": "Language",

        "box_objective": "What we optimize",
        "lbl_objective": "objective function",
        "obj_quad_name": "Anisotropic bowl ½·a·x² + ½·b·y²",
        "obj_rosen_name": "Rosenbrock valley (1 − x)² + c·(y − x²)²",
        "lbl_kappa": "κ = b/a — how stretched the bowl is",
        "kappa_locked": "the Rosenbrock valley has a fixed shape: κ is not used",
        "hint_min": "minimum: ({mx}, {my}), loss there {ml}",

        "box_start": "Start (the same for every method)",
        "lbl_x0": "x₀",
        "lbl_y0": "y₀",

        "box_optimizers": "Optimizers in the experiment",
        "lbl_focus": "detailed calculation for",
        "opt_sgd_gloss": "steps strictly against the current gradient, no memory",
        "opt_momentum_gloss": "gradients pile up into a velocity: inertia damps oscillation",
        "opt_adagrad_gloss": "the denominator is the sum of every squared gradient: steps only shrink",
        "opt_rmsprop_gloss": "the denominator averages recent squares: steps do not freeze",
        "opt_adam_gloss": "gradient average + square average + bias correction",

        "box_params": "Settings (shared by all selected)",
        "lbl_eta": "η — learning rate",
        "lbl_beta": "β — memory: Momentum, RMSProp and Adam's β₁",
        "lbl_beta2": "β₂ — second-moment memory (Adam)",
        "lbl_eps": "ε — denominator floor",
        "beta2_text": "β₂ = 1 − {p} = {v}",
        "eps_text": "ε = {p}",
        "hint_shared": "η and β are shared by every selected method: the comparison is about optimizers, not about settings.",
        "hint_stability": "Memoryless descent is stable on this bowl while η < 2/b = {v}. Right now η is {cmp} that limit.",
        "cmp_below": "below",
        "cmp_above": "above",
        "cmp_equal": "exactly at",

        "graph_title": "Loss landscape and trajectories",
        "axis_x": "x",
        "axis_y": "y",
        "lbl_start": "start",
        "lbl_offframe": "points outside the frame: {n}",
        "chart_title": "Loss against iteration number",
        "chart_axis_iter": "iteration t",
        "chart_axis_loss": "L(θ)",
        "lbl_log_scale": "logarithmic loss scale",
        "legend_title": "trajectories",

        "btn_step": "One Step",
        "btn_auto": "Auto Run",
        "btn_stop": "Stop",
        "btn_fast": "Run to Convergence",
        "btn_reset": "Reset",
        "run_note": "Every selected method gets the same iteration number: one step advances them together.",

        "tab_step": "Step & math",
        "tab_compare": "Comparison",
        "tab_exp": "Experiments",
        "tab_help": "Help",

        "det_head": "Iteration {n} · {opt}",
        "det_head_next": "Iteration {n} · {opt} · step not applied yet",
        "det_idle": "Press \"One Step\": the full substitution chain appears here — gradient, optimizer state and the new parameters.",
        "det_finished": "This run has finished ({status}). Press \"Reset\" to start over.",
        "chain_grad": "Gradient",
        "chain_velocity": "Velocity update",
        "chain_accumulate": "Squared-gradient accumulation",
        "chain_average": "Exponential average of squares",
        "chain_moment1": "First moment",
        "chain_moment2": "Second moment",
        "chain_bias": "Bias correction",
        "chain_update": "Parameter update",
        "ln_state": "Optimizer state after the step",
        "ln_params": "θₜ → θₜ₊₁",
        "ln_status": "Run status",
        "tpl_theta_at": "θ{th} = (x, y) = ({x}, {y})",
        "tpl_loss_at": "L({x}, {y}) = {v}",
        "tpl_grad_at": "g{th} = ({x}, {y})   ‖g{th}‖ = {n}",
        "tpl_new_theta": "θ{th1} = ({x}, {y})",
        "tpl_delta": "Δθ = θ{th1} − θ{th} = ({x}, {y})",
        "tpl_new_loss": "L(θ{th1}) = {v}",
        "tpl_dist": "‖θ − θ*‖: {a} → {b}",
        "tpl_state_vec": "{name}: ({x}, {y})",
        "tpl_state_num": "{name}: {v}",
        "st_velocity": "velocity v",
        "st_accum": "squared sum G",
        "st_avg": "square average s",
        "st_moment1": "first moment m",
        "st_moment2": "second moment v",
        "st_step": "iteration counter t",
        "formula_hdr": "Method formulas",
        "opt_formulas": {
            "sgd": "θₜ₊₁ = θₜ − η·gₜ",
            "momentum": "vₜ = β·vₜ₋₁ + gₜ,  θₜ₊₁ = θₜ − η·vₜ",
            "adagrad": "Gₜ = Gₜ₋₁ + gₜ²,  θₜ₊₁ = θₜ − η·gₜ/(√Gₜ + ε)",
            "rmsprop": "sₜ = β·sₜ₋₁ + (1−β)·gₜ²,  θₜ₊₁ = θₜ − η·gₜ/(√sₜ + ε)",
            "adam": "mₜ = β₁mₜ₋₁ + (1−β₁)gₜ,  vₜ = β₂vₜ₋₁ + (1−β₂)gₜ²,\n"
                    "m̂ₜ = mₜ/(1−β₁ᵗ),  v̂ₜ = vₜ/(1−β₂ᵗ),\n"
                    "θₜ₊₁ = θₜ − η·m̂ₜ/(√v̂ₜ + ε)",
        },

        "head_opt": "Optimizer",
        "head_iter": "iter",
        "head_loss": "L(θ)",
        "head_x": "x",
        "head_y": "y",
        "head_dist": "‖θ − θ*‖",
        "head_gnorm": "‖g‖",
        "head_state": "status",
        "cmp_note": "Same function, same start, same η and β. Only the way a gradient becomes a step differs.",
        "cmp_none": "No optimizer selected — there is nothing to compare.",
        "cmp_best": "At iteration {t} the lowest loss belongs to: {names}",
        "cmp_tied": "Several methods land on a similar result: {names}",
        "cmp_progress": "Converged: {done}. Still running: {rest}.",
        "cmp_caution": "That is an observation about one problem and one set of settings. It does not mean the method is better in general.",

        "stt_running": "running",
        "stt_converged": "converged",
        "stt_diverged": "diverged",
        "stt_limit": "hit the {n}-step limit",
        "stt_stalled": "steps stopped lowering the loss",
        "cell_limit": "limit",
        "cell_stalled": "no progress",

        "status_ready": "Ready. Set the bowl and η, then press \"One Step\" or \"Run to Convergence\".",
        "status_lang": "Language switched: the problem, settings, iteration number, trajectories and the current tab stayed exactly as they were.",
        "status_step": "Iteration {n}: {opt} — loss {loss}, ‖g‖ = {gn}, ‖θ − θ*‖ = {dist}.",
        "status_step_done": "Every selected run has finished ({detail}).",
        "status_nothing": "No step taken: {opt} already finished ({status}).",
        "status_reset": "Experiment restarted: θ₀, each optimizer's state, counters, loss history and trajectories are back to launch values.",
        "status_cfg": "{what} changed: the run restarted and every optimizer's state was zeroed — otherwise old accumulated moments would belong to different settings.",
        "status_obj": "Objective: {formula}. Minimum at ({mx}, {my}).",
        "status_preset": "Experiment \"{name}\" loaded: {goal}",
        "status_err": "Field \"{field}\": \"{raw}\" is not a number. The last valid value is kept.",
        "status_range": "Field \"{field}\": {raw} is outside [{lo}; {hi}]. The last valid value is kept.",
        "status_auto": "Auto run: one iteration every {ms} ms. \"Stop\" interrupts it.",
        "status_fast": "Fast run: {n} iterations at once. Outcome — {opt}: {status}.",
        "status_diverged": "{opt}: the numbers left the sane range, the run stopped and no wrong result is drawn.",
        "status_focus": "The detailed calculation is shown for: {opt}.",
        "status_noopt": "Select at least one optimizer — otherwise the experiment is empty.",

        "help_nav_prev": "<",
        "help_nav_next": ">",
        "help_pick": "Help topic",
        "exp_pick": "Ready-made experiment",
        "exp_load": "Load this experiment",
        "exp_note": "Each entry is a real run, not a caption: objective, start, η, β, ε and the set of methods.",
    },
}


def T(key, **kw):
    """Текст текущего языка по ключу; подстановки {name} — только если переданы."""
    try:
        text = TEXTS[LANG][key]
    except KeyError:
        text = TEXTS["ru"].get(key, "[" + key + "]")
    if not kw or not isinstance(text, str):
        return text
    try:
        return text.format(**kw)
    except (IndexError, KeyError, ValueError):
        return text


def set_language(code):
    """Сменить язык интерфейса (перевиджетывание делает retranslate)."""
    global LANG
    LANG = code


def status_text(status):
    """Одна строка состояния прогона на текущем языке."""
    if status == S_LIMIT:
        return T("stt_limit", n=MAX_STEPS)
    return T("stt_" + status)


# ---------------------------------------------------------------
# 7. Справка и готовые эксперименты
# ---------------------------------------------------------------
# Разделы справки. Формулы внутри строк помечены символом «», предупреждения —
# знаком !: теги текста раскрашивают эти строки иначе, и метка не переводится.
HELP_ORDER = (
    "h_chain", "h_optimizer", "h_gd_review", "h_sgd", "h_momentum",
    "h_adagrad", "h_rmsprop", "h_adam", "h_eta", "h_moving", "h_adaptive",
    "h_bias", "h_state", "h_conv", "h_osc", "h_problem", "h_howto",
)

HELP = {
    "ru": {
        "h_chain": (
            "От потерь к обновлению параметров",
            (
                "Цепочка, вокруг которой построен весь симулятор:",
                "» Потери → градиент → оптимизатор → обновление параметров → новое состояние",
                "Потери говорят, насколько плохо. Градиент — куда локально расти хуже всего.",
                "Дальше работа оптимизатора: он решает, как именно превратить этот вектор в шаг.",
                "Все пять методов в этом окне получают ровно один и тот же градиент. Разница между",
                "ними — только в том, что они с ним делают и что помнят о прошлых шагах.",
                "! Здесь нет обучения нейросети: вместо модели — явная функция двух переменных, а",
                "её градиент посчитан формулой, а не подобран численно.",
            ),
        ),
        "h_optimizer": (
            "Что такое оптимизатор",
            (
                "Оптимизатор — правило, которое по текущим параметрам и их градиенту предлагает",
                "новые параметры. Правило живёт поверх градиента: градиент даёт направление, а",
                "оптимизатор — длину, знак накопления и память.",
                "Зачем их столько, если градиент один? Потому что градиент — это локальная",
                "информация только об этой точке. На реальном холме важно ещё и то, что было в",
                "предыдущих точках: там градиент уже указывал куда-то, и это можно использовать.",
                "Каждый из пяти методов — это ответ на один вопрос:",
                "  SGD — идти туда, куда градиент показывает сейчас?",
                "  Momentum — сложить последние направления в одну скорость?",
                "  AdaGrad — уменьшить шаг там, где градиент всё время большой?",
                "  RMSProp — то же самое, но помнить только последние шаги?",
                "  Adam — совместить направление и размер шага, да ещё и поправить начало?",
            ),
        ),
        "h_gd_review": (
            "Напоминание: градиентный спуск",
            (
                "Коротко, потому что подробно это разобрано в проекте 04 «Градиентный спуск».",
                "» θₜ₊₁ = θₜ − η·gₜ",
                "Градиент указывает вверх по функции, поэтому его вычитают. η превращает",
                "направление в конкретное число. Этого хватает, чтобы спускаться, сходиться,",
                "перемахивать минимум и расходиться.",
                "Вопрос этого проекта другой: у всех пяти методов есть этот градиент — почему же",
                "пути получаются разными? Ниже разобрано, что именно каждый метод добавляет к",
                "одной строке «вычесть η·g».",
            ),
        ),
        "h_sgd": (
            "SGD: спуск по текущему градиенту",
            (
                "» θₜ₊₁ = θₜ − η·gₜ",
                "Ни памяти, ни накопления: величина шага прямо пропорциональна текущему градиенту.",
                "Отсюда две стороны.",
                "Хорошая: поведение легко предсказать, один множитель η — и всё правило известно.",
                "Плохая: на растянутом холме крутое направление даёт большой градиент и большой",
                "шаг, пологое — маленький градиент и крошечный шаг. η ограничена сверху самым",
                "крутым направлением (для холма это η < 2/b): чем больше κ, тем мельче шаг, который",
                "разрешён, и тем заметнее перемахивание, если его превысить.",
                "Зато когда η подобрана, SGD доходит до минимума быстрее методов с нормировкой: у",
                "них шаг около минимума ограничен величиной η и уменьшаться не умеет. В готовом",
                "эксперименте «Высокая кривизна» SGD делает 64 итерации, а Adam — 133. Это не",
                "ошибка симулятора, а прямое следствие нормировки градиента.",
                "! Названия «SGD» здесь соответствует полному (batch) градиенту: шума нет, задача",
                "детерминированная. Настоящий stochastic gradient descent — это тот же правило,",
                "но с зашумлённой оценкой градиента; шум в этом симуляторе не моделируется.",
            ),
        ),
        "h_momentum": (
            "Momentum: накопленная скорость",
            (
                "» vₜ = β·vₜ₋₁ + gₜ",
                "» θₜ₊₁ = θₜ − η·vₜ",
                "Соглашение записано явно: η стоит только в обновлении параметров. В некоторых",
                "учебниках speed накапливают уже умноженной на η — формулы тогда выглядят иначе,",
                "но смысл тот же. Здесь η всегда снаружи.",
                "Скорость — это скользящее среднее всех градиентов с весами 1, β, β², β³… Если",
                "направления совпадают, они складываются и шаг вырастает примерно в 1/(1−β) раз.",
                "Если направления противоположны (точность поперёк долины), они взаимно гасятся —",
                "колебания уменьшаются.",
                "! Раз шаг стал крупнее, прежняя η уже не «безопасна»: при β = 0.9 устойчивость",
                "требует η·кривизна < 2(1 + β). Момент не отменяет выбор скорости обучения, а",
                "делает его ещё важнее.",
            ),
        ),
        "h_adagrad": (
            "AdaGrad: накопленные квадраты градиентов",
            (
                "» Gₜ = Gₜ₋₁ + gₜ²",
                "» θₜ₊₁ = θₜ − η·gₜ/(√Gₜ + ε)",
                "Квадраты градиентов считаются отдельно для каждой координаты: у x и y своя",
                "накопленная сумма. Деление на √G убирает масштаб градиента: координата, по",
                "которой градиент всё время большой, получает маленькую долю и шаг по ней",
                "уменьшается; редкое направление остаётся с крупным шагом.",
                "Сумма только растёт — старых градиентов нельзя забыть. Значит эффективный шаг",
                "неуклонно уменьшается, и на длинном прогоне метод может остановиться в стороне от",
                "минимума. Это видно в эксперименте «Адаптивные шаги»: траектория AdaGrad",
                "замедляется раньше остальных.",
                "ε — добавка к знаменателю, чтобы не делить на ноль, когда градиент обнулился.",
            ),
        ),
        "h_rmsprop": (
            "RMSProp: среднее последних квадратов",
            (
                "» sₜ = β·sₜ₋₁ + (1 − β)·gₜ²",
                "» θₜ₊₁ = θₜ − η·gₜ/(√sₜ + ε)",
                "Идея та же, что у AdaGrad: поделить шаг на характерный размер градиента. Отличие",
                "одно, но ключевое — вместо бесконечной суммы используется экспоненциальное",
                "среднее. Оно «помнит» примерно последние 1/(1−β) шагов и забывает остальное.",
                "Из-за этого шаг не схлопывается навсегда: направление крутизны может уйти, и",
                "метод это заметит. При β → 1 помнит почти всё и ведет себя ближе к AdaGrad, при",
                "β → 0 остается только текущий градиент.",
                "На почти квадратичном холме RMSProp и Adam идут очень похоже; различие видно на",
                "меняющемся ландшафте — например, в изогнутой долине Розенброка.",
            ),
        ),
        "h_adam": (
            "Adam: два момента и поправка на смещение",
            (
                "» mₜ = β₁·mₜ₋₁ + (1 − β₁)·gₜ",
                "» vₜ = β₂·vₜ₋₁ + (1 − β₂)·gₜ²",
                "» m̂ₜ = mₜ/(1 − β₁ᵗ),  v̂ₜ = vₜ/(1 − β₂ᵗ)",
                "» θₜ₊₁ = θₜ − η·m̂ₜ/(√v̂ₜ + ε)",
                "m — первый момент: среднее направлений (как скорость у Momentum). v — второй",
                "момент: среднее квадратов (как s у RMSProp). Adam просто объединяет обе идеи:",
                "сглаживает направление и нормирует размер шага.",
                "Поправка на смещение обязательна: m₀ = v₀ = 0, и на первых шагах среднее",
                "занижено, потому что в нём почти ничего нет. Деление на (1 − βᵗ) возвращает",
                "среднему правильный масштаб. Без него Adam на коротком запуске выглядел бы",
                "вялым — именно поэтому в этом симуляторе m̂ и v̂ показаны отдельной строкой.",
                "Размер шага у Adam почти всегда близок к η: дробь m̂/√v̂ ограничена примерно",
                "единицей по модулю. Поэтому метод не боится разного масштаба координат — но η",
                "всё равно надо выбирать.",
            ),
        ),
        "h_eta": (
            "Скорость обучения остаётся главной настройкой",
            (
                "Адаптивные методы делят градиент на его же характерный размер — кажется, что η",
                "больше не важна. Это не так.",
                "У AdaGrad, RMSProp и Adam обновление почти не зависит от масштаба градиента, зато",
                "прямо зависит от η: шаг по каждой координате ≈ η по модулю. η перестаёт быть",
                "«размером наклона» и становится «насколько далеко я вообще готов отойти за шаг».",
                "Поэтому в этом симуляторе η одна на все пять методов: иначе сравнивать оптимизаторы",
                "некорректно. Подвигайте η при выбранных всех методах и посмотрите, как меняется",
                "график потерь — сдвигаются все кривые сразу.",
                "! Ни Adam, ни RMSProp не снимают вопрос выбора скорости обучения. Они лишь делают",
                "его менее критичным к масштабу градиентов.",
            ),
        ),
        "h_moving": (
            "Момент и скользящие средние",
            (
                "Скользящее среднее — один и тот же приём в трёх местах этого симулятора:",
                "» vₜ = β·vₜ₋₁ + gₜ                 (Momentum: среднее направлений)",
                "» sₜ = β·sₜ₋₁ + (1 − β)·gₜ²      (RMSProp: среднее квадратов)",
                "» mₜ = β₁·mₜ₋₁ + (1 − β₁)·gₜ   (Adam: среднее направлений)",
                "β — это «как долго помнить». Память держится примерно 1/(1−β) шагов: β = 0.9 —",
                "порядка десяти шагов, β = 0.99 — сотни.",
                "Momentum копирует сумму без множителя (1 − β), поэтому v растет до η/(1−β);",
                "RMSProp и Adam домножают на (1 − β), и среднее остаётся того же масштаба, что и",
                "сам градиент. Из-за этого β у Momentum влияет на размер шага сильнее.",
                "Зачем сглаживать? Один градиент — это локальность и (в реальном обучении) шум.",
                "Среднее по последним шагам усредняет направление и оставляет только устойчивую",
                "составляющую.",
            ),
        ),
        "h_adaptive": (
            "Адаптивные размеры шага",
            (
                "«Адаптивный» означает: у каждой координаты своя эффективная скорость обучения,",
                "и её считает сам метод из наблюдений над градиентом.",
                "» η_эфф = η / (√Gₜ + ε)          (AdaGrad)",
                "» η_эфф = η / (√sₜ + ε)          (RMSProp, Adam — там вместо s стоит v̂)",
                "Крутое направление (большой градиент) получает маленький шаг, пологое — большой.",
                "На растянутом холме это то, чего SGD сам себе дать не может: там один общий η",
                "вынужден угождать обоим направлениям сразу.",
                "Цена: накопленная статистика устаревает и занимает место в памяти; при",
                "экстремально малом градиенте ε начинает доминировать, и шаг ведёт себя иначе, чем",
                "описывается формулой. Поэтому ε тоже показан на панели настроек.",
            ),
        ),
        "h_bias": (
            "Поправка на смещение",
            (
                "m₀ = 0 и v₀ = 0, а рекурсия mₜ = β₁mₜ₋₁ + (1 − β₁)gₜ усредняет по всем шагам,",
                "включая эти нули. На первых итерациях среднее заметно ближе к нулю, чем должно быть.",
                "» E[mₜ] = (1 − β₁ᵗ)·(среднее g)",
                "Отсюда деление: m̂ₜ = mₜ/(1 − β₁ᵗ), v̂ₜ = vₜ/(1 − β₂ᵗ). Оно ровно компенсирует",
                "нулевое начало, и m̂, v̂ становятся честными средними.",
                "Величина поправки падает быстро: при β₁ = 0.9 на десятом шаге (1 − β₁¹⁰) ≈ 0.65,",
                "на сороковом — почти 1. Значит поправка важна в начале прогона — как раз там,",
                "где строятся первые кривые на графике потерь.",
                "В симуляторе m̂ и v̂ показаны отдельной строкой с настоящими числами: их можно",
                "пересчитать на калькуляторе.",
            ),
        ),
        "h_state": (
            "Состояние оптимизатора",
            (
                "Состояние — это всё, что метод помнит между шагами:",
                "  SGD: ничего",
                "  Momentum: скорость v",
                "  AdaGrad: накопленная сумма квадратов G",
                "  RMSProp: среднее квадратов s",
                "  Adam: m, v и счётчик итераций t",
                "Из-за состояния результат зависит не только от точки, но и от истории пути к ней.",
                "Одна и та же формула, запущенная с накопленными моментами, даёт другой шаг.",
                "! Любая смена настроек — η, β, ε, функции, старта или набора методов — обнуляет",
                "состояние и счётчики. Иначе метод дошагивал бы по старым накоплениям, и",
                "эксперимент стал бы невоспроизводимым.",
                "t у Adam обязателен: без него поправка (1 − βᵗ) не вычислима.",
            ),
        ),
        "h_conv": (
            "Сходимость",
            (
                "Прогон считается сошедшимся, когда выполняются оба условия:",
                "» ‖θ − θ*‖ < 0.005   и   ‖∇L‖ < 0.005",
                "Первое — мы близко к минимуму. Второе — склон у нас под ногами почти плоский.",
                "Только близости мало: можно перескакивать минимум туда-сюда, оставаясь рядом, и",
                "при этом не сходиться. Только маленького градиента тоже мало: в широком плоском",
                "месте он мал и далеко от минимума.",
                "Равенства нулю программа не обещает: формулировка всегда «в пределах допуска».",
                "Кроме сходимости, прогон может закончиться по другой причине: предел числа шагов",
                "(метод идёт слишком медленно) или выход чисел за разумный предел (расхождение).",
                "Все три случая помечены в таблице сравнения по-разному — это разные наблюдения.",
            ),
        ),
        "h_osc": (
            "Колебания",
            (
                "Для направления с кривизной c обновление SGD даёт множитель (1 − η·c). Он",
                "отрицательный при η·c > 1: точка каждый шаг перелетает через минимум,",
                "и при |1 − η·c| < 1 перелёты затухают, а при |1 − η·c| > 1 — растут.",
                "На растянутом холме крутое направление почти всегда в этой зоне перелётов, пока",
                "пологое ещё не дошло: отсюда «ёлочка» траектории.",
                "Momentum ту же картину сглаживает: противоположные по знаку градиенты в",
                "скорости вычитаются, и поперёк долины метод идёт меньше, чем вдоль.",
                "Адаптивные методы делят на √(квадратов), поэтому амплитуда шага ограничена η и",
                "перемахивание становится равномерным, а не растущим.",
                "Посмотреть на колебания проще всего в эксперименте «Чувствительность к η»: η",
                "заведомо больше предела 2/b, и SGD уходит в растущие прыжки.",
            ),
        ),
        "h_problem": (
            "Почему «лучший» оптимизатор зависит от задачи",
            (
                "Здесь нет победителя, и это не недоработка интерфейса.",
                "На слабовыраженной растянутости (эксперимент «Базовая сходимость», κ = 2) памяти",
                "не нужно вовсе: SGD доходит за 48 итераций, а Adam за 92 — лишние состояния только",
                "добавляют настройки.",
                "На сильной растянутости (κ = 16) η ограничена крутым направлением; RMSProp на тех",
                "же настройках проходит 35 итераций, тогда как Adam и Momentum — больше ста.",
                "На слишком большой η чистый спуск расходится растущими прыжками, а Momentum при",
                "том же η ещё устойчив и доходит до минимума.",
                "На длинном прогоне AdaGrad замедляет сам себя, а RMSProp и Adam продолжают.",
                "В изогнутой долине инерция вдоль дна важнее длины шага: там Momentum проходит 114",
                "итераций против 178 у SGD.",
                "! Поэтому в справке нет строки «Adam — лучший метод». Правило такое: чем проще",
                "задача и чем меньше хочется настраивать, тем привлекательнее SGD; чем сложнее",
                "ландшафт, тем полезнее моменты и нормировка. Проверяется это кнопкой «До",
                "схождения» на конкретной задаче, а не мнением из интернета.",
            ),
        ),
        "h_howto": (
            "Как работать с симулятором",
            (
                "1. Выберите функцию и крутизну холма; координаты старта задайте сами — старт",
                "   у всех выбранных методов один и тот же.",
                "2. Отметьте галочками методы, которые участвуют в сравнении, и η с β.",
                "3. «Один шаг» — ровно одна итерация, вкладка «Шаг и расчёт» показывает все",
                "   подстановки для выбранного метода. «Автопрогон» — по шагу с паузой, так видно",
                "   форму траектории. «До схождения» — быстро прокручивает все прогоны до конца.",
                "4. «Сравнение» — числа по методам и график L(θ) против номера итерации. Кривые",
                "   построены из настоящих историй шагов, идеализированных линий здесь нет.",
                "5. «Эксперименты» — пять готовых запусков, каждый из которых о чём-то говорит:",
                "   базовая сходимость, высокая кривизна, чувствительность к η, адаптивные шаги,",
                "   долина Розенброка.",
                "Язык переключают кнопки RU и EN в правом верхнем углу: прогон, номер",
                "итерации, открытая вкладка и траектории при этом сохраняются.",
            ),
        ),
    },

    "en": {
        "h_chain": (
            "From loss to a parameter update",
            (
                "The chain this whole simulator is built around:",
                "» Loss → gradient → optimizer → parameter update → new model state",
                "The loss says how bad things are. The gradient says which way gets worse fastest.",
                "From there the optimizer takes over: it decides how that vector becomes a step.",
                "All five methods in this window are handed exactly the same gradient. They differ",
                "only in what they do with it and in what they remember about earlier steps.",
                "! No neural network is trained here: an explicit function of two variables stands",
                "in its place, and its gradient comes from a formula rather than a numerical probe.",
            ),
        ),
        "h_optimizer": (
            "What an optimizer is",
            (
                "An optimizer is a rule that takes the current parameters and their gradient and",
                "proposes new parameters. The rule sits on top of the gradient: the gradient gives",
                "a direction, the optimizer supplies the length, the accumulation and the memory.",
                "Why so many of them for one gradient? Because a gradient is local information",
                "about this point only. On a real landscape it also matters what previous points",
                "said — and that can be put to use.",
                "Each of the five methods answers one question:",
                "  SGD — go where the gradient points right now?",
                "  Momentum — fold recent directions into a velocity?",
                "  AdaGrad — shrink the step where the gradient is persistently large?",
                "  RMSProp — same, but remember only the most recent steps?",
                "  Adam — combine direction and step size, and fix the beginning on top?",
            ),
        ),
        "h_gd_review": (
            "Quick reminder: gradient descent",
            (
                "Briefly, because Project 04 (Gradient Descent) works this out in full.",
                "» θₜ₊₁ = θₜ − η·gₜ",
                "The gradient points uphill, so we subtract it. η turns a direction into a number.",
                "That is enough to descend, to converge, to overshoot and to diverge.",
                "This project asks a different question: all five methods get the same gradient —",
                "so why do the paths come out different? The sections below show exactly what each",
                "method adds to the single line \"subtract η·g\".",
            ),
        ),
        "h_sgd": (
            "SGD: descent on the current gradient",
            (
                "» θₜ₊₁ = θₜ − η·gₜ",
                "No memory, no accumulation: the step is directly proportional to the current gradient.",
                "That has two sides to it.",
                "The good one: the behaviour is easy to predict — one multiplier η and the rule is",
                "fully known.",
                "The bad one: on a stretched bowl the steep direction produces a large gradient and",
                "a large step, while the flat direction produces a small gradient and a tiny step.",
                "η is capped from above by the steepest direction (for this bowl η < 2/b): the larger",
                "κ is, the smaller the allowed step, and the more visible the overshoot once that",
                "limit is crossed.",
                "The upside: when η is well chosen, SGD reaches the minimum faster than the",
                "normalized methods, whose step stays bounded by η near the minimum and cannot",
                "shrink further. In the ready-made \"High curvature\" experiment SGD takes 64",
                "iterations while Adam takes 133. That is not a bug in the simulator — it follows",
                "directly from normalizing the gradient.",
                "! \"SGD\" here means the full (batch) gradient: the problem is deterministic and",
                "carries no noise. True stochastic gradient descent is the same rule applied to a",
                "noisy gradient estimate; noise is not simulated in this app.",
            ),
        ),
        "h_momentum": (
            "Momentum: accumulated velocity",
            (
                "» vₜ = β·vₜ₋₁ + gₜ",
                "» θₜ₊₁ = θₜ − η·vₜ",
                "The convention is stated on purpose: η appears only in the parameter update. Some",
                "textbooks accumulate an already-scaled velocity, which makes the formula look",
                "different while meaning the same thing. Here η always stays outside.",
                "The velocity is a moving average of gradients weighted 1, β, β², β³… When",
                "directions agree they add up and the step grows by roughly 1/(1−β). When they",
                "oppose each other (bouncing across a valley) they cancel — oscillation shrinks.",
                "! Since the step became larger, the old η is no longer \"safe\": stability now asks",
                "for η·curvature < 2(1 + β). Momentum does not remove the learning-rate question,",
                "it makes it more important.",
            ),
        ),
        "h_adagrad": (
            "AdaGrad: accumulated squared gradients",
            (
                "» Gₜ = Gₜ₋₁ + gₜ²",
                "» θₜ₊₁ = θₜ − η·gₜ/(√Gₜ + ε)",
                "The squares are accumulated per coordinate: x and y keep separate sums. Dividing by",
                "√G removes the scale of the gradient — a coordinate whose gradient is always large",
                "gets a small share and its step shrinks, while a rare direction keeps a big step.",
                "The sum only grows; old gradients cannot be forgotten. So the effective step",
                "decreases monotonically and a long run can stall somewhere short of the minimum.",
                "That shows up in the \"Adaptive steps\" experiment, where the AdaGrad trajectory",
                "slows down before the others do.",
                "ε is added to the denominator so that a vanished gradient cannot cause a division",
                "by zero.",
            ),
        ),
        "h_rmsprop": (
            "RMSProp: an average of recent squares",
            (
                "» sₜ = β·sₜ₋₁ + (1 − β)·gₜ²",
                "» θₜ₊₁ = θₜ − η·gₜ/(√sₜ + ε)",
                "The same idea as AdaGrad — divide the step by the characteristic size of the",
                "gradient. One difference, and it is the important one: an exponential average",
                "instead of an endless sum. It remembers roughly the last 1/(1−β) steps and forgets",
                "the rest.",
                "That is why the step does not collapse for good: the steep direction can change and",
                "the method notices. As β → 1 it remembers nearly everything and behaves like",
                "AdaGrad; as β → 0 only the current gradient is left.",
                "On a nearly quadratic bowl RMSProp and Adam walk almost identically; the difference",
                "appears on a changing landscape, for instance in the curved Rosenbrock valley.",
            ),
        ),
        "h_adam": (
            "Adam: two moments and a bias correction",
            (
                "» mₜ = β₁·mₜ₋₁ + (1 − β₁)·gₜ",
                "» vₜ = β₂·vₜ₋₁ + (1 − β₂)·gₜ²",
                "» m̂ₜ = mₜ/(1 − β₁ᵗ),  v̂ₜ = vₜ/(1 − β₂ᵗ)",
                "» θₜ₊₁ = θₜ − η·m̂ₜ/(√v̂ₜ + ε)",
                "m, the first moment, averages directions (Momentum's velocity). v, the second",
                "moment, averages squares (RMSProp's s). Adam simply joins both ideas: it smooths",
                "the direction and normalizes the step size.",
                "Bias correction is not decoration: m₀ = v₀ = 0, so early averages are pulled toward",
                "zero by those initial values. Dividing by (1 − βᵗ) restores the proper scale. With-",
                "out it Adam would look sluggish in a short run — which is exactly why m̂ and v̂ get",
                "their own line in this simulator.",
                "Adam's step stays close to η in size: the fraction m̂/√v̂ is bounded by roughly one.",
                "That is why the method tolerates different coordinate scales — yet η still has to",
                "be chosen.",
            ),
        ),
        "h_eta": (
            "The learning rate stays a fundamental hyperparameter",
            (
                "Adaptive methods divide the gradient by its own characteristic size, so it looks as",
                "if η stopped mattering. It did not.",
                "For AdaGrad, RMSProp and Adam the update barely depends on the gradient's scale, but",
                "it depends directly on η: each coordinate moves by about η per step. η stops being",
                "\"how steep is it\" and becomes \"how far am I willing to go in one step at all\".",
                "That is why this simulator keeps one shared η for all five methods — otherwise the",
                "comparison would be about settings, not about optimizers. Move η with all methods",
                "selected and watch the loss chart: every curve shifts at once.",
                "! Neither Adam nor RMSProp removes the need to choose a learning rate. They only",
                "make it less sensitive to gradient scale.",
            ),
        ),
        "h_moving": (
            "Momentum and moving averages",
            (
                "The moving average is the same trick in three places of this simulator:",
                "» vₜ = β·vₜ₋₁ + gₜ                 (Momentum: average of directions)",
                "» sₜ = β·sₜ₋₁ + (1 − β)·gₜ²      (RMSProp: average of squares)",
                "» mₜ = β₁·mₜ₋₁ + (1 − β₁)·gₜ   (Adam: average of directions)",
                "β means \"how long to remember\". Memory lasts about 1/(1−β) steps: β = 0.9 is some",
                "ten steps, β = 0.99 is hundreds.",
                "Momentum sums without the (1 − β) factor, so v grows toward g/(1−β); RMSProp and",
                "Adam multiply by (1 − β), keeping the average on the gradient's own scale. That is",
                "why β changes Momentum's step size far more strongly.",
                "Why smooth at all? A single gradient is local and — in real training — noisy. An",
                "average over recent steps keeps only the persistent component of the direction.",
            ),
        ),
        "h_adaptive": (
            "Adaptive step sizes",
            (
                "\"Adaptive\" means each coordinate has its own effective learning rate, computed by",
                "the method itself out of the observed gradients:",
                "» η_eff = η / (√Gₜ + ε)          (AdaGrad)",
                "» η_eff = η / (√sₜ + ε)          (RMSProp; for Adam the denominator uses v̂)",
                "A steep direction (large gradient) gets a small step, a gentle one gets a large",
                "step. On a stretched bowl that is precisely what SGD cannot do for itself: one",
                "shared η has to satisfy both directions at once.",
                "The price: the accumulated statistic goes stale and takes memory, and when the",
                "gradient becomes extremely small ε starts to dominate, so the step no longer",
                "behaves like the formula suggests. That is why ε is on the settings panel too.",
            ),
        ),
        "h_bias": (
            "Bias correction",
            (
                "m₀ = 0 and v₀ = 0, while the recursion mₜ = β₁mₜ₋₁ + (1 − β₁)gₜ averages over",
                "every step including those zeros. Early on the average is noticeably closer to zero",
                "than it should be.",
                "» E[mₜ] = (1 − β₁ᵗ)·(average of g)",
                "Hence the division: m̂ₜ = mₜ/(1 − β₁ᵗ), v̂ₜ = vₜ/(1 − β₂ᵗ). It exactly compensates",
                "the zero start, so m̂ and v̂ become honest averages.",
                "The correction decays fast: with β₁ = 0.9, (1 − β₁¹⁰) ≈ 0.65 on step ten and nearly",
                "one by step forty. So it matters at the beginning of a run — which is where the",
                "first part of the loss curve is drawn.",
                "The simulator shows m̂ and v̂ on their own line with the actual numbers, so you can",
                "reproduce them with a calculator.",
            ),
        ),
        "h_state": (
            "Optimizer state",
            (
                "State is everything the method keeps between steps:",
                "  SGD: nothing",
                "  Momentum: velocity v",
                "  AdaGrad: accumulated squared sum G",
                "  RMSProp: square average s",
                "  Adam: m, v and the iteration counter t",
                "Because of state, the result depends not only on the current point but on the",
                "history of how the run got there. The same formula with different accumulated",
                "moments produces a different step.",
                "! Any change of settings — η, β, ε, the objective, the start or the set of methods —",
                "zeroes the state and the counters. Otherwise a method would keep stepping on",
                "stale accumulations and the experiment would stop being reproducible.",
                "Adam needs t: without it the correction (1 − βᵗ) cannot be computed.",
            ),
        ),
        "h_conv": (
            "Convergence",
            (
                "A run counts as converged when both conditions hold:",
                "» ‖θ − θ*‖ < 0.005   and   ‖∇L‖ < 0.005",
                "The first says we are close to the minimum; the second that the ground under us is",
                "nearly flat. Closeness alone is not enough — a run can bounce across the minimum",
                "forever while staying nearby. A small gradient alone is not enough either: a wide",
                "flat region has a tiny gradient far away from the minimum.",
                "The app never promises exact equality; the wording is always \"within tolerance\".",
                "A run can also end for a different reason: the iteration limit (the method crawls)",
                "or numbers leaving the sane range (divergence). All three cases are labelled",
                "separately in the comparison table, because they are separate observations.",
            ),
        ),
        "h_osc": (
            "Oscillation",
            (
                "For a direction with curvature c, an SGD step multiplies the distance by (1 − η·c).",
                "The factor is negative once η·c > 1: the point flies over the minimum every step,",
                "and with |1 − η·c| < 1 those overshoots decay, while |1 − η·c| > 1 makes them grow.",
                "On a stretched bowl the steep direction sits in this overshooting zone while the",
                "flat one has not arrived yet — hence the zig-zag trajectory.",
                "Momentum smooths the same picture: gradients of opposite sign cancel inside the",
                "velocity, so the method crosses the valley less than it travels along it.",
                "Adaptive methods divide by √(squares), so the step amplitude is capped near η and",
                "overshooting becomes steady rather than growing.",
                "Easiest place to see oscillation: the \"Learning-rate sensitivity\" experiment, where",
                "η is deliberately above the 2/b limit and SGD breaks into growing jumps.",
            ),
        ),
        "h_problem": (
            "Why the \"best\" optimizer depends on the problem",
            (
                "There is no winner here, and that is not a missing feature of the interface.",
                "With mild stretching (the \"Basic convergence\" experiment, κ = 2) no memory is",
                "needed at all: SGD arrives in 48 iterations while Adam takes 92 — extra state only",
                "adds knobs to tune.",
                "With strong stretching (κ = 16) η is capped by the steep direction; on the same",
                "settings RMSProp gets there in 35 iterations while Adam and Momentum need over a",
                "hundred.",
                "With too large an η plain descent diverges into growing jumps, yet Momentum at that",
                "very η stays stable and reaches the minimum.",
                "On a long run AdaGrad slows itself down while RMSProp and Adam keep going.",
                "In a curved valley inertia along the floor matters more than step length: there",
                "Momentum needs 114 iterations against SGD's 178.",
                "! So there is no line saying \"Adam is the best method\". The rule is: the simpler",
                "the problem and the fewer knobs you want, the more attractive SGD looks; the",
                "rougher the landscape, the more useful moments and normalization become. Verify it",
                "with the \"Run to Convergence\" button on a concrete problem.",
            ),
        ),
        "h_howto": (
            "How to use this simulator",
            (
                "1. Pick the objective and how stretched the bowl is; set the start yourself — every",
                "   selected method begins from that same point.",
                "2. Tick the methods that take part in the comparison, then set η, β, β₂ and ε.",
                "3. \"One Step\" performs exactly one iteration and the \"Step & math\" tab shows every",
                "   substitution for the focused method. \"Auto Run\" advances one step at a time so",
                "   you can watch the trajectory take shape. \"Run to Convergence\" fast-forwards all",
                "   runs to their end.",
                "4. \"Comparison\" holds the numbers per method and the L(θ)-against-iteration chart.",
                "   The curves come from the real histories — no idealized line is hard-coded.",
                "5. \"Experiments\" offers five ready runs, each making one point: basic convergence,",
                "   high curvature, learning-rate sensitivity, adaptive steps, Rosenbrock valley.",
                "The language can be switched at any moment: the run, the iteration number, the open",
                "tab and all trajectories stay exactly as they were.",
            ),
        ),
    },
}

# Ссылка на «ключ -> корень статуса» больше не нужна; оставим пустой словарь,
# если ключ окажется в TEXTS ошибочным — проверка локализации его находит.
EXP_ORDER = ("ex_basic", "ex_curvature", "ex_eta", "ex_adaptive", "ex_rosen")

# Каждый пункт настраивает настоящий запуск: функцию, стартовую точку, η, β, ε
# и набор методов. Константы подобраны прогоном (не «на глаз»): η для эффекта
# перемахивания обязана попадать в диапазон 1 < η·b < 2, иначе oscillation нет.
PRESETS = (
    {
        # Спокойный случай: все пять методов доходят до одного минимума,
        # различается только число итераций (порядка 20…260).
        "key": "ex_basic",
        "obj": "quad", "kappa": 2.0, "theta0": (3.0, 1.8),
        "eta": 0.25, "beta": 0.90, "beta2_k": 3, "eps_k": 8,
        "selected": OPT_ORDER,
    },
    {
        # Растянутый холм: η·b = 1.6 — шаг поперёк долины меняет знак, но затухает.
        "key": "ex_curvature",
        "obj": "quad", "kappa": 16.0, "theta0": (4.0, 1.6),
        "eta": 0.20, "beta": 0.90, "beta2_k": 3, "eps_k": 8,
        "selected": OPT_ORDER,
    },
    {
        # η = 0.30 > предела устойчивости 2/b = 0.25: SGD расходится.
        "key": "ex_eta",
        "obj": "quad", "kappa": 16.0, "theta0": (3.0, 2.0),
        "eta": 0.30, "beta": 0.90, "beta2_k": 3, "eps_k": 8,
        "selected": OPT_ORDER,
    },
    {
        # Длинный прогон при малой η: накопленные квадраты съедают шаг AdaGrad.
        "key": "ex_adaptive",
        "obj": "quad", "kappa": 4.0, "theta0": (4.0, 2.5),
        "eta": 0.05, "beta": 0.90, "beta2_k": 3, "eps_k": 8,
        "selected": OPT_ORDER,
    },
    {
        # Изогнутая долина: у RMSProp шаг прыгает поперёк долины до предела шагов.
        "key": "ex_rosen",
        "obj": "rosen", "c": 2.0, "theta0": (-0.8, 2.2),
        "eta": 0.08, "beta": 0.90, "beta2_k": 3, "eps_k": 8,
        "selected": OPT_ORDER,
    },
)

EXP_TEXTS = {
    "ru": {
        "ex_basic": (
            "Базовая сходимость",
            "мягкий холм κ = 2 и η = 0.25: все пять методов доходят до одного минимума, "
            "различается только число итераций — откройте вкладку «Сравнение»",
        ),
        "ex_curvature": (
            "Высокая кривизна",
            "холм растянут в 16 раз (κ = 16), η·b = 1.6: SGD скачет поперёк долины — это видно "
            "по траектории, где знак y меняется каждый шаг. Momentum те прыжки гасит, а AdaGrad "
            "на этих настройках доходит только до предела в 600 шагов — кто именно и за сколько "
            "итераций дошёл быстрее, смотрите в таблице сравнения",
        ),
        "ex_eta": (
            "Чувствительность к η",
            "η = 0.30 выше предела устойчивости чистого спуска (2/b = 0.25): SGD расходится "
            "растущими прыжками, Momentum при тех же η ещё держится, AdaGrad, RMSProp и Adam "
            "продолжают уменьшать потери — один общий η на все методы",
        ),
        "ex_adaptive": (
            "Адаптивные шаги",
            "тот же тип задачи, но η = 0.05 и длинный прогон: у AdaGrad знаменатель G только "
            "растёт, и к пределу в 600 шагов он замирает в ~2 единицах от минимума, тогда как "
            "RMSProp и Adam доходят — им мешает только собственная η",
        ),
        "ex_rosen": (
            "Долина Розенброка",
            "узкая изогнутая долина: SGD, Momentum и Adam доходят до минимума, AdaGrad застревает "
            "на склоне, а RMSProp с β = 0.9 до самого конца скачет поперёк долины: направление "
            "вдоль дна он не запоминает",
        ),
    },
    "en": {
        "ex_basic": (
            "Basic convergence",
            "a gentle bowl (κ = 2) with η = 0.25: all five methods reach the same minimum and only "
            "the iteration count differs — open the \"Comparison\" tab to see it",
        ),
        "ex_curvature": (
            "High curvature",
            "the bowl is stretched 16 times (κ = 16) and η·b = 1.6: SGD hops across the valley — "
            "the trajectory shows it, its y flips sign on every step. Momentum damps those hops, "
            "while AdaGrad here only reaches the 600-step limit; the comparison table shows who "
            "actually got there sooner and in how many iterations",
        ),
        "ex_eta": (
            "Learning-rate sensitivity",
            "η = 0.30 sits above the stability limit of plain descent (2/b = 0.25): SGD diverges "
            "with growing jumps, Momentum still holds at the same η, and AdaGrad, RMSProp and Adam "
            "keep reducing the loss — with one shared η",
        ),
        "ex_adaptive": (
            "Adaptive step intuition",
            "same kind of problem, but η = 0.05 and a long run: AdaGrad's denominator G only grows, "
            "so by the 600-step limit it freezes about 2 units short of the minimum, while RMSProp "
            "and Adam get there — only their own η holds them back",
        ),
        "ex_rosen": (
            "Rosenbrock valley",
            "a narrow curved valley: SGD, Momentum and Adam reach the minimum, AdaGrad sticks on "
            "the slope, and RMSProp with β = 0.9 keeps hopping across the valley until the step "
            "limit — it does not remember the direction along the floor",
        ),
    },
}


def help_text(key):
    """Раздел справки на текущем языке: (заголовок, кортеж строк)."""
    entry = HELP.get(LANG, HELP["ru"]).get(key) or HELP["ru"].get(key)
    if entry is None:
        return ("[" + key + "]", ())
    return entry


def exp_text(key):
    """Название и цель готового эксперимента на текущем языке."""
    entry = EXP_TEXTS.get(LANG, EXP_TEXTS["ru"]).get(key) or EXP_TEXTS["ru"][key]
    return entry


# ---------------------------------------------------------------
# 8. Виджеты
# ---------------------------------------------------------------
TEXT_WIDTH = 46            # ширина панелей текста в символах
PAD_L = (62, 18, 50, 42)   # поля графика: влево, вправо, вверх, вниз
# Минимум окна диктует макет, а не пожелание: 330 px левой колонки + 430 px
# холстов + 486 px таблицы + поля. Меньше — правая колонка уезжает за край,
# и заголовок таблицы обрезается без всякого скролла.
MIN_WIDTH = 1290
MIN_HEIGHT = 700
# Минимальный отступ между подписями делений по вертикали: меньше — и строки
# «10⁻⁷» и «10⁻⁶» наезжают друг на друга.
TICK_GAP = 15
# Левая панель: 330 px всего, минус полоса прокрутки и поля фреймов. Подписи
# длиннее этой ширины обязана переносить, иначе они обрезаются без предупреждения.
LEFT_W = 330
PANEL_TEXT_W = 262
# Ширина переноса пояснений при минимальном окне (1240 px): колонка 430 px,
# минус поля вкладки ≈ 406. Больше брать нельзя: подпись упрётся в край и
# обрежется без всякого скролла.
NOTE_TEXT_W = 380

# Теги панелей текста: k — формула, t — обычный текст, b — предупреждение.
TAGS = {"k": "k", "t": "t", "b": "b", "h": "h"}


def fill_text(widget, entries):
    """Перезаписать панель текста: entries = [(tag, строка), ...]."""
    widget.configure(state="normal")
    widget.delete("1.0", "end")
    for tag, line in entries:
        widget.insert("end", line + "\n", tag)
    widget.configure(state="disabled")


def level_tint(index, total):
    """Оттенка изолиний от светлого к синему: так читается «глубина» холма."""
    lo = (233, 238, 242)
    hi = (72, 112, 138)
    f = index / max(total - 1, 1)
    r = int(lo[0] + (hi[0] - lo[0]) * f)
    g = int(lo[1] + (hi[1] - lo[1]) * f)
    b = int(lo[2] + (hi[2] - lo[2]) * f)
    return f"#{r:02x}{g:02x}{b:02x}"


def nice_step(raw):
    """Шаг деления из {1, 2, 2.5, 5}·10ⁿ — чтобы подписи осей были круглыми."""
    if raw <= 0:
        return 1.0
    mag = 10.0 ** math.floor(math.log10(raw))
    for mult in (1, 2, 2.5, 5, 10):
        if mag * mult >= raw:
            return mag * mult
    return mag * 10


def tick_values(lo, hi, want=6):
    """Список делений оси в границах [lo, hi]."""
    step = nice_step((hi - lo) / max(want, 1))
    first = math.ceil(lo / step) * step
    out = []
    v = first
    while v <= hi + step * 1e-6:
        out.append(round(v, 10))
        v += step
    if not out:
        out = [lo]
    return out


def _edge_point(a, b, level):
    """Доля, на которой уровень пересекает отрезок значений [a, b]."""
    d = b - a
    if d == 0.0:
        return 0.5
    return clamp((level - a) / d, 0.0, 1.0)


def marching_segments(vals, cols, rows, level):
    """Изолиния уровня методом marching squares в координатах сетки.

    Возвращает список ((x1, y1), (x2, y2)) — отрезки между точками пересечения
    уровня с рёбрами ячейки. Углы ячейки: TL=8, TR=4, BR=2, BL=1.
    """
    segs = []
    for j in range(rows - 1):
        row = vals[j]
        nxt = vals[j + 1]
        for i in range(cols - 1):
            a = row[i]
            b = row[i + 1]
            c = nxt[i + 1]
            d = nxt[i]
            code = ((a > level) << 3) | ((b > level) << 2) | ((c > level) << 1) | (d > level)
            if code in (0, 15):
                continue
            tp = (i + _edge_point(a, b, level), j)
            rp = (i + 1.0, j + _edge_point(b, c, level))
            bp = (i + _edge_point(d, c, level), j + 1.0)
            lp = (i, j + _edge_point(a, d, level))
            if code == 5 or code == 10:
                # Седловая ячейка: четыре пересечения, паруем по центру ячейки.
                above = 0.25 * (a + b + c + d) > level
                if code == 5:
                    pairs = [(tp, lp), (bp, rp)] if above else [(tp, rp), (lp, bp)]
                else:
                    pairs = [(tp, rp), (lp, bp)] if above else [(tp, lp), (bp, rp)]
            else:
                table = {
                    1: (lp, bp), 2: (bp, rp), 3: (lp, rp), 4: (rp, tp),
                    6: (tp, bp), 7: (tp, lp), 8: (tp, lp), 9: (tp, bp),
                    11: (tp, rp), 12: (lp, rp), 13: (rp, bp), 14: (lp, bp),
                }
                pairs = [table[code]]
            for p1, p2 in pairs:
                segs.append((p1, p2))
    return segs


class ScrollFrame(ttk.Frame):
    """Вертикально прокручиваемая колонка: на ней строятся блоки LabelFrame."""

    def __init__(self, parent, width=330, bg=COL_BG):
        super().__init__(parent)
        self.canvas = tk.Canvas(self, width=width, height=100, bg=bg,
                               highlightthickness=0, bd=0)
        self.bar = ttk.Scrollbar(self, orient="vertical", command=self.canvas.yview)
        self.inner = ttk.Frame(self.canvas, padding=(12, 4))
        self._win = self.canvas.create_window((0, 0), window=self.inner, anchor="nw")
        self.canvas.configure(yscrollcommand=self.bar.set)
        self.canvas.pack(side="left", fill="both", expand=True)
        self.bar.pack(side="right", fill="y")
        self.inner.bind("<Configure>", self._on_inner)
        self.canvas.bind("<Configure>", self._on_canvas)
        self.canvas.bind_all("<MouseWheel>", self._on_wheel)
        self.bind("<Destroy>", self._unbind_wheel)

    def _on_inner(self, _):
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))

    def _on_canvas(self, event):
        # Внутренний фрейм шире окна не выпускаем: иначе появляется горизонтальный
        # сдвиг, и подписи уходят за правый край.
        self.canvas.itemconfigure(self._win, width=max(event.width, 200))

    def _on_wheel(self, event):
        # bind_all даёт событие на всякий прокрутке: крутим только тот список,
        # над которым находится курсор, иначе колесо двигало бы все панели сразу.
        if not self.canvas.winfo_ismapped():
            return
        widget = self.winfo_containing(event.x_root, event.y_root)
        while widget is not None:
            if widget is self:
                self.canvas.yview_scroll(int(-event.delta / 120), "units")
                return
            widget = getattr(widget, "master", None)

    def _unbind_wheel(self, _):
        self.canvas.unbind_all("<MouseWheel>")


class NumberControl(ttk.Frame):
    """Одно настраиваемое число: подпись + ползунок + поле + кнопки шага.

    Поле и ползунок связаны; нечисловое или запредельное поле значение не
    меняет — остаётся последнее правильное, а поле подсвечивается.
    """

    def __init__(self, parent, app, label_key, lo, hi, value, step=0.01):
        super().__init__(parent)
        self.app = app
        self.label_key = label_key
        self.lo, self.hi = lo, hi
        self.step = step
        self.value = value
        self._lock = False

        self.lbl = ttk.Label(self, text=T(label_key), wraplength=PANEL_TEXT_W,
                             justify="left")
        # columnspan=5: подпись обязана доставать до последней колонки, иначе
        # grid добирает её 262 px за счёт колонок кнопок, и «+» вылезает за
        # правый край контрола.
        self.lbl.grid(row=0, column=0, columnspan=5, sticky="w")
        self.scale = tk.Scale(self, from_=lo, to=hi, orient="horizontal",
                              # 118, а не 132: контрол целиком занимает 278 px,
                              # и в панели под него ровно 282. Шире — кнопка «+»
                              # уходила за край рамки и обрезалась.
                              length=118, showvalue=False, sliderlength=20,
                              sliderrelief="flat",
                              # resolution=0: ползунок непрерывный, а значение
                              # округляем к шагу сами. Иначе tk.Scale привязывает
                              # его к сетке от lo, и 0.25 превращалось в 0.252.
                              resolution=0, bd=0,
                              # Дорожка почти совпадает с фоном панели, поэтому
                              # вокруг неё обязательна рамка: без неё виден только
                              # бегунок, и непонятно, где границы значения.
                              highlightthickness=1, highlightbackground=COL_FRAME,
                              troughcolor=COL_PAPER, bg=COL_HEAD, fg=COL_TEXT,
                              command=self._from_scale)
        self.scale.grid(row=1, column=0, columnspan=2, sticky="w")
        self.var = tk.StringVar(value=fmt(value))
        self.entry = ttk.Entry(self, width=8, textvariable=self.var, justify="right")
        self.entry.grid(row=1, column=2, sticky="e", padx=(6, 2))
        self.entry.bind("<Return>", self._from_entry)
        self.entry.bind("<FocusOut>", self._from_entry)
        self.btn_down = ttk.Button(self, text="−", width=2, command=lambda: self._nudge(-1))
        self.btn_down.grid(row=1, column=3, sticky="e")
        self.btn_up = ttk.Button(self, text="+", width=2, command=lambda: self._nudge(1))
        self.btn_up.grid(row=1, column=4, sticky="e", padx=(2, 0))
        self.set_value(value, fire=False)

    # ---------- перевод ----------
    def retranslate(self):
        self.lbl.configure(text=T(self.label_key))

    # ---------- значения ----------
    def set_bounds(self, lo, hi):
        self.lo, self.hi = lo, hi
        self.scale.configure(from_=lo, to=hi)

    def set_value(self, value, fire=True):
        """Программная установка (пресет, сброс) — без вызова on_change."""
        self.value = value
        self._lock = True
        self.var.set(fmt(value))
        self.scale.set(clamp(value, self.lo, self.hi))
        self._lock = False
        if fire:
            self.app.param_changed(self)

    def _nudge(self, sign):
        raw = self._round(self.value + sign * self.step)
        self.set_value(raw)

    def _round(self, v):
        """Округлить к шагу и ограничить пределами: ползунок не показывает мусор."""
        v = round(round(v / self.step) * self.step, 10)
        return clamp(v, self.lo, self.hi)

    def _from_scale(self, raw):
        if self._lock:
            return
        self._accept(self._round(float(raw)), good=True)

    def _from_entry(self, _=None):
        if self._lock:
            return
        text = self.var.get().strip().replace(",", ".").replace(MINUS, "-")
        try:
            value = float(text)
        except ValueError:
            if text:
                self._mark(False)
                self.app.bad_input(self, text)
            return
        if value < self.lo - 1e-12 or value > self.hi + 1e-12:
            self._mark(False)
            self.app.out_of_range(self, value)
            self._lock = True
            self.var.set(fmt(self.value))
            self._lock = False
            return
        self._accept(self._round(value), good=True)

    def _accept(self, value, good):
        changed = abs(value - self.value) > 1e-12
        self.value = value
        self._lock = True
        self.var.set(fmt(value))
        self.scale.set(clamp(value, self.lo, self.hi))
        self._lock = False
        self._mark(good)
        if changed:
            self.app.param_changed(self)

    def _mark(self, good):
        self.entry.configure(style="TEntry" if good else "BadEntry.TEntry")


class PowerControl(ttk.Frame):
    """Целая степень: ε = 10⁻ᵏ и β₂ = 1 − 10⁻ᵏ.

    Поле ввода здесь не нужно: диапазоны маленькие, а рукописная «1e-8» легко
    превращается в «1e-80», что уже ничего не объясняет.
    """

    def __init__(self, parent, app, label_key, lo, hi, value, template_key):
        super().__init__(parent)
        self.app = app
        self.label_key = label_key
        self.template_key = template_key
        self.lo, self.hi = lo, hi
        self.value = int(value)
        self._lock = False

        self.lbl = ttk.Label(self, text=T(label_key), wraplength=PANEL_TEXT_W,
                             justify="left")
        self.lbl.grid(row=0, column=0, columnspan=3, sticky="w")
        self.scale = tk.Scale(self, from_=lo, to=hi, orient="horizontal", length=132,
                              # showvalue=False: число уже подписано ниже (ε = 10⁻ᵏ),
                              # а собственное поле значения Scale рисует на тёмном
                              # фоне бегунка — читается плохо.
                              showvalue=False, sliderlength=20, sliderrelief="flat",
                              resolution=1, bd=0,
                              highlightthickness=1, highlightbackground=COL_FRAME,
                              troughcolor=COL_PAPER, bg=COL_HEAD, fg=COL_TEXT,
                              command=self._from_scale)
        self.scale.grid(row=1, column=0, columnspan=3, sticky="w")
        self.note = ttk.Label(self, text="", font=("TkFixedFont", 11),
                              foreground=COL_KEY)
        self.note.grid(row=2, column=0, columnspan=3, sticky="w", pady=(1, 0))
        self.set_value(value, fire=False)

    def retranslate(self):
        self.lbl.configure(text=T(self.label_key))
        self.refresh_note()

    def refresh_note(self):
        p = pow10_text(self.value)
        value = 10.0 ** (-self.value)
        if self.template_key == "eps_text":
            self.note.configure(text=T(self.template_key, p=p))
        else:
            self.note.configure(text=T(self.template_key, p=p, v=f"{1.0 - value:.6f}"))

    def set_value(self, value, fire=True):
        k = int(clamp(round(value), self.lo, self.hi))
        changed = k != self.value
        self.value = k
        self._lock = True
        self.scale.set(k)
        self._lock = False
        self.refresh_note()
        if changed and fire:
            self.app.param_changed(self)

    def _from_scale(self, raw):
        if self._lock:
            return
        self.set_value(float(raw))


class ColorPick(ttk.Frame):
    """Галочка метода с цветным квадратом: цвет тот же, что на графике."""

    def __init__(self, parent, app, opt_key):
        super().__init__(parent)
        self.app = app
        self.opt_key = opt_key
        self._lock = False
        self.swatch = tk.Canvas(self, width=14, height=14, highlightthickness=0,
                               bd=0, bg=COL_BG)
        self.swatch.create_rectangle(1, 1, 13, 13, fill=OPT_COLORS[opt_key],
                                     outline=OPT_COLORS[opt_key])
        self.swatch.grid(row=0, column=0, sticky="w")
        self.var = tk.BooleanVar(value=True)
        self.box = ttk.Checkbutton(self, text=OPT_NAMES[opt_key], variable=self.var,
                                   command=self._toggled)
        self.box.grid(row=0, column=1, sticky="w", padx=(6, 0))

    def _toggled(self):
        if self._lock:
            return
        self.app.optimizer_toggled(self.opt_key, self.var.get())

    def set_state(self, on):
        # Программная установка (пресет, смена языка) не должна запускать
        # optimizer_toggled: эксперимент перестраивает вызывающий код.
        self._lock = True
        self.var.set(on)
        self._lock = False

    def retranslate(self):
        self.box.configure(text=OPT_NAMES[self.opt_key])


def _boxes_hit(a, b, margin=0):
    """Пересекаются ли два прямоугольника холста с запасом margin."""
    return (a[0] < b[2] + margin and b[0] < a[2] + margin and
            a[1] < b[3] + margin and b[1] < a[3] + margin)


def _overlap_area(a, b):
    """Площадь наезда двух прямоугольников; 0, если они не пересекаются.

    Нужна для предпоследнего случая: на минимальном окне свободных мест не
    остаётся вообще, и подписи приходится ставить «где меньше плохо».
    """
    ox = min(a[2], b[2]) - max(a[0], b[0])
    oy = min(a[3], b[3]) - max(a[1], b[1])
    return ox * oy if ox > 0 and oy > 0 else 0


class ContourView(ttk.Frame):
    """Ландшафт: изолинии функции, минимум, старт и настоящие траектории."""

    GRID_COLS, GRID_ROWS = 150, 110
    LEVELS = 14

    def __init__(self, master, app):
        super().__init__(master)
        self.app = app
        self.canvas = tk.Canvas(self, width=540, height=330, bg=COL_PAPER,
                                highlightthickness=1,
                                highlightbackground=COL_FRAME, bd=0)
        self.canvas.pack(fill="both", expand=True)
        self.canvas.bind("<Configure>", lambda e: self.redraw())
        self._bg_key = None
        self._level_boxes = []
        self._scale = (1.0, 0.0, 0.0, 0.0, 0.0)   # unit, left, top, xlo, yhi

    # ---------- пересчёт кадра ----------
    def _frame(self, obj, width, height):
        """Равный масштаб по обеим осям: иначе холм «сплющится» и вывод о форме
        траектории станет неверным."""
        left, right, top, bottom = PAD_L
        plot_w = max(width - left - right, 60)
        plot_h = max(height - top - bottom, 60)
        unit = min(plot_w, plot_h) / (2.0 * obj.radius)
        half_w = plot_w / (2.0 * unit)
        half_h = plot_h / (2.0 * unit)
        cx, cy = obj.center
        self._scale = (unit, left, top, cx - half_w, cy + half_h)
        return unit, left, top, plot_w, plot_h, cx - half_w, cy - half_h, cx + half_w, cy + half_h

    def px(self, x):
        unit, left, top, xlo, yhi = self._scale
        return left + (x - xlo) * unit

    def py(self, y):
        unit, left, top, xlo, yhi = self._scale
        return top + (yhi - y) * unit

    # ---------- изолинии ----------
    def _sample(self, obj, xlo, ylo, xhi, yhi):
        cols, rows = self.GRID_COLS, self.GRID_ROWS
        vals = []
        for j in range(rows):
            y = yhi - (yhi - ylo) * j / (rows - 1)
            row = [obj.loss(xlo + (xhi - xlo) * i / (cols - 1), y) for i in range(cols)]
            vals.append(row)
        return vals

    def _draw_background(self, obj, plot_w, plot_h, xlo, ylo, xhi, yhi):
        c = self.canvas
        vals = self._sample(obj, xlo, ylo, xhi, yhi)
        finite = [v for row in vals for v in row if math.isfinite(v)]
        vmax = max(finite) if finite else 1.0
        levels = obj.levels(self.LEVELS, vmax)
        sx = plot_w / (self.GRID_COLS - 1)
        sy = plot_h / (self.GRID_ROWS - 1)
        left, top = PAD_L[0], PAD_L[2]
        for idx, level in enumerate(levels):
            segs = marching_segments(vals, self.GRID_COLS, self.GRID_ROWS, level)
            color = level_tint(idx, len(levels))
            for (i1, j1), (i2, j2) in segs:
                c.create_line(left + i1 * sx, top + j1 * sy, left + i2 * sx, top + j2 * sy,
                              fill=color, width=1, tags="bg")
        # подписи нескольких уровней, чтобы шкала читалась
        step_lbl = max(1, len(levels) // 4)
        placed = []
        for idx in range(0, len(levels), step_lbl):
            level = levels[idx]
            segs = marching_segments(vals, self.GRID_COLS, self.GRID_ROWS, level)
            if not segs:
                continue
            # метку ставим у самой левой точки изолинии, но не выпускаем за левый
            # край поля: иначе она ложится поверх подписей делений оси y
            i1, j1 = min(segs, key=lambda s: s[0][0])[0]
            x, y = left + i1 * sx, top + j1 * sy
            inside = x < left + 34
            t = c.create_text(x + 4 if inside else x - 4, y, text=level_text(level),
                              font=self.app.fonts["tiny"], fill=COL_HEAD,
                              anchor="w" if inside else "e", tags="bg")
            bb = c.bbox(t)
            if any(_boxes_hit(bb, other, 3) for other in placed):
                # соседняя метка уже заняла это место — вторая только помешает
                c.delete(t)
                continue
            placed.append(bb)
            # светлая подложка: подпись не должна сливаться с пересечёнными
            # изолиниями
            chip = c.create_rectangle(bb[0] - 2, bb[1] - 1, bb[2] + 2, bb[3] + 1,
                                      fill=COL_PAPER, outline="", tags="bg")
            c.tag_lower(chip, t)
        self._level_boxes = placed

    # ---------- главный вывод ----------
    def redraw(self):
        c = self.canvas
        width, height = c.winfo_width(), c.winfo_height()
        # Реальный размер, а не «не меньше 480»: при завышенной оценке изолинии
        # и подпись формулы дорисовывались за правый край холста и обрезались.
        # До первой раскладки холст 1x1 — там рисовать нечего.
        if width < 120 or height < 120:
            return
        obj = self.app.objective
        unit, left, top, plot_w, plot_h, xlo, ylo, xhi, yhi = self._frame(obj, width, height)
        # Кадрирование зависит от стартовой точки (radius), а подписи осей — от
        # языка: оба случая обязаны сбрасывать кэш изолиний.
        key = (width, height, obj.formula, obj.key, round(obj.radius, 6), LANG)
        if key != self._bg_key:
            c.delete("bg")
            c.create_rectangle(left, top, left + plot_w, top + plot_h, fill=COL_PAPER,
                               outline="", tags="bg")
            self._draw_background(obj, plot_w, plot_h, xlo, ylo, xhi, yhi)
            self._draw_axes(left, top, plot_w, plot_h, xlo, ylo, xhi, yhi, obj)
            self._bg_key = key
        c.delete("dyn")
        self._draw_paths(left, top, plot_w, plot_h, obj)

    def _draw_axes(self, left, top, plot_w, plot_h, xlo, ylo, xhi, yhi, obj):
        c = self.canvas
        c.create_line(left, top, left, top + plot_h, fill=COL_LINE, width=2, tags="bg")
        c.create_line(left, top + plot_h, left + plot_w, top + plot_h, fill=COL_LINE,
                      width=2, tags="bg")
        for v in tick_values(xlo, xhi, 7):
            x = self.px(v)
            c.create_line(x, top + plot_h, x, top + plot_h + 5, fill=COL_LINE, tags="bg")
            c.create_text(x, top + plot_h + 17, text=tick_text(v),
                          font=self.app.fonts["tiny"], fill=COL_AXIS_TEXT, tags="bg")
        for v in tick_values(ylo, yhi, 6):
            y = self.py(v)
            c.create_line(left - 5, y, left, y, fill=COL_LINE, tags="bg")
            c.create_text(left - 9, y, text=tick_text(v), font=self.app.fonts["tiny"],
                          fill=COL_AXIS_TEXT, anchor="e", tags="bg")
        # Заголовок и формула — двумя строками: при ширине холста 450 px они
        # не помещаются рядом и начинали наползать друг на друга.
        c.create_text(left, 14, text=T("graph_title"), font=self.app.fonts["midb"],
                      fill=COL_TEXT, anchor="w", tags="bg")
        c.create_text(left + plot_w, 34, text=obj.formula, font=self.app.fonts["small"],
                      fill=COL_HEAD, anchor="e", tags="bg")
        c.create_text(left + plot_w + 6, top + plot_h + 17, text=T("axis_x"),
                      font=self.app.fonts["smallb"], fill=COL_TEXT, anchor="w", tags="bg")
        c.create_text(14, top + plot_h / 2, text=T("axis_y"),
                      font=self.app.fonts["smallb"], fill=COL_TEXT, angle=90, tags="bg")

    def _draw_paths(self, left, top, plot_w, plot_h, obj):
        c = self.canvas
        traces = self.app.traces
        # минимум: крест рисуем сразу, а подпись — после легенды, чтобы она не
        # ушла под белый прямоугольник
        mx, my = self.px(obj.min_theta[0]), self.py(obj.min_theta[1])
        c.create_line(mx - 9, my, mx + 9, my, fill=COL_MIN, width=2, tags="dyn")
        c.create_line(mx, my - 9, mx, my + 9, fill=COL_MIN, width=2, tags="dyn")
        c.create_oval(mx - 6, my - 6, mx + 6, my + 6, outline=COL_MIN, width=1, tags="dyn")
        min_text = (f"θ* = ({tick_text(obj.min_theta[0])}, "
                    f"{tick_text(obj.min_theta[1])})")
        # старт рисуем после легенды и подписываем так, чтобы подпись в неё не
        # попала (см. _place_label)
        s = self.app.theta0
        sxp, syp = self.px(s[0]), self.py(s[1])
        c.create_oval(sxp - 6, syp - 6, sxp + 6, syp + 6, fill="white",
                      outline=COL_START, width=2, tags="dyn")

        off = 0
        marks = [(mx, my), (sxp, syp)]
        for tr in traces:
            pts = []
            for (x, y) in tr.trail:
                if not (math.isfinite(x) and math.isfinite(y)):
                    continue
                px, py = self.px(x), self.py(y)
                if x < self._xlo() or x > self._xhi() or y < self._ylo() or y > self._yhi():
                    off += 1
                pts.extend((px, py))
            if len(pts) >= 4:
                c.create_line(pts, fill=tr.color, width=2, tags="dyn",
                              capstyle="round", joinstyle="round")
            if tr.trail:
                cx, cy = self.px(tr.theta[0]), self.py(tr.theta[1])
                marks.append((cx, cy))
                r = 8 if tr is self.app.focus_trace() else 6
                c.create_oval(cx - r, cy - r, cx + r, cy + r, fill=tr.color,
                              outline="white", width=2, tags="dyn")
                # последний сделанный шаг отдельной стрелкой: видно, куда именно
                # оптимизатор решил сдвинуть параметры
                rec = tr.records[-1] if tr.records else None
                if rec is not None:
                    ax, ay = self.px(rec.theta[0]), self.py(rec.theta[1])
                    if abs(ax - cx) > 3 or abs(ay - cy) > 3:
                        c.create_line(ax, ay, cx, cy, fill=tr.color, width=1,
                                      arrow="last", arrowshape=(8, 10, 4), tags="dyn")
        box = self._legend(left, top, plot_w, plot_h, traces, marks)
        avoid = [r for r in ([box] if box else []) + self._level_boxes if r]
        # Каждая следующая подпись обязана обходить уже поставленные: на
        # минимальном окне «θ* = (0, 0)» и «старт» ложились друг на друга.
        b = self._place_label(mx, my, min_text, self.app.fonts["smallb"], COL_MIN,
                              avoid,
                              ((12, -10, "w"), (12, 14, "w"), (-12, -10, "e"),
                               (-12, 14, "e"), (0, -26, "s"), (0, 24, "n"),
                               (34, -10, "w"), (34, 14, "w"), (-34, -10, "e"),
                               (-34, 14, "e")))
        if b:
            avoid = avoid + [b]
        b = self._place_label(sxp, syp, T("lbl_start"), self.app.fonts["tiny"],
                              COL_START, avoid,
                              ((-10, -10, "e"), (10, -10, "w"), (-10, 12, "e"),
                               (10, 12, "w"), (0, -22, "s"), (0, 20, "n"),
                               (-8, -30, "e"), (8, -30, "w"), (-8, 32, "e"),
                               (8, 32, "w")))
        if off:
            if b:
                avoid = avoid + [b]
            self._place_label(left + plot_w - 6, top + plot_h - 8,
                              T("lbl_offframe", n=off), self.app.fonts["tiny"],
                              COL_BAD_FG, avoid,
                              ((0, 0, "e"), (0, -18, "e"), (0, -36, "e"),
                               (-plot_w + 12, 0, "w")))

    def _place_label(self, x, y, text, font, fill, avoid, candidates):
        """Подписать точку плашкой на цвете холста.

        Перебираем варианты позиции и берём первый, который не задевает легенду,
        подписи уровней и уже поставленные метки. Если свободных мест нет (на
        минимальном окне стартовая точка стоит в щели между легендой и подписью
        минимума), берём вариант с наименьшей площадью наезда: раньше он брался
        «первый по списку», и «старт» ложился ровно на «θ* = (0, 0)».
        """
        c = self.canvas
        t = None
        worst = None
        worst_cost = None
        for dx, dy, anchor in candidates:
            t = c.create_text(x + dx, y + dy, text=text, font=font, fill=fill,
                              anchor=anchor, tags="dyn")
            bb = c.bbox(t)
            cost = sum(_overlap_area(bb, r) for r in avoid)
            if cost == 0:
                break
            if worst_cost is None or cost < worst_cost:
                worst, worst_cost = (dx, dy, anchor), cost
            c.delete(t)
            t = None
        if t is None:
            dx, dy, anchor = worst if worst is not None else candidates[0]
            t = c.create_text(x + dx, y + dy, text=text, font=font, fill=fill,
                              anchor=anchor, tags="dyn")
        x0, y0, x1, y1 = c.bbox(t)
        chip = c.create_rectangle(x0 - 2, y0 - 1, x1 + 2, y1 + 1, fill=COL_PAPER,
                                  outline="", tags="dyn")
        c.tag_lower(chip, t)
        return (x0 - 2, y0 - 1, x1 + 2, y1 + 1)

    def _xlo(self):
        return self._scale[3]

    def _xhi(self):
        unit, left, top, xlo, yhi = self._scale
        return xlo + (self.canvas.winfo_width() - left - PAD_L[1]) / unit

    def _ylo(self):
        unit, left, top, xlo, yhi = self._scale
        return yhi - (self.canvas.winfo_height() - top - PAD_L[3]) / unit

    def _yhi(self):
        return self._scale[4]

    def _legend(self, left, top, plot_w, plot_h, traces, marks=()):
        """Легенда на белой подложке в том углу, где она задевает меньше точек.

        В правом верхнем углу она стояла не потому, что там лучше, а потому,
        что там свободнее на большом холсте. На маленьком холме стартовая
        точка попадала ровно под белый прямоугольник и исчезала с картинки.
        """
        c = self.canvas
        if not traces:
            return
        font = self.app.fonts["small"]
        widths = [font.measure(OPT_NAMES[tr.opt_key]) for tr in traces]
        box_w = max(widths) + 34
        box_h = 18 * len(traces) + 28
        corners = ((left + plot_w - box_w - 6, top + 8),
                   (left + 6, top + 8),
                   (left + plot_w - box_w - 6, top + plot_h - box_h - 6),
                   (left + 6, top + plot_h - box_h - 6))

        def hits(corner):
            bx, by = corner
            r = (bx, by, bx + box_w, by + box_h)
            near = sum(1 for (px, py) in marks
                       if _boxes_hit((px - 7, py - 7, px + 7, py + 7), r))
            over = sum(1 for lb in self._level_boxes if _boxes_hit(lb, r, 2))
            # Сначала угол без чужих надписей: текст на тексте не читается,
            # а закрытая точка ещё видна по своей траектории.
            return (over, near)

        bx, by = min(corners, key=hits)
        c.create_rectangle(bx, by, bx + box_w, by + box_h, fill="white",
                           outline=COL_FRAME, tags="dyn")
        c.create_text(bx + 8, by + 10, text=T("legend_title"),
                      font=self.app.fonts["tiny"], fill=COL_DIM, anchor="w", tags="dyn")
        for i, tr in enumerate(traces):
            yy = by + 26 + i * 18
            c.create_line(bx + 8, yy, bx + 22, yy, fill=tr.color, width=3, tags="dyn")
            c.create_text(bx + 28, yy, text=OPT_NAMES[tr.opt_key], font=font,
                          fill=COL_TEXT, anchor="w", tags="dyn")
        return (bx, by, bx + box_w, by + box_h)


class LossChart(ttk.Frame):
    """График «итерация → потери» по фактическим историям прогонов."""

    def __init__(self, master, app):
        super().__init__(master)
        self.app = app
        self.canvas = tk.Canvas(self, width=540, height=220, bg=COL_PAPER,
                                highlightthickness=1,
                                highlightbackground=COL_FRAME, bd=0)
        self.canvas.pack(fill="both", expand=True)
        self.canvas.bind("<Configure>", lambda e: self.redraw())

    def redraw(self):
        c = self.canvas
        c.delete("all")
        width, height = c.winfo_width(), c.winfo_height()
        # Только фактический размер: завышенный минимум рисовал график шире холста.
        if width < 120 or height < 120:
            return
        # нижнее поле 42: подпись «итерация t» на 38 px упиралась в край холста
        # и обрезалась на один пиксел
        left, right, top, bottom = 74, 18, 34, 42
        plot_w = max(width - left - right, 60)
        plot_h = max(height - top - bottom, 60)
        c.create_text(left, 14, text=T("chart_title"), font=self.app.fonts["midb"],
                      fill=COL_TEXT, anchor="w")
        traces = self.app.traces
        points = [tr for tr in traces if len(tr.losses) > 1]
        if not points:
            c.create_text(left + plot_w / 2, top + plot_h / 2, text=T("det_idle"),
                          font=self.app.fonts["small"], fill=COL_DIM, anchor="c",
                          justify="center", width=plot_w - 20)
            return
        xmax = max(len(tr.losses) - 1 for tr in points)
        values = [v for tr in points for v in tr.losses if math.isfinite(v) and v > 0]
        if not values:
            values = [1e-9]
        log_mode = self.app.var_log.get()
        if log_mode:
            lo_d = math.floor(math.log10(min(values)))
            hi_d = math.ceil(math.log10(max(values))) if max(values) > 1 else 0
            if hi_d <= lo_d:
                hi_d = lo_d + 1
            span = hi_d - lo_d
            if span > 24:
                lo_d = hi_d - 24
            def py(v):
                v = max(v, 10.0 ** lo_d)
                f = (math.log10(v) - lo_d) / (hi_d - lo_d)
                return top + (1.0 - f) * plot_h
        else:
            hi_v = max(values) * 1.08
            lo_v = 0.0

            def py(v):
                f = (v - lo_v) / (hi_v - lo_v) if hi_v > lo_v else 0.0
                return top + (1.0 - f) * plot_h

        def px(i):
            return left + (i / xmax) * plot_w if xmax else left

        # сетка и деления по оси потерь
        if log_mode:
            # Шаг декад зависит от высоты графика, а не от числа декад: на 100 px
            # восемь подписей «10⁻⁷»…«10¹» печатались плотнее строки и наезжали
            # друг на друга.
            step_d = max(1, int(math.ceil(TICK_GAP * (hi_d - lo_d) / float(plot_h))))
            for d in range(lo_d, hi_d + 1, step_d):
                y = py(10.0 ** d)
                c.create_line(left, y, left + plot_w, y, fill=COL_GRID)
                c.create_text(left - 8, y, text="10" + ("⁻" if d < 0 else "") + sup(abs(d)),
                              font=self.app.fonts["tiny"], fill=COL_AXIS_TEXT, anchor="e")
        else:
            for v in tick_values(0.0, max(values) * 1.08, 5):
                y = py(v)
                c.create_line(left, y, left + plot_w, y, fill=COL_GRID)
                c.create_text(left - 8, y, text=tick_text(v), font=self.app.fonts["tiny"],
                              fill=COL_AXIS_TEXT, anchor="e")
        for i in tick_values(0, xmax, 7):
            x = px(i)
            c.create_line(x, top + plot_h, x, top + plot_h + 4, fill=COL_LINE)
            c.create_text(x, top + plot_h + 14, text=str(int(i)),
                          font=self.app.fonts["tiny"], fill=COL_AXIS_TEXT)
        c.create_line(left, top, left, top + plot_h, fill=COL_LINE, width=2)
        c.create_line(left, top + plot_h, left + plot_w, top + plot_h, fill=COL_LINE, width=2)
        c.create_text(left + plot_w, top + plot_h + 30, text=T("chart_axis_iter"),
                      font=self.app.fonts["small"], fill=COL_TEXT, anchor="e")
        c.create_text(12, top + plot_h / 2, text=T("chart_axis_loss"),
                      font=self.app.fonts["small"], fill=COL_TEXT, angle=90)

        for tr in points:
            pts = []
            for i, v in enumerate(tr.losses):
                if not math.isfinite(v):
                    continue
                pts.extend((px(i), py(max(v, 1e-30) if log_mode else v)))
            if len(pts) >= 4:
                c.create_line(pts, fill=tr.color, width=2, smooth=False,
                              capstyle="round", joinstyle="round")
            last_i = len(tr.losses) - 1
            last_v = tr.losses[-1]
            if math.isfinite(last_v):
                x, y = px(last_i), py(max(last_v, 1e-30) if log_mode else last_v)
                c.create_oval(x - 4, y - 4, x + 4, y + 4, fill=tr.color, outline="white",
                              width=1)
        self._legend(left + 8, top + 6, points)

    def _legend(self, bx, by, traces):
        c = self.canvas
        font = self.app.fonts["tiny"]
        box_w = max(font.measure(OPT_NAMES[tr.opt_key]) for tr in traces) + 30
        box_h = 15 * len(traces) + 8
        c.create_rectangle(bx, by, bx + box_w, by + box_h, fill="white",
                           outline=COL_FRAME)
        for i, tr in enumerate(traces):
            yy = by + 9 + i * 15
            c.create_line(bx + 6, yy, bx + 18, yy, fill=tr.color, width=3)
            c.create_text(bx + 23, yy, text=OPT_NAMES[tr.opt_key], font=font,
                          fill=COL_TEXT, anchor="w")


# ---------------------------------------------------------------
# 9. Окно приложения
# ---------------------------------------------------------------
# Порядок выбора задачи и её подписи: сам класс не переводится, переводится
# только пункт списка.
OBJ_KEYS = ("quad", "rosen")
OBJ_LABELS = {"quad": "obj_quad_name", "rosen": "obj_rosen_name"}

# Колонки таблицы сравнения. Их набор простой: потери,
# координаты, номер итерации, расстояние до минимума и статус. ‖g‖ добавлено
# потому, что сходимость засчитывается именно по расстоянию и по градиенту.
CMP_COLS = ("opt", "iter", "loss", "x", "y", "dist", "gnorm", "state")
CMP_HEADS = ("head_opt", "head_iter", "head_loss", "head_x", "head_y",
             "head_dist", "head_gnorm", "head_state")
# Ширины колонок задаёт не список, а _fit_tree_columns(): Treeview обрезает
# ячейку молча, и жёсткий список уже резал английское «converged».
# 10, а не 12: у восьми колонок запас суммируется, и на минимальном окне
# подогнанная таблица выходила на 482 px при видимых 475 — «сошёлся» терял
# последнюю букву уже не внутри колонки, а за краем вьюпорта.
CMP_CELL_PAD = 10          # запас на поля ячейки и заголовка (у clam — 3 px)


class OptimizersLabApp(tk.Tk):
    """Слева настройка эксперимента, в центре холм и график потерь, справа подробности."""

    def __init__(self):
        super().__init__()
        self.configure(bg=COL_BG)
        self.title(T("app_title"))
        screen_w = self.winfo_screenwidth()
        screen_h = self.winfo_screenheight()
        # Минимум не больше экрана: на маленьком ноутбуке окно хотя бы
        # помещается, а макет при этом не режется ниже разумного.
        self.minsize(min(MIN_WIDTH, screen_w - 40), min(MIN_HEIGHT, screen_h - 60))
        self.geometry("%dx%d" % (min(max(1330, int(screen_w * 0.92)), screen_w - 20),
                                min(max(760, int(screen_h * 0.90)), screen_h - 40)))
        try:
            self.state("zoomed")
        except tk.TclError:
            pass
        self._retranslators = []
        self._auto_job = None
        self._build_fonts()
        self._build_styles()

        # Состояние эксперимента: одна задача, один старт, один набор настроек.
        self.obj_key = "quad"
        self.kappa = DEFAULT_KAPPA
        self.rosen_c = 2.0
        self.theta0 = DEFAULT_THETA0
        self.hp = Hyper(DEFAULT_ETA, DEFAULT_BETA, DEFAULT_BETA2_K, DEFAULT_EPS_K)
        self.selected = {key: True for key in OPT_ORDER}
        self.focus_key = "sgd"
        self.help_key = HELP_ORDER[0]
        self.exp_index = 0
        self.trace_map = {}
        self._make_objective()
        self._rebuild_traces()

        self.var_log = tk.BooleanVar(value=True)
        self._build_header()
        self._build_status()
        self._build_body()
        self.protocol("WM_DELETE_WINDOW", self._on_close)
        self._refresh_focus_choices()
        self._render_help()
        self._render_exp()
        self.render_all()
        self.set_status(T("status_ready"))

    # ---------- оформление ----------
    def _mono_family(self):
        known = {f.lower(): f for f in tkfont.families(self)}
        for name in ("consolas", "menlo", "dejavu sans mono", "courier new"):
            if name in known:
                return known[name]
        return "courier"

    def _build_fonts(self):
        base = tkfont.nametofont("TkDefaultFont").actual()
        fam = base["family"]
        size = abs(base["size"]) or 10
        if size < 9:
            size = 9
        mono = self._mono_family()
        # Объекты шрифтов, а не кортежи: canvas и Text меряют строки через
        # font.measure(), а ttk и tk принимают именованный шрифт одинаково.
        table = {
            "tiny": (fam, size - 2, "normal"),
            "small": (fam, size - 1, "normal"),
            "smallb": (fam, size - 1, "bold"),
            "mid": (fam, size, "normal"),
            "midb": (fam, size, "bold"),
            "big": (fam, size + 4, "bold"),
            "mono": (mono, size, "normal"),
            "monob": (mono, size, "bold"),
        }
        self.fonts = {name: tkfont.Font(self, family=spec[0], size=spec[1],
                                       weight=spec[2])
                      for name, spec in table.items()}

    def _build_styles(self):
        style = ttk.Style(self)
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass
        f = self.fonts
        style.configure(".", background=COL_BG, foreground=COL_TEXT)
        style.configure("TFrame", background=COL_BG)
        style.configure("Paper.TFrame", background=COL_PAPER)
        style.configure("TLabelframe", background=COL_BG, bordercolor=COL_FRAME,
                        lightcolor=COL_BG, darkcolor=COL_BG, relief="solid",
                        labelfont=f["smallb"])
        style.configure("TLabelframe.Label", background=COL_BG, foreground=COL_HEAD,
                        font=f["smallb"])
        style.configure("TLabel", background=COL_BG, foreground=COL_TEXT, font=f["mid"])
        style.configure("Title.TLabel", background=COL_BG, foreground=COL_HEAD,
                        font=f["big"])
        style.configure("Dim.TLabel", background=COL_BG, foreground=COL_DIM,
                        font=f["small"])
        style.configure("Key.TLabel", background=COL_BG, foreground=COL_KEY,
                        font=f["mono"])
        style.configure("Min.TLabel", background=COL_BG, foreground=COL_MIN,
                        font=f["small"])
        style.configure("Author.TLabel", background=COL_BG, foreground=COL_DIM,
                        font=f["mono"])
        style.configure("TButton", font=f["mid"], padding=(9, 4))
        style.map("TButton", background=[("active", "#ecece4"), ("disabled", COL_BG)])
        style.configure("Go.TButton", font=f["midb"])
        style.configure("TCheckbutton", background=COL_BG, font=f["mid"])
        style.map("TCheckbutton", background=[("active", COL_BG)])
        style.configure("TRadiobutton", background=COL_BG, font=f["small"])
        style.map("TRadiobutton", background=[("active", COL_BG)])
        style.configure("TNotebook", background=COL_BG, bordercolor=COL_FRAME)
        style.configure("TNotebook.Tab", font=f["small"], padding=(11, 5))
        style.map("TNotebook.Tab",
                  background=[("selected", COL_PAPER), ("!selected", "#f0f0ea")],
                  foreground=[("selected", COL_HEAD), ("!selected", COL_DIM)])
        style.configure("Treeview", background=COL_PAPER, fieldbackground=COL_PAPER,
                        foreground=COL_TEXT, rowheight=22, font=f["small"],
                        bordercolor=COL_FRAME)
        style.configure("Treeview.Heading", font=f["smallb"], background=COL_PAPER,
                        foreground=COL_HEAD, relief="flat", padding=3)
        style.map("Treeview", background=[("selected", "#e4e4da")])
        style.configure("TCombobox", font=f["small"], arrowsize=13, padding=3)
        style.configure("TEntry", font=f["mono"], fieldbackground=COL_PAPER)
        # Поле с неверным вводом: значение не применяется, но видно, что именно
        # пользователь вписал и почему оно не принято.
        style.configure("BadEntry.TEntry", fieldbackground=COL_BAD_BG,
                        foreground=COL_BAD_FG, bordercolor=COL_BAD_FG,
                        lightcolor=COL_BAD_FG, darkcolor=COL_BAD_FG)
        style.configure("TScrollbar", background=COL_BG, troughcolor=COL_BG,
                        bordercolor=COL_FRAME)

    # ---------- служебное ----------
    def _reg(self, fn):
        """Зарегистрировать перестановку текста: вызывается сейчас и при смене языка."""
        fn()
        self._retranslators.append(fn)
        return fn

    def _show(self, widget, text):
        """Показать подпись или убрать её, если текста нет: пустая строка съедает место."""
        widget.configure(text=text)
        if text:
            widget.grid()
        else:
            widget.grid_remove()

    @staticmethod
    def _num(v):
        """Число для панелей: обычно четыре знака, крупным и совсем мелким — степень.

        Верхний порог 10⁴ важнее, чем граница 10⁶ внутри fs(): пока мы
        перепоручали её fs, разошедшийся прогон печатал «−66041.7138» и
        «956593.8000», и две колонки таблицы вырастали так, что статус
        переставал влезать в видимую ширину.
        """
        if v is None or not math.isfinite(v):
            return "∞"
        if abs(v) >= 1e4:
            return exp_text_num(v)
        # Ниже 5·10⁻⁵ четыре знака дают «−0.0000» — ноль со знаком, которого не
        # бывает. Степень здесь только шире колонки (таблица перестала влезать),
        # а смысла в ней для этой панели нет: читателю важнее статус и потери.
        text = f4(v)
        return text[1:] if text.startswith(MINUS + "0.0000") else text

    # ---------- состояние эксперимента ----------
    def _make_objective(self):
        if self.obj_key == "quad":
            self.objective = Quadratic(self.kappa)
        else:
            self.objective = Rosenbrock(self.rosen_c)
        self.objective.set_radius_for(self.theta0)

    def _rebuild_traces(self):
        """Свежие прогоны для выбранных методов: никакого переноса состояния."""
        self.trace_map = {key: Trace(key, self.objective, self.theta0, self.hp)
                          for key in OPT_ORDER if self.selected[key]}
        self._fix_focus()

    def _fix_focus(self):
        if self.focus_key not in self.trace_map:
            for key in OPT_ORDER:
                if key in self.trace_map:
                    self.focus_key = key
                    return

    @property
    def traces(self):
        return [self.trace_map[key] for key in OPT_ORDER if key in self.trace_map]

    def focus_trace(self):
        return self.trace_map.get(self.focus_key)

    # ---------- верх окна ----------
    def _build_header(self):
        head = ttk.Frame(self, padding=(12, 8, 12, 0))
        head.pack(fill="x")
        head.grid_columnconfigure(0, weight=1)
        title = ttk.Label(head, style="Title.TLabel")
        title.grid(row=0, column=0, sticky="w")
        self._reg(lambda: title.configure(text=T("app_title")))
        sub = ttk.Label(head, style="Dim.TLabel")
        sub.grid(row=1, column=0, sticky="w")
        self._reg(lambda: sub.configure(text=T("subtitle")))
        box = ttk.LabelFrame(head, text=T("lang_label"), padding=(6, 3))
        box.grid(row=0, column=1, rowspan=2, sticky="ne", padx=(10, 0))
        self._reg(lambda: box.configure(text=T("lang_label")))
        ru = ttk.Button(box, text="RU", width=4, command=lambda: self.switch_language("ru"))
        ru.grid(row=0, column=0, padx=2)
        en = ttk.Button(box, text="EN", width=4, command=lambda: self.switch_language("en"))
        en.grid(row=0, column=1, padx=2)

    # ---------- три колонки ----------
    def _build_body(self):
        main = ttk.Frame(self, padding=(10, 8, 10, 4))
        main.pack(fill="both", expand=True)
        main.grid_columnconfigure(1, weight=3, minsize=430)
        main.grid_columnconfigure(2, weight=3, minsize=486)
        main.grid_rowconfigure(0, weight=1)
        self._build_left(main)
        self._build_center(main)
        self._build_right(main)

    def _build_left(self, parent):
        col = ScrollFrame(parent, width=LEFT_W)
        col.grid(row=0, column=0, sticky="ns")
        inner = col.inner
        self._box_objective(inner)
        self._box_start(inner)
        self._box_optimizers(inner)
        self._box_params(inner)

    def _box_objective(self, parent):
        box = ttk.LabelFrame(parent, text=T("box_objective"), padding=(10, 2, 10, 6))
        box.pack(fill="x", pady=(0, 8))
        self._reg(lambda b=box: b.configure(text=T("box_objective")))
        box.grid_columnconfigure(0, weight=1)
        lbl = ttk.Label(box, style="Dim.TLabel")
        lbl.grid(row=0, column=0, sticky="w")
        self._reg(lambda: lbl.configure(text=T("lbl_objective")))
        self.obj_combo = ttk.Combobox(box, state="readonly")
        self.obj_combo.grid(row=1, column=0, sticky="we", pady=(1, 4))
        self.obj_combo.bind("<<ComboboxSelected>>", self._objective_chosen)
        self._reg(self._fill_objective_combo)
        self.kappa_ctl = NumberControl(box, self, "lbl_kappa", KAPPA_MIN, KAPPA_MAX,
                                       self.kappa, step=0.5)
        self.kappa_ctl.grid(row=2, column=0, sticky="w")
        self.kappa_note = ttk.Label(box, style="Dim.TLabel", wraplength=PANEL_TEXT_W,
                                    justify="left")
        self.kappa_note.grid(row=3, column=0, sticky="w")
        self.deriv_lbl = ttk.Label(box, font=self.fonts["mono"], foreground=COL_HEAD,
                                   wraplength=PANEL_TEXT_W, justify="left")
        self.deriv_lbl.grid(row=4, column=0, sticky="w", pady=(3, 0))
        self.min_lbl = ttk.Label(box, style="Min.TLabel", wraplength=PANEL_TEXT_W,
                                 justify="left")
        self.min_lbl.grid(row=5, column=0, sticky="w", pady=(2, 0))
        self.stab_lbl = ttk.Label(box, style="Key.TLabel", wraplength=PANEL_TEXT_W,
                                  justify="left")
        self.stab_lbl.grid(row=6, column=0, sticky="w", pady=(3, 0))

    def _box_start(self, parent):
        box = ttk.LabelFrame(parent, text=T("box_start"), padding=(10, 2, 10, 6))
        box.pack(fill="x", pady=(0, 8))
        self._reg(lambda b=box: b.configure(text=T("box_start")))
        self.x0_ctl = NumberControl(box, self, "lbl_x0", -START_SPAN, START_SPAN,
                                    self.theta0[0], step=0.1)
        self.x0_ctl.grid(row=0, column=0, sticky="w")
        self.y0_ctl = NumberControl(box, self, "lbl_y0", -START_SPAN, START_SPAN,
                                    self.theta0[1], step=0.1)
        self.y0_ctl.grid(row=1, column=0, sticky="w")

    def _box_optimizers(self, parent):
        box = ttk.LabelFrame(parent, text=T("box_optimizers"), padding=(10, 2, 10, 6))
        box.pack(fill="x", pady=(0, 8))
        self._reg(lambda b=box: b.configure(text=T("box_optimizers")))
        box.grid_columnconfigure(0, weight=1)
        self.picks = {}
        for i, key in enumerate(OPT_ORDER):
            pick = ColorPick(box, self, key)
            pick.grid(row=i, column=0, sticky="w")
            self.picks[key] = pick
        row = len(OPT_ORDER)
        lbl = ttk.Label(box, style="Dim.TLabel")
        lbl.grid(row=row, column=0, sticky="w", pady=(6, 0))
        self._reg(lambda: lbl.configure(text=T("lbl_focus")))
        self.focus_combo = ttk.Combobox(box, state="readonly")
        self.focus_combo.grid(row=row + 1, column=0, sticky="we", pady=(1, 2))
        self.focus_combo.bind("<<ComboboxSelected>>", self._focus_chosen)
        self.gloss_lbl = ttk.Label(box, style="Dim.TLabel", wraplength=PANEL_TEXT_W,
                                   justify="left")
        self.gloss_lbl.grid(row=row + 2, column=0, sticky="w")

    def _box_params(self, parent):
        box = ttk.LabelFrame(parent, text=T("box_params"), padding=(10, 2, 10, 6))
        box.pack(fill="x", pady=(0, 8))
        self._reg(lambda b=box: b.configure(text=T("box_params")))
        self.eta_ctl = NumberControl(box, self, "lbl_eta", ETA_MIN, ETA_MAX,
                                     DEFAULT_ETA, step=0.005)
        self.eta_ctl.grid(row=0, column=0, sticky="w")
        self.beta_ctl = NumberControl(box, self, "lbl_beta", BETA_MIN, BETA_MAX,
                                      DEFAULT_BETA, step=0.01)
        self.beta_ctl.grid(row=1, column=0, sticky="w")
        self.beta2_ctl = PowerControl(box, self, "lbl_beta2", BETA2_K_MIN, BETA2_K_MAX,
                                      DEFAULT_BETA2_K, "beta2_text")
        self.beta2_ctl.grid(row=2, column=0, sticky="w")
        self.eps_ctl = PowerControl(box, self, "lbl_eps", EPS_K_MIN, EPS_K_MAX,
                                    DEFAULT_EPS_K, "eps_text")
        self.eps_ctl.grid(row=3, column=0, sticky="w")
        shared = ttk.Label(box, style="Dim.TLabel", wraplength=PANEL_TEXT_W, justify="left")
        shared.grid(row=4, column=0, sticky="w", pady=(5, 0))
        self._reg(lambda: shared.configure(text=T("hint_shared")))

    def _build_center(self, parent):
        col = ttk.Frame(parent)
        col.grid(row=0, column=1, sticky="nsew", padx=(8, 8))
        col.grid_columnconfigure(0, weight=1)
        col.grid_rowconfigure(0, weight=3)
        col.grid_rowconfigure(3, weight=2)
        self.contour = ContourView(col, self)
        self.contour.grid(row=0, column=0, sticky="nsew")

        row = ttk.Frame(col)
        row.grid(row=1, column=0, sticky="we", pady=(6, 2))
        # Две строки по две кнопки: в один ряд четыре не влезают никогда —
        # сумме нужно 488 px в русском и 522 в английском, а колонка даёт 414.
        self.step_btn = ttk.Button(row, style="Go.TButton", command=self.do_step)
        self.step_btn.grid(row=0, column=0, sticky="w")
        self.auto_btn = ttk.Button(row, command=self.toggle_auto)
        self.auto_btn.grid(row=0, column=1, sticky="w", padx=(6, 0))
        self.fast_btn = ttk.Button(row, command=self.run_fast)
        self.fast_btn.grid(row=1, column=0, sticky="w", pady=(6, 0))
        self.reset_btn = ttk.Button(row, command=self.do_reset)
        self.reset_btn.grid(row=1, column=1, sticky="w", padx=(6, 0), pady=(6, 0))
        self._reg(lambda: self.step_btn.configure(text=T("btn_step")))
        self._reg(lambda: self.fast_btn.configure(text=T("btn_fast")))
        self._reg(lambda: self.reset_btn.configure(text=T("btn_reset")))
        self._reg(self._refresh_auto_btn)
        note = ttk.Label(row, style="Dim.TLabel", wraplength=NOTE_TEXT_W, justify="left")
        note.grid(row=2, column=0, columnspan=2, sticky="w", pady=(5, 0))
        self._reg(lambda: note.configure(text=T("run_note")))

        head = ttk.Frame(col)
        head.grid(row=2, column=0, sticky="we")
        self.log_check = ttk.Checkbutton(head, variable=self.var_log,
                                         command=self._log_switched)
        self.log_check.grid(row=0, column=0)
        self._reg(lambda: self.log_check.configure(text=T("lbl_log_scale")))

        self.chart = LossChart(col, self)
        self.chart.grid(row=3, column=0, sticky="nsew", pady=(2, 0))

    def _build_right(self, parent):
        self.nb = ttk.Notebook(parent)
        self.nb.grid(row=0, column=2, sticky="nsew")
        self.tab_step = ttk.Frame(self.nb, padding=(8, 6))
        self.tab_compare = ttk.Frame(self.nb, padding=(8, 6))
        self.tab_exp = ttk.Frame(self.nb, padding=(8, 6))
        self.tab_help = ttk.Frame(self.nb, padding=(8, 6))
        for frame, key in ((self.tab_step, "tab_step"), (self.tab_compare, "tab_compare"),
                           (self.tab_exp, "tab_exp"), (self.tab_help, "tab_help")):
            self.nb.add(frame, text=T(key))
        self._reg(self._retitle_tabs)
        self._tab_step(self.tab_step)
        self._tab_compare(self.tab_compare)
        self._tab_exp(self.tab_exp)
        self._tab_help(self.tab_help)

    def _retitle_tabs(self):
        self.nb.tab(self.tab_step, text=T("tab_step"))
        self.nb.tab(self.tab_compare, text=T("tab_compare"))
        self.nb.tab(self.tab_exp, text=T("tab_exp"))
        self.nb.tab(self.tab_help, text=T("tab_help"))

    def _text_panel(self, parent, wrap="none"):
        """Одно читаемый блок текста: только для чтения, с двумя полосами прокрутки."""
        frame = ttk.Frame(parent)
        txt = tk.Text(frame, bg=COL_PAPER, fg=COL_TEXT, relief="flat", bd=0,
                      wrap=wrap, padx=8, pady=6, takefocus=0, highlightthickness=1,
                      highlightbackground=COL_FRAME, insertbackground=COL_TEXT)
        vs = ttk.Scrollbar(frame, orient="vertical", command=txt.yview)
        hs = ttk.Scrollbar(frame, orient="horizontal", command=txt.xview)
        txt.configure(yscrollcommand=vs.set, xscrollcommand=hs.set)
        txt.grid(row=0, column=0, sticky="nsew")
        vs.grid(row=0, column=1, sticky="ns")
        hs.grid(row=1, column=0, sticky="we")
        frame.grid_rowconfigure(0, weight=1)
        frame.grid_columnconfigure(0, weight=1)
        f = self.fonts
        txt.tag_configure("h", font=f["midb"], foreground=COL_HEAD,
                          spacing1=5, spacing3=1)
        txt.tag_configure("k", font=f["monob"], foreground=COL_TEXT, spacing3=1)
        txt.tag_configure("t", font=f["mono"], foreground=COL_TEXT)
        txt.tag_configure("b", font=f["small"], foreground=COL_BAD_FG)
        txt.configure(state="disabled")
        return frame, txt

    def _tab_step(self, tab):
        frame, self.detail_txt = self._text_panel(tab)
        frame.pack(fill="both", expand=True)

    def _tab_compare(self, tab):
        head = ttk.Label(tab, style="Dim.TLabel", wraplength=NOTE_TEXT_W, justify="left")
        head.pack(fill="x")
        self._reg(lambda: head.configure(text=T("cmp_note")))
        grid = ttk.Frame(tab)
        grid.pack(fill="both", expand=True, pady=(5, 4))
        self.tree = ttk.Treeview(grid, columns=CMP_COLS, show="headings",
                                 selectmode="none", height=6)
        bar = ttk.Scrollbar(grid, orient="horizontal", command=self.tree.xview)
        self.tree.configure(xscrollcommand=bar.set)
        self.tree.grid(row=0, column=0, sticky="nsew")
        bar.grid(row=1, column=0, sticky="we")
        grid.grid_rowconfigure(0, weight=1)
        grid.grid_columnconfigure(0, weight=1)
        for i, col in enumerate(CMP_COLS):
            # stretch=False обязателен: растяжимую колонку ttk ужимает ниже
            # подогнанной ширины, когда места не хватает, — и подпись обрезается.
            self.tree.column(col, stretch=False, anchor="w" if i == 0 else "e")
        self._fill_tree_headings()
        for key in OPT_ORDER:
            self.tree.tag_configure(key, foreground=OPT_COLORS[key])
        self._reg(self._fill_tree_headings)
        frame, self.cmp_txt = self._text_panel(tab, wrap="word")
        frame.pack(fill="both", expand=True)

    def _fill_tree_headings(self):
        for col, key in zip(CMP_COLS, CMP_HEADS):
            self.tree.heading(col, text=T(key))
        self._fit_tree_columns()

    def _fit_tree_columns(self):
        """Ширина каждой колонки — по самому длинному заголовку и значению.

        Считается по фактическим шрифтам темы, поэтому и перевод, и смена
        размера чисел (ушедший в степень прогон) не обрезают ячейку.
        """
        if not hasattr(self, "_tree_fonts"):
            # Те же объекты шрифтов, что заданы стилю (см. _build_styles), а не
            # lookup() по имени: в момент сборки темы ttk ещё не успевает
            # подставить жирный шрифт заголовков, и замер давал меньшую ширину.
            f = self.fonts
            self._tree_fonts = (f["smallb"], f["small"])
        fhd, fval = self._tree_fonts
        rows = [self.tree.item(row, "values") for row in self.tree.get_children()]
        for i, (col, key) in enumerate(zip(CMP_COLS, CMP_HEADS)):
            need = fhd.measure(T(key))
            for row in rows:
                need = max(need, fval.measure(str(row[i])))
            w = need + CMP_CELL_PAD
            if self.tree.column(col, "width") != w:
                self.tree.column(col, width=w)

    def _tab_exp(self, tab):
        lbl = ttk.Label(tab, style="Dim.TLabel")
        lbl.pack(anchor="w")
        self._reg(lambda: lbl.configure(text=T("exp_pick")))
        self.exp_var = tk.IntVar(value=0)
        self.exp_radios = []
        box = ttk.Frame(tab)
        box.pack(fill="x", pady=(2, 4))
        for i, key in enumerate(EXP_ORDER):
            rb = ttk.Radiobutton(box, text=exp_text(key)[0], variable=self.exp_var,
                                 value=i, command=self._exp_chosen)
            rb.grid(row=i, column=0, sticky="w")
            self.exp_radios.append(rb)
            self._reg(lambda r=rb, k=key: r.configure(text=exp_text(k)[0]))
        note = ttk.Label(tab, style="Dim.TLabel", wraplength=NOTE_TEXT_W, justify="left")
        note.pack(fill="x", pady=(0, 4))
        self._reg(lambda: note.configure(text=T("exp_note")))
        # Кнопку резервируем ДО расширяющейся панели: pack отдаёт место первым
        # заявленным, иначе при нехватке высоты кнопка сжимается до 18 px.
        self.exp_btn = ttk.Button(tab, style="Go.TButton", command=self.load_preset)
        self.exp_btn.pack(side="bottom", anchor="w", pady=(5, 0))
        self._reg(lambda: self.exp_btn.configure(text=T("exp_load")))
        frame, self.exp_txt = self._text_panel(tab, wrap="word")
        frame.pack(fill="both", expand=True)

    def _tab_help(self, tab):
        row = ttk.Frame(tab)
        row.pack(fill="x")
        lbl = ttk.Label(row, style="Dim.TLabel")
        lbl.grid(row=0, column=0, sticky="w")
        self._reg(lambda: lbl.configure(text=T("help_pick")))
        self.help_combo = ttk.Combobox(row, state="readonly")
        self.help_combo.grid(row=1, column=0, sticky="we", pady=(1, 0))
        self.help_combo.bind("<<ComboboxSelected>>", self._help_chosen)
        row.grid_columnconfigure(0, weight=1)
        prev = ttk.Button(row, text=T("help_nav_prev"), width=3,
                          command=lambda: self._help_move(-1))
        prev.grid(row=1, column=1, padx=(4, 0))
        nxt = ttk.Button(row, text=T("help_nav_next"), width=3,
                         command=lambda: self._help_move(1))
        nxt.grid(row=1, column=2, padx=(2, 0))
        self._reg(self._fill_help_combo)
        frame, self.help_txt = self._text_panel(tab, wrap="word")
        frame.pack(fill="both", expand=True, pady=(5, 0))

    def _fill_help_combo(self):
        titles = [help_text(key)[0] for key in HELP_ORDER]
        self.help_combo.configure(values=titles)
        self.help_combo.current(HELP_ORDER.index(self.help_key))

    def _build_status(self):
        bar = ttk.Frame(self, padding=(12, 2, 12, 6))
        # Полоса статуса резервирует место снизу ДО того, как расширяющееся
        # тело заберёт всю высоту: иначе при нехватке пикселей она уходит за
        # нижний край окна вместе с подписью автора.
        bar.pack(side="bottom", fill="x")
        bar.grid_columnconfigure(0, weight=1)
        self.status_lbl = ttk.Label(bar, style="Dim.TLabel", wraplength=880,
                                    justify="left")
        self.status_lbl.grid(row=0, column=0, sticky="w")
        who = ttk.Label(bar, text=AUTHOR, style="Author.TLabel")
        who.grid(row=0, column=1, sticky="e", padx=(10, 0))

    def set_status(self, text):
        self.status_lbl.configure(text=text)

    # ---------- списки выбора ----------
    def _fill_objective_combo(self):
        values = [T(OBJ_LABELS[key]) for key in OBJ_KEYS]
        self.obj_combo.configure(values=values)
        self.obj_combo.current(OBJ_KEYS.index(self.obj_key))

    def _refresh_focus_choices(self):
        keys = [key for key in OPT_ORDER if key in self.trace_map]
        names = [OPT_NAMES[key] for key in keys]
        self.focus_combo.configure(values=names)
        if keys:
            self.focus_combo.current(keys.index(self.focus_key))
        else:
            self.focus_combo.set("")

    # ---------- обработчики полей ----------
    def param_changed(self, control):
        """Любое изменение настройки перезапускает эксперимент."""
        if control is self.x0_ctl or control is self.y0_ctl:
            self.theta0 = (self.x0_ctl.value, self.y0_ctl.value)
            self.objective.set_radius_for(self.theta0)
            self._restart(T("box_start"))
        elif control is self.kappa_ctl:
            self.kappa = control.value
            self._make_objective()
            self._restart(T("lbl_kappa"))
        else:
            self.hp = Hyper(self.eta_ctl.value, self.beta_ctl.value,
                            self.beta2_ctl.value, self.eps_ctl.value)
            self._restart(T(control.label_key))

    def bad_input(self, control, text):
        self.set_status(T("status_err", field=T(control.label_key), raw=text))

    def out_of_range(self, control, value):
        self.set_status(T("status_range", field=T(control.label_key), raw=fmt(value),
                          lo=fmt(control.lo), hi=fmt(control.hi)))

    def optimizer_toggled(self, key, on):
        self.selected[key] = on
        if on:
            # Метод, включённый в середине опыта, стартует сначала: его состояние
            # не могло накопиться, пока он был выключен.
            self.trace_map[key] = Trace(key, self.objective, self.theta0, self.hp)
            self.focus_key = key
        else:
            self.trace_map.pop(key, None)
        self._fix_focus()
        self._refresh_focus_choices()
        self.render_all()
        if not self.trace_map:
            self.set_status(T("status_noopt"))

    def _focus_chosen(self, _=None):
        keys = [key for key in OPT_ORDER if key in self.trace_map]
        index = self.focus_combo.current()
        if 0 <= index < len(keys):
            self.focus_key = keys[index]
            self.render_all()
            self.set_status(T("status_focus", opt=OPT_NAMES[self.focus_key]))

    def _objective_chosen(self, _=None):
        key = OBJ_KEYS[self.obj_combo.current()]
        if key == self.obj_key:
            return
        self.obj_key = key
        self._make_objective()
        self._apply_start_bounds()
        self._set_kappa_enabled()
        self._restart(None)
        obj = self.objective
        self.set_status(T("status_obj", formula=obj.formula,
                          mx=f4(obj.min_theta[0]), my=f4(obj.min_theta[1])))

    def _log_switched(self):
        self.chart.redraw()

    def _help_chosen(self, _=None):
        index = self.help_combo.current()
        if 0 <= index < len(HELP_ORDER):
            self.help_key = HELP_ORDER[index]
            self._render_help()

    def _help_move(self, delta):
        index = clamp(HELP_ORDER.index(self.help_key) + delta, 0, len(HELP_ORDER) - 1)
        self.help_key = HELP_ORDER[index]
        self.help_combo.current(index)
        self._render_help()

    def _exp_chosen(self):
        self.exp_index = int(self.exp_var.get())
        self._render_exp()

    # ---------- применение настроек ----------
    def _apply_start_bounds(self):
        span = self.objective.start_span
        cx, cy = self.objective.center
        self.x0_ctl.set_bounds(cx - span, cx + span)
        self.y0_ctl.set_bounds(cy - span, cy + span)
        nx = clamp(self.theta0[0], cx - span, cx + span)
        ny = clamp(self.theta0[1], cy - span, cy + span)
        self.theta0 = (nx, ny)
        self.x0_ctl.set_value(nx, fire=False)
        self.y0_ctl.set_value(ny, fire=False)

    def _set_kappa_enabled(self):
        """κ относится только к квадратичному холму: при долине поле выключается."""
        on = self.obj_key == "quad"
        for widget in (self.kappa_ctl.scale, self.kappa_ctl.btn_down,
                       self.kappa_ctl.btn_up):
            widget.configure(state="normal" if on else "disabled")
        self.kappa_ctl.entry.state(["!disabled"] if on else ["disabled"])
        self.kappa_ctl.set_value(self.kappa, fire=False)

    def _restart(self, what):
        """Перезапуск: накопленные моменты относятся к прежним настройкам."""
        self.stop_auto()
        self._rebuild_traces()
        self._refresh_focus_choices()
        self.render_all()
        if what:
            self.set_status(T("status_cfg", what=what))

    # ---------- запуск ----------
    def do_step(self):
        traces = self.traces
        if not traces:
            self.set_status(T("status_noopt"))
            return
        moved = 0
        for tr in traces:
            if tr.step() is not None:
                moved += 1
        trf = self.focus_trace()
        if not moved:
            self.set_status(T("status_nothing", opt=OPT_NAMES[trf.opt_key],
                              status=status_text(trf.status)))
        elif trf.status == S_DIVERGED:
            self.set_status(T("status_diverged", opt=OPT_NAMES[trf.opt_key]))
        elif trf.records:
            rec = trf.records[-1]
            self.set_status(T("status_step", n=rec.t,
                              opt=OPT_NAMES[trf.opt_key], loss=self._num(rec.new_loss),
                              gn=self._num(trf.grad_norm), dist=self._num(trf.dist)))
        if all(tr.finished() for tr in traces) and trf.status != S_DIVERGED:
            self.stop_auto()
            detail = ", ".join(f"{OPT_NAMES[t.opt_key]}: {status_text(t.status)}"
                               for t in traces)
            self.set_status(T("status_step_done", detail=detail))
        self.render_all()

    def toggle_auto(self):
        if self._auto_job is None:
            self.start_auto()
        else:
            self.stop_auto()

    def start_auto(self):
        if not self.traces or all(tr.finished() for tr in self.traces):
            return
        self.set_status(T("status_auto", ms=AUTO_DELAY_MS))
        self._auto_job = self.after(AUTO_DELAY_MS, self._auto_tick)
        self._refresh_auto_btn()

    def _auto_tick(self):
        self._auto_job = None
        if not self.traces or all(tr.finished() for tr in self.traces):
            self._refresh_auto_btn()
            self._refresh_buttons()
            return
        self.do_step()
        if self.traces and not all(tr.finished() for tr in self.traces):
            self._auto_job = self.after(AUTO_DELAY_MS, self._auto_tick)
        self._refresh_auto_btn()

    def stop_auto(self):
        if self._auto_job is not None:
            self.after_cancel(self._auto_job)
            self._auto_job = None
        self._refresh_auto_btn()

    def run_fast(self):
        """Мгновенный перебор до конца: тот же шаг, только без пауз на анимацию."""
        traces = self.traces
        if not traces:
            self.set_status(T("status_noopt"))
            return
        self.stop_auto()
        trf = self.focus_trace()
        before = trf.iterations
        rounds = 0
        while rounds <= MAX_STEPS + 1:
            pending = [tr for tr in traces if not tr.finished()]
            if not pending:
                break
            for tr in pending:
                tr.step()
            rounds += 1
        trf = self.focus_trace()
        self.render_all()
        self.set_status(T("status_fast", n=max(trf.iterations - before, 0),
                          opt=OPT_NAMES[trf.opt_key], status=status_text(trf.status)))

    def do_reset(self):
        self.stop_auto()
        self._rebuild_traces()
        self.render_all()
        self.set_status(T("status_reset"))

    def load_preset(self):
        ps = PRESETS[self.exp_index]
        self.obj_key = ps["obj"]
        self.kappa = ps.get("kappa", DEFAULT_KAPPA)
        self.rosen_c = ps.get("c", 2.0)
        self.theta0 = tuple(ps["theta0"])
        self._make_objective()
        self._apply_start_bounds()
        self._set_kappa_enabled()
        self.obj_combo.current(OBJ_KEYS.index(self.obj_key))
        self.kappa_ctl.set_value(self.kappa, fire=False)
        self.eta_ctl.set_value(ps["eta"], fire=False)
        self.beta_ctl.set_value(ps["beta"], fire=False)
        self.beta2_ctl.set_value(ps["beta2_k"], fire=False)
        self.eps_ctl.set_value(ps["eps_k"], fire=False)
        self.selected = {key: (key in ps["selected"]) for key in OPT_ORDER}
        for key, pick in self.picks.items():
            pick.set_state(self.selected[key])
        self.focus_key = OPT_ORDER[0]
        self.hp = Hyper(self.eta_ctl.value, self.beta_ctl.value,
                        self.beta2_ctl.value, self.eps_ctl.value)
        self._restart(None)
        name, goal = exp_text(ps["key"])
        self.set_status(T("status_preset", name=name, goal=goal))

    # ---------- вывод ----------
    def render_all(self):
        self._refresh_hints()
        self._refresh_buttons()
        self._render_gloss()
        self.contour.redraw()
        self.chart.redraw()
        self._render_detail()
        self._render_compare()

    def _refresh_buttons(self):
        running = bool(self.traces) and not all(tr.finished() for tr in self.traces)
        state = "normal" if running else "disabled"
        self.step_btn.configure(state=state)
        self.fast_btn.configure(state=state)
        self.auto_btn.configure(state="normal" if (running or self._auto_job)
                                else "disabled")

    def _refresh_auto_btn(self):
        self.auto_btn.configure(text=T("btn_stop") if self._auto_job else T("btn_auto"))

    def _refresh_hints(self):
        obj = self.objective
        self.deriv_lbl.configure(text=obj.deriv_text)
        mx, my = self.objective.min_theta
        self.min_lbl.configure(text=T("hint_min", mx=f4(mx), my=f4(my),
                                      ml=f4(obj.min_loss)))
        if obj.key == "quad":
            self._show(self.kappa_note, "")
            limit = 2.0 / obj.b
            eta = self.hp.eta
            if abs(eta - limit) < 1e-9:
                cmp_key = "cmp_equal"
            else:
                cmp_key = "cmp_below" if eta < limit else "cmp_above"
            self._show(self.stab_lbl, T("hint_stability", v=f4(limit), cmp=T(cmp_key)))
        else:
            self._show(self.kappa_note, T("kappa_locked"))
            self._show(self.stab_lbl, "")

    def _render_gloss(self):
        tr = self.focus_trace()
        key = tr.opt_key if tr else self.focus_key
        self.gloss_lbl.configure(text=T("opt_" + key + "_gloss"))

    def _detail_lines(self):
        """Содержимое вкладки «Шаг и расчёт»: настоящие подстановки фокусного метода."""
        tr = self.focus_trace()
        if tr is None:
            return [("t", T("cmp_none"))]
        name = OPT_NAMES[tr.opt_key]
        formulas = T("opt_formulas")[tr.opt_key]
        rec = tr.records[-1] if tr.records else None
        out = []
        if rec is None:
            out.append(("h", T("det_head_next", n=tr.opt.t + 1, opt=name)))
            out.append(("t", T("det_idle")))
        else:
            out.append(("h", T("det_head", n=rec.t, opt=name)))
        out.append(("h", T("formula_hdr")))
        for line in formulas.split("\n"):
            out.append(("k", line))
        out.append(("h", T("chain_grad")))
        theta = rec.theta if rec is not None else tr.theta
        loss = rec.loss if rec is not None else tr.loss
        grad = rec.grad if rec is not None else tr.grad
        out.append(("t", T("tpl_theta_at", th=TH, x=f4(theta[0]), y=f4(theta[1]))))
        out.append(("t", T("tpl_loss_at", x=f4(theta[0]), y=f4(theta[1]), v=self._num(loss))))
        out.append(("t", T("tpl_grad_at", th=TH, x=fs(grad[0]), y=fs(grad[1]),
                           n=fs(norm2(grad)))))
        if rec is not None:
            chain = tr.cls.chain
            group = -1
            for kind, text in rec.trace:
                if kind == "f":
                    group += 1
                    if group + 1 < len(chain):
                        out.append(("h", T(chain[group + 1])))
                    out.append(("k", text))
                else:
                    out.append(("t", text))
            out.append(("h", T("ln_params")))
            out.append(("t", T("tpl_new_theta", th1=TH1, x=f4(rec.new_theta[0]),
                               y=f4(rec.new_theta[1]))))
            out.append(("t", T("tpl_delta", th=TH, th1=TH1, x=fs(rec.delta[0]),
                               y=fs(rec.delta[1]))))
            out.append(("t", T("tpl_new_loss", th1=TH1, v=self._num(rec.new_loss))))
            out.append(("t", T("tpl_dist", a=self._num(rec.dist),
                               b=self._num(rec.dist_new))))
        out.append(("h", T("ln_state")))
        state = tr.snapshot_state()
        if state:
            for key, value in state:
                if isinstance(value, tuple):
                    out.append(("t", T("tpl_state_vec", name=T(key), x=fs(value[0]),
                                       y=fs(value[1]))))
                else:
                    out.append(("t", T("tpl_state_num", name=T(key),
                                       v=str(int(value)) if float(value).is_integer()
                                       else fs(value))))
        else:
            # У чистого SGD памяти нет — и это тоже ответ про состояние метода.
            out.append(("t", T("opt_" + tr.opt_key + "_gloss")))
        out.append(("h", T("ln_status")))
        out.append(("t", status_text(tr.status)))
        if tr.finished():
            out.append(("t", T("det_finished", status=status_text(tr.status))))
        return out

    def _render_detail(self):
        fill_text(self.detail_txt, self._detail_lines())

    def _cell_status(self, status):
        if status == S_LIMIT:
            return T("cell_limit")
        if status == S_STALLED:
            return T("cell_stalled")
        return status_text(status)

    def _render_compare(self):
        tree = self.tree
        tree.delete(*tree.get_children())
        traces = self.traces
        for tr in traces:
            tree.insert("", "end", iid=tr.opt_key, tags=(tr.opt_key,),
                        values=(OPT_NAMES[tr.opt_key], tr.iterations,
                                self._num(tr.loss), self._num(tr.theta[0]),
                                self._num(tr.theta[1]),
                                self._num(tr.dist), self._num(tr.grad_norm),
                                self._cell_status(tr.status)))
        lines = []
        if not traces:
            lines.append(("b", T("cmp_none")))
        else:
            losses = [tr.loss for tr in traces if math.isfinite(tr.loss)]
            if losses:
                lowest = min(losses)
                near = [OPT_NAMES[tr.opt_key] for tr in traces
                        if math.isfinite(tr.loss) and tr.loss <= lowest * 1.05 + 1e-12]
                top = max(tr.iterations for tr in traces)
                if len(near) == 1:
                    lines.append(("t", T("cmp_best", t=top, names=near[0])))
                else:
                    lines.append(("t", T("cmp_tied", names=", ".join(near))))
            done = [OPT_NAMES[tr.opt_key] for tr in traces if tr.status == S_CONVERGED]
            rest = [OPT_NAMES[tr.opt_key] for tr in traces if not tr.finished()]
            lines.append(("t", T("cmp_progress", done=", ".join(done) or "—",
                                 rest=", ".join(rest) or "—")))
            lines.append(("b", T("cmp_caution")))
        fill_text(self.cmp_txt, lines)
        self._fit_tree_columns()

    def _render_exp(self):
        name, goal = exp_text(EXP_ORDER[self.exp_index])
        fill_text(self.exp_txt, [("h", name), ("t", goal)])

    def _help_lines(self, key):
        """Раздел справки в тегах: «» — формула, ! — предупреждение и его продолжения."""
        title, lines = help_text(key)
        out = [("h", title)]
        warn = False
        for line in lines:
            if line.startswith("»"):
                warn = False
                out.append(("k", line))
            elif line.startswith("!"):
                warn = True
                out.append(("b", line))
            else:
                out.append(("b" if warn else "t", line))
        return out

    def _render_help(self):
        fill_text(self.help_txt, self._help_lines(self.help_key))

    # ---------- смена языка ----------
    def switch_language(self, code):
        """Перевод без перезапуска: задача, настройки, итерации и вкладка остаются."""
        if code == LANG:
            self.set_status(T("status_lang"))
            return
        set_language(code)
        for fn in self._retranslators:
            fn()
        for ctl in (self.kappa_ctl, self.x0_ctl, self.y0_ctl, self.eta_ctl,
                    self.beta_ctl, self.beta2_ctl, self.eps_ctl):
            ctl.retranslate()
        for pick in self.picks.values():
            pick.retranslate()
        self._refresh_focus_choices()
        self._render_help()
        self._render_exp()
        self.title(T("app_title"))
        self.render_all()
        self.set_status(T("status_lang"))

    def _on_close(self):
        self.stop_auto()
        self.destroy()


# ---------------------------------------------------------------
# 10. Точка входа
# ---------------------------------------------------------------
def main():
    app = OptimizersLabApp()
    app.mainloop()


if __name__ == "__main__":
    main()
