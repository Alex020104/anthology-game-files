from __future__ import annotations

import json
import os
import re
import time
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import full_sources


ROOT = Path(__file__).resolve().parent
KNOWLEDGE_DIR = ROOT / "knowledge"
HOST = os.environ.get("ANTHOLOGY_AI_HOST", "127.0.0.1")
PORT = int(os.environ.get("ANTHOLOGY_AI_PORT", "8787"))
MODEL = os.environ.get("ANTHOLOGY_AI_MODEL", "gpt-4.1-mini")
RATE_SECONDS = int(os.environ.get("ANTHOLOGY_AI_RATE_SECONDS", "20"))
ANTHOLOGY_CLOUD_AI_URL = os.environ.get("ANTHOLOGY_CLOUD_AI_URL", "").strip()
ANTHOLOGY_CLOUD_AI_TOKEN = os.environ.get("ANTHOLOGY_CLOUD_AI_TOKEN", "").strip()
MAX_QUESTION_CHARS = 700
MAX_ANSWER_CHARS = 1100

last_request_by_ip: dict[str, float] = {}
conversation_context_by_ip: dict[str, str] = {}


def load_knowledge() -> str:
    parts: list[str] = []
    for path in sorted(KNOWLEDGE_DIR.rglob("*.md")):
        parts.append(f"## {path.relative_to(KNOWLEDGE_DIR).as_posix()}\n{path.read_text(encoding='utf-8', errors='replace')}")
    return "\n\n".join(parts)


KNOWLEDGE = load_knowledge()


def normalize_query(text: str) -> str:
    text = (text or "").lower().replace("ё", "е")
    return re.sub(r"\s+", " ", text).strip()


def is_english_question(text: str) -> bool:
    raw = text or ""
    latin = len(re.findall(r"[A-Za-z]", raw))
    cyrillic = len(re.findall(r"[А-Яа-яЁё]", raw))
    if latin == 0:
        return False
    if cyrillic == 0:
        return True
    return latin >= cyrillic * 2


def knowledge_snippet(question: str) -> str | None:
    q = normalize_query(question)
    if not q:
        return None

    words = [w for w in re.findall(r"[a-zа-я0-9_+-]{3,}", q, re.IGNORECASE) if w not in {
        "что", "как", "где", "это", "мне", "надо", "нужно", "можно", "если", "или", "для", "при", "про"
    }]
    if not words:
        return None

    best_score = 0
    best_block = None
    for path in sorted(KNOWLEDGE_DIR.rglob("*.md")):
        text = path.read_text(encoding="utf-8", errors="replace")
        blocks = re.split(r"\n(?=##? )", text)
        for block in blocks:
            nblock = normalize_query(block)
            score = sum(1 for word in words if word in nblock)
            if score > best_score:
                best_score = score
                best_block = block

    if best_score < 2 or not best_block:
        return None

    lines = []
    for line in best_block.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        lines.append(line)
        if len(" ".join(lines)) > 850:
            break
    return " ".join(lines) if lines else None


def local_fallback_answer_en(question: str) -> str:
    q = normalize_query(question)

    if any(word in q for word in ["third person", "3rd person", "third-person", "camera mode", "camera view", "cam_1", "cam_2", "cam_3"]):
        return (
            "Third-person view in Anomaly/Anthology is switched with the camera keys: "
            "Left Arrow — cam_1, Down Arrow — cam_2, Right Arrow — cam_3. "
            "Try Left/Down/Right Arrow. Camera zoom is T, zoom out is ]. "
            "If it does not work, open Settings -> Controls -> Camera and check cam_1/cam_2/cam_3 bindings."
        )

    if any(word in q for word in ["jam", "unjam", "weapon stuck", "misfire", "wpo"]):
        return (
            "Weapon jam/unjam is configured in MCM. Go to MCM -> WPO -> WPO weapon -> unjam key. "
            "Also check MCM -> MCM MENU -> All assigned keys -> WPO: inspect / unjam. "
            "The default is often F, but it is safer to set a separate press mode, for example double tap, "
            "so it does not conflict with use/search actions."
        )

    if any(word in q for word in ["controls", "keybind", "hotkey", "mcm", "settings"]):
        return (
            "Basic controls are changed in Main menu -> Settings -> Controls. MCM is also part of game settings "
            "for modded features. Common keys: W/A/S/D movement, Space jump, Left Ctrl crouch, Left Shift sprint, "
            "F use, I inventory, P tasks, Tab status/PDA table, Esc menu. Mod hotkeys are usually in "
            "MCM -> MCM MENU -> All assigned keys."
        )

    if any(word in q for word in ["inventory", "backpack"]):
        return "Inventory opens with I. Backpack animations are configured separately in MCM -> Animat - Animations -> Backpack."

    if any(word in q for word in ["flashlight", "torch", "night vision", "nvg", "detector"]):
        return (
            "Flashlight is usually L, night vision/NVG is N, detector is O. "
            "If a key does nothing, check that you have the required device equipped and check MCM/keybind conflicts."
        )

    if any(word in q for word in ["quicksave", "quick save", "quickload", "quick load", "f5", "f9"]):
        return "Quicksave is F5, quickload is F9. Quick item slots are F1, F2, F3 and F4."

    if any(word in q for word in ["mouse", "sensitivity", "invert"]):
        return (
            "Mouse sensitivity is changed in the control/mouse settings. In user.ltx the common values are "
            "mouse_sens, mouse_sens_aim, mouse_sens_vertical and mouse_invert."
        )

    if any(word in q for word in ["hud", "crosshair", "interface", "fov"]):
        return (
            "HUD/crosshair/FOV depend on the game and modpack settings. Common user.ltx values: hud_crosshair, "
            "hud_draw, cl_dynamiccrosshair, fov and hud_fov. Beginners should change these through the menu when possible."
        )

    if any(word in q for word in ["fps", "performance", "stutter", "microfreeze", "micro freeze", "lag"]):
        return (
            "For FPS/stutters, first try disabling the heavy graphics mods: [GFX] Enhanced Shaders & Color Grading, "
            "[GFX] Beefs NVGs Shaders and [GFX] ScreenSpaceShaders Update 23.5. Also reduce grass, shadows, view distance and shaders. "
            "For microstutters, use FreeSync/G-Sync if available and cap FPS 1-2 frames below monitor refresh rate."
        )

    if any(word in q for word in ["download", "links", "where download", "install"]):
        return (
            "Download links are in the Anthology Discord. Right-click the server, enable 'Show all channels', "
            "then check the pinned messages and download/link channels. I will not invent direct download links."
        )

    if any(word in q for word in ["launcher", "update", "github", "permission denied", "chat.exe"]):
        return (
            "If the launcher/update fails: close the game, launcher and Chernobyl Relay Chat, then try again. "
            "For GitHub connection errors, try changing VPN state/server. Permission denied for Chernobyl Relay Chat.exe "
            "usually means the chat is still running in the background."
        )

    if any(word in q for word in ["standard", "hard", "profile", "profiles"]):
        return (
            "MO2 profile warning: switching from Standard to Hard is allowed at any time. "
            "Switching back from Hard to Standard is not safe and can break saves; do it only with a new game."
        )

    if any(word in q for word in ["modpack", "original", "difference"]):
        return (
            "Original is the lighter Anomaly base with Anthology storylines, suitable for weaker PCs and old renderers. "
            "Modpack is the DX11 version with the weapon pack, graphics, gameplay features and heavier mechanics."
        )

    if any(word in q for word in ["mags", "magazine", "reload"]):
        return "To enable/disable the magazine system, toggle Mags Redux in MO2. Direct magazine loading is usually MOUSE4 in MCM keybinds."

    if any(word in q for word in ["skills", "perks", "pda"]):
        return "The skills/perks menu is inside the PDA."

    return (
        "I do not have an exact local answer for this yet. Please ask in the Anthology Discord, "
        "then add the confirmed answer to the helper knowledge guides."
    )


def split_sentences(text: str) -> list[str]:
    cleaned = re.sub(r"\s+", " ", text or "").strip()
    if not cleaned:
        return []
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", cleaned) if s.strip()]


def local_story_answer_from_context(question: str, context: dict) -> str:
    title = context.get("title") or "Источник"
    source = context.get("source") or "гайд"
    text = context.get("text") or ""
    q = (question or "").casefold().replace("ё", "е")
    sentences = split_sentences(text)
    direct = ""
    if any(word in q for word in ("убить", "перебить", "застрелить", "атаковать")):
        if re.search(r"\b(убить|перебить|расправ|атак|рейд|бой|бандит)", text.casefold().replace("ё", "е")):
            direct = "Да, можно."
        else:
            direct = "В тексте гайда прямого варианта с убийством не подтверждено."
    elif any(word in q for word in ("спасти", "жив", "выживет")):
        low = text.casefold().replace("ё", "е")
        if any(mark in low for mark in ("не удалось", "погиб", "мертв", "мёртв", "умер")):
            direct = "Судя по гайду, нет — спасти не получится."
        elif any(mark in low for mark in ("спасти", "выручить", "выживет", "освободить")):
            direct = "Да, по гайду это можно сделать."
    elif any(word in q for word in ("можно", "можно ли", "получится")):
        direct = "По гайду — да, если выполнить описанный вариант." if sentences else ""

    consequence_words = (
        "если", "после", "когда", "в итоге", "тогда", "награ", "получ", "вариант",
        "выберите", "придется", "придётся", "вернит", "отпуст", "начнут", "обыск",
        "рейд", "перебить", "выкуп", "обмен",
    )
    picked = []
    for sentence in sentences:
        low = sentence.casefold().replace("ё", "е")
        if any(word in low for word in consequence_words):
            picked.append(sentence)
        if len(picked) >= 4:
            break
    if not direct:
        picked = sentences[:4]
    elif not picked:
        picked = sentences[:4]
    answer = (" ".join([direct, " ".join(picked)]).strip() if direct else " ".join(picked).strip())
    return f"{title} ({source}): {trim_answer(answer)}"


def local_fallback_answer(question: str) -> str:
    full_context = full_sources.find_context(question, str(ROOT))
    if full_context:
        return local_story_answer_from_context(question, full_context)
    if is_english_question(question):
        return local_fallback_answer_en(question)

    q = normalize_query(question)

    if any(word in q for word in ["клин", "заклин", "заело", "осеч", "устранить", "убрать клин", "снять клин", "не стреляет", "wpo"]):
        return (
            "Если оружие заклинило: действие клина настраивается в MCM, это тоже настройки игры. "
            "Путь: MCM -> WPO -> WPO оружие -> Клавиша снятия клина. Обычно стоит F, но лучше поставить отдельный режим, "
            "например Двойное нажатие, чтобы не конфликтовало с обыском/использованием. "
            "Также проверь MCM -> MCM MENU -> Все назначенные клавиши -> WPO: Осмотр / устранить клин. "
            "Для механики клинов рекомендуется включить в MCM -> WPO -> WPO оружие: Разрешить \"супер-клин\", "
            "\"Выброс\" магазина из оружия при \"супер-клине\" и \"Старые боеприпасы влияют на деградацию\"."
        )

    if any(word in q for word in ["провер", "патрон", "магазин", "занятые руки", "занятых рук", "осмотр оруж", "анимац оруж"]):
        return (
            "Проверка патронов и осмотр оружия настраиваются в MCM -> Проверка патронов -> Проверка патронов. "
            "Назначь удобную клавишу проверки патронов, по умолчанию это MINUS или клавиша '-'. "
            "Пункт \"Исправление занятых рук\" нужно отключить галочкой: этот мод включает красивую анимацию проверки магазина "
            "и анимацию осмотра оружия по клавише F."
        )

    if any(word in q for word in ["прямая заряд", "зарядка", "зарядить магазин", "mags redux", "mags", "магазины"]):
        return (
            "Прямая зарядка магазинов: MCM -> MCM MENU -> Все назначенные клавиши -> Клавиша прямой зарядки. "
            "По умолчанию MOUSE4. Она заряжает магазины в один клик из разгрузки. "
            "Важно: функция работает только при включённом аддоне [WPN][MAG][R.A.K Weapon Pack Adaptation Global A.N.T.H.O.L.O.G.Y Mags Redux]."
        )

    if any(word in q for word in ["управлен", "кнопк", "клавиш", "бинд", "controls", "control"]):
        return (
            "Базовое управление меняется в игре: Главное меню -> Настройки -> Управление. MCM тоже считается настройками игры для модовых функций. "
            "Основные кнопки: W/A/S/D — движение, Space — прыжок, Left Ctrl — присесть, Left Shift — бег, "
            "F — использовать, I — инвентарь, P — активные задания, Tab — статус/таблица, Esc — меню. "
            "Модовые клавиши ищи в MCM -> MCM MENU -> Все назначенные клавиши."
        )

    if any(word in q for word in ["инвент", "inventory", "рюкзак"]):
        return "Инвентарь открывается клавишей I. Если нужна анимация рюкзака, она настраивается отдельно в MCM -> Animat - Анимации -> Рюкзак."

    if any(word in q for word in ["фонарь", "фонар", "torch", "пнв", "ночн", "nvg", "детектор"]):
        return (
            "Фонарь по умолчанию включается на L. ПНВ / ночное видение — на N. Детектор достаётся на O. "
            "Если не работает, проверь наличие нужного устройства/экипировки и конфликты клавиш в MCM."
        )

    if any(word in q for word in ["быстрое сохран", "быстрый сейв", "quick save", "quicksave", "загруз", "quickload", "f5", "f9"]):
        return "Быстрое сохранение — F5, быстрая загрузка — F9. Быстрые слоты предметов: F1, F2, F3, F4."

    if any(word in q for word in ["мыш", "сенс", "чувств", "sensitivity", "инвер", "mouse"]):
        return (
            "Чувствительность мыши меняется в настройках управления/мыши. В user.ltx параметры: mouse_sens — общая чувствительность, "
            "mouse_sens_aim — чувствительность при прицеливании, mouse_sens_vertical — вертикальная, mouse_invert — инверсия."
        )

    if (
        any(word in q for word in [
            "треть", "третье лицо", "третьего лица", "3 лицо", "3 лица", "3его",
            "3-е", "вид от третьего", "вид от 3", "от 3 лица", "third person",
            "режим лица", "режим камеры", "cam_1", "cam_2", "cam_3"
        ])
        or ("лиц" in q and any(word in q for word in ["режим", "вид", "камера", "переключ", "включ"]))
    ):
        return (
            "Вид от третьего лица в Anomaly/Anthology переключается клавишами камер. В текущей сборке: "
            "Left Arrow — cam_1, Down Arrow — cam_2, Right Arrow — cam_3. "
            "Попробуй стрелки влево/вниз/вправо. Приблизить камеру можно на T, отдалить на ]. "
            "Если не работает — открой Настройки -> Управление -> Камера и проверь назначения cam_1/cam_2/cam_3."
        )

    if any(word in q for word in ["прицел", "перекрест", "crosshair", "худ", "hud", "интерфейс"]):
        return (
            "HUD и перекрестие зависят от настроек игры и сборки. Основные параметры: hud_crosshair — перекрестие, "
            "hud_crosshair_dist — дистанция, hud_draw — общий HUD, hud_fov — FOV рук/оружия, cl_dynamiccrosshair — динамическое перекрестие. "
            "В Anthology обычное перекрестие может быть выключено для иммерсивности."
        )

    if any(word in q for word in ["настройк игры", "настройки игры", "оригинальн", "anomaly settings", "user.ltx", "автоподбор", "автоперезар", "fov"]):
        return (
            "Настройки игры Anomaly — это меню игры, user.ltx и MCM для модовых функций. Частые параметры: "
            "g_game_difficulty — сложность, g_autopickup — автоподбор, g_auto_reload — автоперезарядка, "
            "hud_draw — общий HUD, hud_crosshair — перекрестие, cl_dynamiccrosshair — динамическое перекрестие, "
            "fov — FOV камеры, hud_fov — FOV рук/оружия, mouse_sens — чувствительность мыши. "
            "Новичкам лучше менять это через меню, а не вручную в user.ltx."
        )

    if any(word in q for word in ["оруж", "стрел", "перезар", "огонь", "прицелив", "режим огня"]):
        return (
            "Оружие: Mouse1 — огонь, Mouse2 — прицеливание, R — перезарядка, V — функция оружия, "
            "1-6 — оружейные слоты, Y — следующее оружие. Режимы огня: Numpad4 и 0. "
            "Клины, осмотр, магазины и лазеры дополнительно настраиваются в MCM."
        )

    if any(word in q for word in ["сложност", "difficulty", "нович", "мастер"]):
        return (
            "Сложность лучше менять через меню игры или при старте новой игры. В user.ltx она хранится как g_game_difficulty. "
            "Новичкам лучше не редактировать user.ltx вручную, если не понимают последствия."
        )

    if any(word in q for word in ["стандарт", "standard", "хард", "hard", "профиль", "профиля"]):
        return (
            "По профилям MO2 важно: со Стандарта на Хард можно переключаться в любой момент, "
            "а с Харда обратно на Стандарт нельзя. Обратный переход может вызвать краши и поломку сохранений. "
            "Если нужно перейти с Харда на Стандарт — делай это только с новой игрой."
        )

    if any(word in q for word in ["хоткей", "hotkey", "горяч", "клавиш", "mcm", "бинд", "назнач"]):
        return (
            "Горячие клавиши настраиваются в MCM. Основные пути: колесо быстрых действий — "
            "MCM -> MCM MENU -> Все назначенные клавиши -> OAW: Колесо быстрых действий, по умолчанию MOUSE5. "
            "Прямая зарядка — MCM -> MCM MENU -> Все назначенные клавиши -> Клавиша прямой зарядки, по умолчанию MOUSE4. "
            "Осмотр оружия/клин — MCM -> WPO или MCM MENU -> Все назначенные клавиши, часто стоит F, лучше разделить режимом нажатия."
        )

    if any(word in q for word in ["мини-карт", "миникарт", "mini", "карта"]):
        return (
            "Мини-карта переключается через MCM -> Переключатель mini-карты -> Переключатель mini-карты -> "
            "Клавиша переключения mini-карты. По умолчанию Z. Если хочешь, чтобы карта была видна только пока зажата кнопка, "
            "выбери режим нажатия Удержание."
        )

    if any(word in q for word in ["paw", "маркер", "метк", "точк", "маршрут"]):
        return (
            "Для маркеров PAW зайди: MCM -> PAW -> Контекстное меню. Поставь значение Всегда для пунктов: "
            "Настройки точек маршрута, Настройки маркера, Добавить маркер, Удалить маркер."
        )

    if any(word in q for word in ["bhs", "здоров", "hud", "худ", "позиция"]):
        return (
            "BHS настраивается в MCM -> Body Health System -> HUD. Там выбирается Тип HUD и позиции HUD по X/Y. "
            "Важно: после ввода числовых значений позиции обязательно нажми Enter прямо в строке, чтобы игра их зафиксировала."
        )

    if any(word in q for word in ["анимац", "погруж", "рюкзак", "обыск", "мутант", "подбор", "шлем"]):
        return (
            "Анимации персонажа находятся в MCM -> Animat - Анимации. Там можно включить/отключить анимацию рюкзака, "
            "разделки мутантов, подбора и обыска тел. Для лишней анимации шлема зайди в Головные уборы -> "
            "Режим строгих шлемов и сними галочку, если не нужна долгая анимация снятия/надевания."
        )

    if any(word in q for word in ["супер-клин", "суперклин"]):
        return (
            "Опции клина оружия находятся в MCM -> WPO -> WPO оружие. Рекомендуется включить: "
            "Разрешить \"супер-клин\", \"Выброс\" магазина из оружия при \"супер-клине\", "
            "Старые боеприпасы влияют на деградацию. Осмотр/устранение клина часто стоит на F, лучше выбрать отдельный режим нажатия."
        )

    if any(word in q for word in ["лазер", "лцу", "фильтр", "противогаз", "glow", "палоч"]):
        return (
            "Лазер ЛЦУ: MCM -> Лазеры на основе BaS -> Клавиша включения лазера, по умолчанию L, лучше режим Удержание. "
            "Фильтр противогаза: MCM -> MCM MENU -> Все назначенные клавиши -> Клавиша снятия / установки фильтра. "
            "Светящиеся палочки: MCM -> Химический источник света -> GLOWSTICKS -> Клавиша броска, по умолчанию NUMPAD5."
        )

    found = knowledge_snippet(question)
    if found:
        return found

    if any(word in q for word in ["микрофриз", "микро фриз", "статтер", "stutter", "подерг", "фриз"]):
        return (
            "Если есть микро-фризы/статтеры: включи FreeSync/G-Sync в мониторе и драйвере, "
            "а FPS ограничь на 1–2 кадра ниже герцовки монитора: 60 Гц → 58 FPS, 75 Гц → 74 FPS, "
            "120 Гц → 118 FPS, 144 Гц → 142 FPS. Если монитор обычный — поставь лимит 58–59 FPS "
            "через игру или RivaTuner и включи V-Sync."
        )

    if any(word in q for word in ["fps", "фпс", "производ", "просад", "лага", "тормоз"]):
        return (
            "Для поднятия FPS в первую очередь отключи 3 тяжёлых графических мода: "
            "[GFX] Enhanced Shaders & Color Grading, [GFX] Beefs NVGs Shaders, "
            "[GFX] ScreenSpaceShaders Update 23.5. Также снизь траву, тени, дальность и шейдеры."
        )

    if any(word in q for word in ["скач", "ссылк", "download", "где игра", "откуда"]):
        return (
            "Ссылки на скачивание находятся в Discord Anthology. Нажми ПКМ по серверу и включи "
            "«Отобразить все каналы», затем смотри каналы со ссылками и закрепы. Прямые ссылки я не выдумываю."
        )

    if any(word in q for word in ["7zip", "7-zip", "ошибка 2", "архив", "crdownload"]):
        return (
            "Ошибка 7-Zip/ошибка 2 обычно значит, что архив с Яндекс Диска скачался не полностью. "
            "Дождись полного завершения загрузки, проверь что нет .crdownload, собери все 3 файла в одной папке "
            "без лишних символов в названии и только потом запускай установку."
        )

    if any(word in q for word in ["не обнов", "обновлен", "лаунчер", "github", "гитхаб"]):
        return (
            "Если лаунчер не обновляется: закрой игру, лаунчер и Chernobyl Relay Chat, затем попробуй снова. "
            "Если ошибка связана с GitHub — включи/выключи VPN или смени страну/сервер VPN. "
            "Если Permission denied на Chernobyl Relay Chat.exe — чат ещё запущен в фоне."
        )

    if any(word in q for word in ["permission", "denied", "доступ", "заменить", "chat.exe"]):
        return (
            "Permission denied на Chernobyl Relay Chat.exe означает, что файл занят. "
            "Закрой Chernobyl Relay Chat, проверь диспетчер задач и повтори обновление через лаунчер."
        )

    if any(word in q for word in ["оригинал", "модпак", "modpack", "отлич"]):
        return (
            "Оригинал — базовая Anomaly с сюжетами Anthology, легче для слабых ПК и может работать на старых рендерах. "
            "Модпак — DX11-версия с оружейным паком, графикой, геймплейными фичами и более тяжёлыми механиками."
        )

    if any(word in q for word in ["hard", "хард", "bhs", "холод", "профил"]):
        return (
            "HARD-профиль включает BHS, холод, болезни, фильтры радиации, баллоны для лабораторий, "
            "жёсткую экономику, бартер и более сложное поведение NPC. Важно: BHS и систему холода включать/отключать строго вместе."
        )

    if any(word in q for word in ["осень", "раститель", "season", "seasons"]):
        return "Чтобы поменять растительность на осень, в MO2 включи мод в разделе [SEA] Seasons."

    if any(word in q for word in ["магазин", "mags", "redux", "перезар"]):
        return "Чтобы включить или отключить систему магазинов, в MO2 включи/отключи модуль Mags Redux."

    if any(word in q for word in ["сюжет", "сюжетк", "линия", "фриплей"]):
        return (
            "Сюжетные линии выбираются при старте новой игры: выбери фракцию, открой список локаций "
            "и выбери нужную сюжетку. Во многих сюжетах после прохождения доступен переход во фриплей."
        )

    if any(word in q for word in ["скилл", "перки", "перк", "навык", "кпк"]):
        return "Меню навыков и перков находится в КПК."

    if any(word in q for word in ["чат", "relay", "слеш", "/lox", "сообщен"]):
        return (
            "Chernobyl Relay Chat связывает игровой чат Anthology с внешним чатом. "
            "Если сообщение начинается со слеша, старые версии могли считать его командой. "
            "В исправленной версии неизвестные /слова отправляются как обычный текст."
        )

    return (
        "В локальной базе пока нет точного ответа на этот вопрос. Лучше уточнить в Discord Anthology, "
        "а потом добавить ответ в knowledge-гайды помощника."
    )


def trim_answer(text: str) -> str:
    text = " ".join((text or "").replace("\r", "\n").split())
    if len(text) > MAX_ANSWER_CHARS:
        return text[: MAX_ANSWER_CHARS - 1].rstrip() + "…"
    return text


def looks_like_followup(question: str) -> bool:
    q = normalize_query(question)
    words = re.findall(r"[a-zР°-СЏ0-9][a-zР°-СЏ0-9_+\\-]{1,}", q)
    followup_words = (
        "РѕРЅ", "РѕРЅР°", "РѕРЅРё", "РµРіРѕ", "РµРµ", "РµС‘", "РёС…", "С‚Р°Рј", "С‚СѓРґР°", "РґР°Р»СЊС€Рµ",
        "РїРѕС‚РѕРј", "РїРѕСЃР»Рµ", "СЃРїР°СЃС‚Рё", "РјРµСЂС‚РІ", "РјС‘СЂС‚РІ", "РЅР°С€РµР»", "РЅР°С€С‘Р»",
        "РєСѓРґР°", "РєР°Рє Р±С‹С‚СЊ", "С‡С‚Рѕ РґРµР»Р°С‚СЊ", "Р° РµСЃР»Рё", "Р° РјРѕР¶РЅРѕ", "РјРѕР¶РЅРѕ Р»Рё",
    )
    explicit_topic = (
        "С‚РµРЅСЊ С‡РµСЂРЅРѕР±С‹Р»СЏ", "Р·РѕРІ РїСЂРёРїСЏС‚Рё", "С‡РёСЃС‚РѕРµ РЅРµР±Рѕ", "РіР»СѓС…Р°СЂСЊ", "С‚СЂРµРјРѕСЂ",
        "РєР°СЂРґР°РЅ", "Р°Р·РѕС‚", "СЃРѕРєРѕР»РѕРІ", "С‚РѕРїРѕР»СЊ", "СЃС‚СЂРµР»РѕРє", "РєСЂСѓРіР»РѕРІ", "РІРѕР»Рє",
        "РєРѕСЂРґРѕРЅ", "Р·Р°С‚РѕРЅ", "СЋРїРёС‚РµСЂ", "РїСЂРёРїСЏС‚СЊ", "Р°РіСЂРѕРїСЂРѕРј", "С…-8", "x-8",
    )
    return any(word in q for word in followup_words) and not any(topic in q for topic in explicit_topic)


def with_conversation_context(ip: str, question: str) -> str:
    previous = conversation_context_by_ip.get(ip, "")
    if previous and looks_like_followup(question):
        return f"{previous}\n\nРЈС‚РѕС‡РЅРµРЅРёРµ РёРіСЂРѕРєР°: {question}"
    return question


def remember_conversation_context(ip: str, question: str, answer: str) -> None:
    compact_answer = re.sub(r"\s+", " ", answer or "").strip()
    compact_question = re.sub(r"\s+", " ", question or "").strip()
    conversation_context_by_ip[ip] = (
        f"РџСЂРµРґС‹РґСѓС‰РёР№ РІРѕРїСЂРѕСЃ РёРіСЂРѕРєР°: {compact_question}\n"
        f"РџСЂРµРґС‹РґСѓС‰РёР№ РѕС‚РІРµС‚ Р®СЂС‹: {compact_answer[:900]}"
    )
    if len(conversation_context_by_ip) > 300:
        for key in list(conversation_context_by_ip)[:80]:
            conversation_context_by_ip.pop(key, None)


def ask_openai(question: str) -> str:
    full_context = full_sources.find_context(question, str(ROOT))
    if full_context:
        return local_story_answer_from_context(question, full_context)
    if ANTHOLOGY_CLOUD_AI_URL:
        cloud_answer = ask_cloud_yura(question)
        if cloud_answer:
            return cloud_answer

    api_key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not api_key:
        return local_fallback_answer(question)

    system_prompt = (
        "Ты помощник игроков A.N.T.H.O.L.O.G.Y / S.T.A.L.K.E.R. Anthology. "
        "Отвечай кратко, дружелюбно и только по теме Anthology. "
        "Если игрок пишет по-английски, отвечай по-английски. Если игрок пишет по-русски, отвечай по-русски. "
        "Сначала давай конкретное решение, потом пояснение. "
        "Если информации нет в базе, честно скажи, что нужно спросить в Discord Anthology. "
        "Не выдумывай ссылки, версии и инструкции.\n\n"
        "База знаний:\n"
        f"{KNOWLEDGE}"
    )
    payload = {
        "model": MODEL,
        "input": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question},
        ],
        "max_output_tokens": 260,
    }
    request = urllib.request.Request(
        "https://api.openai.com/v1/responses",
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=45) as response:
            data = json.loads(response.read().decode("utf-8", errors="replace"))
    except urllib.error.HTTPError as exc:
        details = exc.read().decode("utf-8", errors="replace")[:300]
        return f"AI helper: OpenAI API вернул ошибку {exc.code}. {details}"
    except Exception as exc:
        return f"AI helper: не удалось получить ответ ({type(exc).__name__})."

    if isinstance(data.get("output_text"), str):
        return trim_answer(data["output_text"])

    # Fallback for Responses API structured output.
    chunks: list[str] = []
    for item in data.get("output", []) or []:
        for content in item.get("content", []) or []:
            if content.get("type") in {"output_text", "text"} and isinstance(content.get("text"), str):
                chunks.append(content["text"])
    return trim_answer("\n".join(chunks) or "AI helper: пустой ответ.")


def ask_cloud_yura(question: str) -> str | None:
    try:
        headers = {
            "Content-Type": "text/plain; charset=utf-8",
        }
        if ANTHOLOGY_CLOUD_AI_TOKEN:
            headers["X-Anthology-Bridge-Token"] = ANTHOLOGY_CLOUD_AI_TOKEN
        request = urllib.request.Request(
            ANTHOLOGY_CLOUD_AI_URL,
            data=question.encode("utf-8"),
            headers=headers,
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=25) as response:
            return trim_answer(response.read().decode("utf-8", errors="replace"))
    except Exception:
        return None


class Handler(BaseHTTPRequestHandler):
    server_version = "AnthologyAIHelper/0.1"

    def do_GET(self) -> None:
        if self.path == "/health":
            self.send_text("ok")
            return
        self.send_text("Anthology AI Helper is running. POST /ask", status=404)

    def do_POST(self) -> None:
        if self.path != "/ask":
            self.send_text("Not found", status=404)
            return

        ip = self.client_address[0]
        now = time.time()
        wait = RATE_SECONDS - (now - last_request_by_ip.get(ip, 0))
        if wait > 0:
            self.send_text(f"Подожди {int(wait) + 1} сек. перед следующим вопросом.", status=429)
            return
        last_request_by_ip[ip] = now

        length = min(int(self.headers.get("Content-Length", "0") or "0"), 4096)
        raw = self.rfile.read(length)
        question = raw.decode("utf-8", errors="replace").strip()
        if not question:
            self.send_text("Напиши вопрос после /ai.", status=400)
            return
        if len(question) > MAX_QUESTION_CHARS:
            question = question[:MAX_QUESTION_CHARS]

        effective_question = with_conversation_context(ip, question)
        answer = ask_openai(effective_question)
        remember_conversation_context(ip, question, answer)
        self.send_text(answer)

    def send_text(self, text: str, status: int = 200) -> None:
        body = trim_answer(text).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "text/plain; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, fmt: str, *args) -> None:
        print("[%s] %s" % (self.address_string(), fmt % args))


def main() -> None:
    print(f"Anthology AI Helper listening on http://{HOST}:{PORT}")
    print(f"Model: {MODEL}")
    print("OpenAI key:", "set" if os.environ.get("OPENAI_API_KEY") else "not set (test mode)")
    ThreadingHTTPServer((HOST, PORT), Handler).serve_forever()


if __name__ == "__main__":
    main()
