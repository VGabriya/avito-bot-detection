"""Подготовка событий и построение поведенческих признаков на уровне cookie_id.

Все признаки считаются только по событиям внутри суточного окна
`window_start_ts <= event_ts < window_end_ts` каждой куки, то есть доступны
на момент окончания окна. Никакой информации из других кук (глобальных
популярностей, target encoding и т.п.) не используется, поэтому между
train и test нет утечек.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

DATE_COLS = ["cookie_created_at", "window_start_ts", "window_end_ts"]
EVENT_COLS = [
    "cookie_id", "event_ts", "eid", "event_name", "platform", "user_agent",
    "item_id", "item_category", "item_location", "seller_type",
    "search_query", "search_page", "pointer_x", "pointer_y",
]
EVENT_TYPES = [
    "search_results_view", "item_view", "photo_swipe", "seller_page_view",
    "contact_phone_show", "contact_chat_open", "contact_message_sent",
    "favorite_add", "login",
]
# Разрыв между событиями больше этого значения начинает новую сессию.
SESSION_GAP_SEC = 30 * 60

# Одна и та же платформа записана в разном регистре/синонимами — это
# технический шум логирования, а не поведение. Приводим к трём значениям.
PLATFORM_MAP = {
    "web": "web", "desktop": "web",
    "android": "android",
    "ios": "ios", "iphone": "ios",
}


# --------------------------------------------------------------------------
# Загрузка и очистка
# --------------------------------------------------------------------------
def load_data(data_dir: str = "data"):
    train = pd.read_csv(f"{data_dir}/train.csv", parse_dates=DATE_COLS)
    test = pd.read_csv(f"{data_dir}/test.csv", parse_dates=DATE_COLS)
    events = pd.read_csv(f"{data_dir}/events.csv.gz", parse_dates=["event_ts"])
    return train, test, events


def clean_events(events: pd.DataFrame, meta: pd.DataFrame) -> pd.DataFrame:
    """Дедупликация, фильтр по окну, сортировка и нормализация полей.

    * Полные дубликаты строк (~1.5% событий, одинаково у обоих классов) —
      артефакт логирования; удаляем, иначе они искажают счётчики и дают
      нулевые интервалы между событиями.
    * События после `window_end_ts` (~12% файла, включая ВСЕ `captcha_shown`)
      отбрасываем: это будущее относительно момента скоринга. Капча вообще
      показывается как реакция антибота, т.е. является прямой утечкой таргета.
    * Файл не отсортирован по времени — сортируем внутри куки.
    """
    ev = events.drop_duplicates(subset=EVENT_COLS)
    ev = ev.merge(meta[["cookie_id", "window_start_ts", "window_end_ts"]], on="cookie_id")
    ev = ev[(ev.event_ts >= ev.window_start_ts) & (ev.event_ts < ev.window_end_ts)]
    ev = ev.sort_values(["cookie_id", "event_ts", "eid"], kind="mergesort").reset_index(drop=True)

    ev["platform_norm"] = ev.platform.str.lower().map(PLATFORM_MAP).fillna("other")
    ev["dt"] = ev.groupby("cookie_id").event_ts.diff().dt.total_seconds()
    ev["hour"] = ev.event_ts.dt.hour
    return ev


# --------------------------------------------------------------------------
# Вспомогательные функции
# --------------------------------------------------------------------------
def _entropy(counts: pd.Series) -> float:
    p = counts.to_numpy(dtype=float)
    p = p[p > 0] / p.sum()
    return float(-(p * np.log(p)).sum()) if len(p) else 0.0


def _group_entropy(ev: pd.DataFrame, col: str) -> pd.Series:
    cnt = ev.dropna(subset=[col]).groupby(["cookie_id", col]).size()
    return cnt.groupby(level=0).apply(_entropy)


def parse_user_agent(ua: pd.Series) -> pd.DataFrame:
    """Разбираем UA на смысловые части вместо использования сырой строки.

    Сырая строка — высококардинальная категория, на которой модель просто
    запомнит конкретные версии из train. Полезны обобщаемые сигналы:
    headless-браузер, семейство браузера, ОС, «возраст» версии Chrome.
    """
    ua = ua.fillna("")
    out = pd.DataFrame(index=ua.index)
    out["ua_headless"] = ua.str.contains(
        r"headless|phantom|selenium|puppeteer|playwright|python|curl|wget|bot|spider",
        case=False, regex=True).astype(int)
    out["ua_linux_desktop"] = (ua.str.contains("X11; Linux") & ~ua.str.contains("Android")).astype(int)
    out["ua_mobile"] = ua.str.contains("Android|iPhone|iPad|Mobile", regex=True).astype(int)
    out["ua_yandex"] = ua.str.contains("YaBrowser").astype(int)
    out["ua_firefox"] = ua.str.contains("Firefox").astype(int)
    out["ua_safari_only"] = (ua.str.contains("Safari") & ~ua.str.contains("Chrome|CriOS|Android", regex=True)).astype(int)
    chrome_ver = ua.str.extract(r"(?:Chrome|CriOS|HeadlessChrome)/(\d+)")[0].astype(float)
    ff_ver = ua.str.extract(r"Firefox/(\d+)")[0].astype(float)
    out["ua_browser_major"] = chrome_ver.fillna(ff_ver)
    out["ua_len"] = ua.str.len()
    return out


# --------------------------------------------------------------------------
# Группы признаков
# --------------------------------------------------------------------------
def meta_features(meta: pd.DataFrame) -> pd.DataFrame:
    f = pd.DataFrame(index=meta.cookie_id.values)
    age = (meta.window_end_ts - meta.cookie_created_at).dt.total_seconds().to_numpy()
    # Возраст куки на момент конца окна. Флаг «кука создана внутри окна» не
    # делаем: в данных все куки созданы раньше начала окна, он константный.
    f["cookie_age_days"] = age / 86400
    return f


def count_features(ev: pd.DataFrame) -> pd.DataFrame:
    g = ev.groupby("cookie_id")
    f = pd.DataFrame({"n_events": g.size()})
    cnt = pd.crosstab(ev.cookie_id, ev.event_name).reindex(columns=EVENT_TYPES, fill_value=0)
    for c in EVENT_TYPES:
        f[f"cnt_{c}"] = cnt[c]
        f[f"share_{c}"] = cnt[c] / f.n_events
    f["event_type_entropy"] = _group_entropy(ev, "event_name")
    n_contacts = cnt[["contact_phone_show", "contact_chat_open", "contact_message_sent"]].sum(axis=1)
    views = cnt["item_view"].clip(lower=1)
    f["photo_per_view"] = cnt["photo_swipe"] / views
    f["contacts_per_view"] = n_contacts / views
    f["fav_per_view"] = cnt["favorite_add"] / views
    f["search_per_view"] = cnt["search_results_view"] / views
    f["has_login"] = (cnt["login"] > 0).astype(int)
    return f


def timing_features(ev: pd.DataFrame) -> pd.DataFrame:
    """Ритм событий: у скрипта интервалы короче и однороднее, чем у человека."""
    g = ev.groupby("cookie_id")
    dt = ev.dropna(subset=["dt"])
    gd = dt.groupby("cookie_id").dt
    f = pd.DataFrame(index=g.size().index)
    span = (g.event_ts.max() - g.event_ts.min()).dt.total_seconds()
    f["span_hours"] = span / 3600
    f["first_event_hour"] = g.event_ts.min().dt.hour + g.event_ts.min().dt.minute / 60

    f["dt_mean"] = gd.mean()
    f["dt_median"] = gd.median()
    f["dt_std"] = gd.std()
    f["dt_min"] = gd.min()
    f["dt_q10"] = gd.quantile(0.10)
    f["dt_q25"] = gd.quantile(0.25)
    f["dt_q75"] = gd.quantile(0.75)
    f["dt_log_std"] = dt.assign(l=np.log1p(dt.dt)).groupby("cookie_id").l.std()
    f["dt_cv"] = f.dt_std / f.dt_mean
    f["dt_share_0"] = (dt.dt == 0).groupby(dt.cookie_id).mean()
    f["dt_share_1_3"] = dt.dt.between(1, 3).groupby(dt.cookie_id).mean()
    f["dt_share_3_20"] = dt.dt.between(3, 20).groupby(dt.cookie_id).mean()
    f["dt_share_20_120"] = dt.dt.between(20, 120, inclusive="right").groupby(dt.cookie_id).mean()
    f["dt_share_4_15"] = dt.dt.between(4, 15).groupby(dt.cookie_id).mean()
    log_dt = np.log1p(dt.dt).groupby(dt.cookie_id)
    f["dt_log_skew"] = log_dt.skew()
    # повторяющиеся значения интервалов — признак таймера в скрипте
    f["dt_nunique_ratio"] = gd.nunique() / gd.size()

    # Сессии: внутри сессии ритм у бота особенно ровный
    sess = ev[["cookie_id", "event_ts", "dt"]].copy()
    sess["new_sess"] = sess.dt.isna() | (sess.dt > SESSION_GAP_SEC)
    sess["sess_id"] = sess.groupby("cookie_id").new_sess.cumsum()
    s = sess.groupby(["cookie_id", "sess_id"]).agg(
        n=("event_ts", "size"), start=("event_ts", "min"), end=("event_ts", "max"))
    s["dur"] = (s.end - s.start).dt.total_seconds()
    s["rate"] = s.n / (s.dur / 60).clip(lower=1)  # событий в минуту
    gs = s.groupby(level=0)
    f["n_sessions"] = gs.size()
    f["events_per_session"] = gs.n.mean()
    f["max_session_events"] = gs.n.max()
    f["max_session_dur_min"] = gs.dur.max() / 60
    f["max_session_rate"] = gs.rate.max()
    f["mean_session_rate"] = gs.rate.mean()
    in_sess = sess[~sess.new_sess]
    gi = in_sess.groupby("cookie_id").dt
    f["insess_dt_mean"] = gi.mean()
    f["insess_dt_median"] = gi.median()
    f["insess_dt_cv"] = gi.std() / gi.mean()
    f["insess_dt_max"] = gi.max()

    # Суточный профиль: ночная активность, разброс по часам
    f["night_share"] = ev.hour.between(0, 5).groupby(ev.cookie_id).mean()
    f["n_active_hours"] = g.hour.nunique()
    f["hour_entropy"] = _group_entropy(ev, "hour")
    f["events_per_active_hour"] = g.size() / f.n_active_hours
    # максимальная плотность: событий в самую загруженную минуту
    per_min = ev.groupby(["cookie_id", ev.event_ts.dt.floor("min")]).size()
    f["max_events_per_minute"] = per_min.groupby(level=0).max()
    return f


def content_features(ev: pd.DataFrame) -> pd.DataFrame:
    """Разнообразие и «логика» просмотренного контента.

    Сервисы сбора данных перебирают страницы выдачи и объявления, часто
    прыгают по номерам страниц и много раз повторяют одни и те же запросы.
    """
    g = ev.groupby("cookie_id")
    n = g.size()
    f = pd.DataFrame(index=n.index)
    for col in ["item_id", "item_category", "item_location", "search_query"]:
        f[f"nuniq_{col}"] = g[col].nunique()
    f["item_uniq_ratio"] = f.nuniq_item_id / g.item_id.count().clip(lower=1)
    f["category_entropy"] = _group_entropy(ev, "item_category")
    f["location_entropy"] = _group_entropy(ev, "item_location")
    f["top_category_share"] = (
        ev.dropna(subset=["item_category"]).groupby(["cookie_id", "item_category"]).size()
        .groupby(level=0).max() / g.item_category.count().clip(lower=1))
    f["share_pro_seller"] = (ev.seller_type == "pro").groupby(ev.cookie_id).sum() / g.seller_type.count().clip(lower=1)
    f["share_missing_category"] = g.item_category.apply(lambda s: s.isna().mean())
    # сколько городов/категорий на одно событие: «сводки по регионам» vs. поиск в своём городе
    f["locations_per_event"] = f.nuniq_item_location / n
    f["categories_per_event"] = f.nuniq_item_category / n
    # сбор контактов: показы телефона по разным объявлениям
    ph = ev[ev.event_name == "contact_phone_show"].groupby("cookie_id").item_id
    f["phone_items_nuniq"] = ph.nunique()
    f["phone_items_per_view"] = f.phone_items_nuniq / g.event_name.apply(lambda s: (s == "item_view").sum()).clip(lower=1)

    se = ev[ev.event_name == "search_results_view"].copy()
    gs = se.groupby("cookie_id")
    f["search_page_mean"] = gs.search_page.mean()
    f["search_page_max"] = gs.search_page.max()
    f["search_page_std"] = gs.search_page.std()
    f["search_page1_share"] = (se.search_page == 1).groupby(se.cookie_id).mean()
    f["search_page_ge5_share"] = (se.search_page >= 5).groupby(se.cookie_id).mean()
    f["searches_per_query"] = gs.size() / gs.search_query.nunique().clip(lower=1)
    # Человек листает выдачу последовательно (1→2→3). Для каждого запроса
    # смотрим, насколько номер страницы «прыгает» относительно предыдущего.
    se["page_step"] = se.groupby(["cookie_id", "search_query"]).search_page.diff()
    f["page_seq_share"] = (se.page_step == 1).groupby(se.cookie_id).sum() / gs.size()
    f["page_jump_share"] = (se.page_step.abs() > 1).groupby(se.cookie_id).sum() / gs.size()
    f["page_back_share"] = (se.page_step < 0).groupby(se.cookie_id).sum() / gs.size()
    # переключения запроса между соседними поисками
    se["q_switch"] = se.search_query.ne(se.groupby("cookie_id").search_query.shift())
    f["query_switch_share"] = se.q_switch.groupby(se.cookie_id).mean()

    # Действия с объявлением (фото, контакты, избранное) по объявлению,
    # которое кука до этого не открывала — нелогичная навигация.
    first_view = (ev[ev.event_name == "item_view"].groupby(["cookie_id", "item_id"]).event_ts.min()
                  .rename("first_view"))
    act = ev[ev.event_name.isin(["photo_swipe", "seller_page_view", "contact_phone_show",
                                 "contact_chat_open", "contact_message_sent", "favorite_add"])]
    act = act.join(first_view, on=["cookie_id", "item_id"])
    f["action_on_unviewed_share"] = (~act.first_view.le(act.event_ts)).groupby(act.cookie_id).mean()
    # сообщение без открытия чата
    chat = ev.event_name.eq("contact_chat_open").groupby(ev.cookie_id).sum()
    msg = ev.event_name.eq("contact_message_sent").groupby(ev.cookie_id).sum()
    f["msg_without_chat"] = ((msg > 0) & (chat == 0)).astype(int)
    return f


def pointer_features(ev: pd.DataFrame) -> pd.DataFrame:
    """Координаты курсора пишутся только на web. Скрипты часто не шлют их
    вовсе или ставят «невозможные» значения (x=1920 / y=1080 на экране 1920×1080)."""
    web = ev[ev.platform_norm == "web"]
    g = web.groupby("cookie_id")
    f = pd.DataFrame(index=ev.cookie_id.unique())
    f["pointer_present_share"] = g.pointer_x.apply(lambda s: s.notna().mean())
    f["pointer_x_std"] = g.pointer_x.std()
    f["pointer_y_std"] = g.pointer_y.std()
    f["pointer_x_mean"] = g.pointer_x.mean()
    f["pointer_y_mean"] = g.pointer_y.mean()
    edge = (web.pointer_x.isin([0, 1919, 1920]) | web.pointer_y.isin([0, 1079, 1080]))
    f["pointer_edge_share"] = edge.groupby(web.cookie_id).sum() / g.pointer_x.count().clip(lower=1)
    oob = (web.pointer_x >= 1920) | (web.pointer_y >= 1080)
    f["pointer_oob_cnt"] = oob.groupby(web.cookie_id).sum()
    p = web.dropna(subset=["pointer_x"])
    f["pointer_repeat_share"] = (p.duplicated(["cookie_id", "pointer_x", "pointer_y"])
                                 .groupby(p.cookie_id).mean())
    # У человека курсор между событиями перемещается по всему экрану,
    # у скрипта — небольшими шагами внутри ограниченной области.
    step = np.hypot(p.groupby("cookie_id").pointer_x.diff(), p.groupby("cookie_id").pointer_y.diff())
    f["pointer_step_mean"] = step.groupby(p.cookie_id).mean()
    f["pointer_step_median"] = step.groupby(p.cookie_id).median()
    center = np.hypot(p.pointer_x - 960, p.pointer_y - 540)
    f["pointer_center_dist"] = center.groupby(p.cookie_id).mean()
    return f


def ua_features(ev: pd.DataFrame) -> pd.DataFrame:
    g = ev.groupby("cookie_id")
    f = pd.DataFrame({"n_user_agents": g.user_agent.nunique(),
                      "n_platforms": g.platform_norm.nunique()})
    # признаки по самому частому UA куки
    top_ua = g.user_agent.agg(lambda s: s.value_counts().index[0])
    f = f.join(parse_user_agent(top_ua))
    plat = g.platform_norm.agg(lambda s: s.value_counts().index[0])
    f["platform_web"] = (plat == "web").astype(int)
    f["platform_android"] = (plat == "android").astype(int)
    f["platform_ios"] = (plat == "ios").astype(int)
    # мобильный UA на web-платформе или десктопный на приложении
    f["ua_platform_mismatch"] = (f.ua_mobile != (plat != "web").astype(int)).astype(int)
    return f


def build_features(meta: pd.DataFrame, ev: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, list[str]]]:
    """Итоговая матрица признаков (одна строка на cookie_id в порядке `meta`)
    и словарь {группа признаков: список колонок} для абляции."""
    ev = ev[ev.cookie_id.isin(meta.cookie_id)]
    parts = {
        "meta": meta_features(meta),
        "counts": count_features(ev),
        "timing": timing_features(ev),
        "content": content_features(ev),
        "pointer": pointer_features(ev),
        "user_agent": ua_features(ev),
    }
    X = pd.concat(parts.values(), axis=1).reindex(meta.cookie_id.values)
    # у куки без событий в окне счётчики = 0; остальные пропуски (нет интервалов,
    # нет web-событий, нет поисков) оставляем NaN — бустинг обрабатывает их сам
    cnt_cols = [c for c in X.columns if c.startswith("cnt_")] + ["n_events"]
    X[cnt_cols] = X[cnt_cols].fillna(0)
    X.index.name = "cookie_id"
    groups = {name: list(df.columns) for name, df in parts.items()}
    return X, groups
