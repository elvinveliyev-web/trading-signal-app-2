import os
import re
import json
import time
import base64
import datetime
import html
import xml.etree.ElementTree as ET
from io import BytesIO
from typing import Optional, Dict, Any, List, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
import yfinance as yf
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import requests
from sklearn.linear_model import Ridge, LinearRegression, RANSACRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.cluster import KMeans
from sklearn.model_selection import TimeSeriesSplit
from scipy.stats import linregress

# =============================
# OPTIONAL PDF SUPPORT (ReportLab)
# =============================
REPORTLAB_OK = True
try:
    from reportlab.pdfgen import canvas
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.units import cm
    from reportlab.lib.utils import ImageReader
except Exception:
    REPORTLAB_OK = False

st.set_page_config(page_title="FA→TA Trading + AI", layout="wide")

# =============================
# BASE DIR
# =============================
BASE_DIR = os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else os.getcwd()


def pjoin(*parts) -> str:
    return os.path.join(BASE_DIR, *parts)


# =============================
# Universe Loader
# =============================
@st.cache_data(ttl=24 * 3600, show_spinner=False)
def load_universe_file(path: str) -> List[str]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            raw = f.read()
        toks = re.split(r"[\s,;]+", raw.strip())
        tickers = [t.strip().upper() for t in toks if t.strip()]
        tickers = list(dict.fromkeys(tickers))
        return sorted(tickers)
    except Exception:
        return []


# =============================
# Helpers
# =============================
def normalize_ticker(raw: str, market: str) -> str:
    t = (raw or "").strip().upper()
    if not t:
        return t
    if market == "BIST" and not t.endswith(".IS"):
        t = f"{t}.IS"
    return t


def naked_ticker(raw: str) -> str:
    return (raw or "").strip().upper().replace(".IS", "")


def safe_float(x):
    try:
        if x is None:
            return np.nan
        if isinstance(x, (int, float, np.number)):
            return float(x)
        return float(str(x).replace(",", ""))
    except Exception:
        return np.nan


def _flatten_yf(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()

    out = df.copy()

    try:
        if isinstance(out.columns, pd.MultiIndex):
            wanted = {"Open", "High", "Low", "Close", "Adj Close", "Volume"}

            best_level = None
            best_score = -1
            for lvl in range(out.columns.nlevels):
                vals = [str(x) for x in out.columns.get_level_values(lvl)]
                score = len(set(vals).intersection(wanted))
                if score > best_score:
                    best_score = score
                    best_level = lvl

            if best_level is not None and best_score > 0:
                out.columns = [str(c[best_level]) for c in out.columns]
            else:
                out.columns = ["_".join([str(x) for x in c if str(x) != ""]) for c in out.columns]

        out.columns = [str(c).strip() for c in out.columns]

        # Duplicate OHLC columns can happen with yfinance when group_by/ticker format changes.
        if len(set(out.columns)) != len(out.columns):
            out = out.loc[:, ~pd.Index(out.columns).duplicated(keep="first")]

        rename_map = {}
        lower_map = {str(c).lower().replace(" ", ""): c for c in out.columns}
        canonical = {
            "open": "Open", "high": "High", "low": "Low", "close": "Close",
            "adjclose": "Adj Close", "volume": "Volume"
        }
        for key, val in canonical.items():
            if key in lower_map and lower_map[key] != val:
                rename_map[lower_map[key]] = val
        if rename_map:
            out = out.rename(columns=rename_map)

        for col in ["Open", "High", "Low", "Close", "Adj Close", "Volume"]:
            if col in out.columns:
                out[col] = pd.to_numeric(out[col], errors="coerce")

        required = [c for c in ["Open", "High", "Low", "Close"] if c in out.columns]
        if len(required) >= 4:
            out = out.dropna(subset=required)
        elif required:
            out = out.dropna(subset=required)

        return out
    except Exception as e:
        log_app_error("_flatten_yf", e, {"columns": str(getattr(df, "columns", ""))[:500]})
        return pd.DataFrame()


def fmt_pct(x: float) -> str:
    try:
        if x is None or (isinstance(x, float) and not np.isfinite(x)):
            return "N/A"
        return f"{x*100:.2f}%"
    except Exception:
        return "N/A"


def fmt_num(x: float, nd=2) -> str:
    try:
        if x is None or (isinstance(x, float) and not np.isfinite(x)):
            return "N/A"
        return f"{float(x):.{nd}f}"
    except Exception:
        return "N/A"




# =============================
# Runtime logging / diagnostics
# =============================
PRICE_CACHE_TTL_SECONDS = 120
FUNDAMENTAL_CACHE_TTL_SECONDS = 6 * 3600
SOCIAL_CACHE_TTL_SECONDS = 10 * 60
SR_CACHE_VERSION = "sr_v2_weighted"

def log_app_error(module: str, error: Any, context: Optional[Dict[str, Any]] = None, level: str = "ERROR") -> None:
    """
    Sessiz except: pass yerine uygulama içi ve dosya bazlı hafif loglama.
    UI akışını bozmaz; debug için st.session_state.app_errors ve logs/app_errors.log kullanır.
    """
    try:
        msg = str(error)
        ctx = context or {}
        item = {
            "ts": datetime.datetime.utcnow().isoformat(timespec="seconds") + "Z",
            "level": str(level),
            "module": str(module),
            "error": msg,
            "context": ctx,
        }
        try:
            if "app_errors" in st.session_state:
                compact = f"{item['module']}: {item['error']}"
                if compact not in st.session_state.app_errors[-20:]:
                    st.session_state.app_errors.append(compact)
        except Exception:
            pass

        try:
            log_dir = pjoin("logs")
            os.makedirs(log_dir, exist_ok=True)
            with open(os.path.join(log_dir, "app_errors.log"), "a", encoding="utf-8") as f:
                f.write(json.dumps(item, ensure_ascii=False, default=str) + "\n")
        except Exception:
            pass
    except Exception:
        pass


def update_app_config(**kwargs) -> Dict[str, Any]:
    """
    Session state kaosunu azaltmak için merkezi ve güvenli config snapshot.
    Mevcut akışı değiştirmez; sadece seçili ayarları tek yerde saklar.
    """
    try:
        cur = st.session_state.get("app_config", {})
        if not isinstance(cur, dict):
            cur = {}
        cur.update({k: v for k, v in kwargs.items() if k is not None})
        st.session_state.app_config = cur
        return cur
    except Exception as e:
        log_app_error("update_app_config", e, {"keys": list(kwargs.keys())})
        return kwargs

def _safe_positive_denominator(s: pd.Series, eps: float = 1e-9) -> pd.Series:
    out = pd.to_numeric(s, errors="coerce").astype(float)
    scale = float(np.nanmedian(np.abs(out.values))) if len(out) else np.nan
    dynamic_eps = max(eps, (scale * 1e-6) if np.isfinite(scale) and scale > 0 else eps)
    return out.mask(out.abs() <= dynamic_eps, np.nan)

APP_EDUCATION_TEXTS: Dict[str, str] = {
    "rsi": "RSI, fiyat hareketinin hızını ve gücünü 0-100 aralığında ölçen momentum göstergesidir. 70 üzeri veya 30 altı tek başına emir değildir; güçlü trendlerde uzun süre aşırı bölgede kalabilir. En iyi kullanım, trend filtresi, destek/direnç ve uyumsuzluklarla birlikte okumaktır.",
    "macd": "MACD, kısa ve uzun EMA farkından türetilen trend-momentum göstergesidir. Histogramın sıfır üstüne geçmesi ivme artışı, sıfır altına inmesi zayıflama anlamına gelebilir. Yatay piyasalarda sık yalancı sinyal üretebildiğinden tek başına değil, bağlam içinde kullanılmalıdır.",
    "atr_pct": "ATR%, oynaklığı fiyatın yüzdesi olarak gösterir. Yüksek ATR% daha geniş günlük salınım, daha büyük stop ve daha yüksek risk anlamına gelir. Pozisyon boyutu ve risk yönetiminde çok önemlidir.",
    "bollinger": "Bollinger Bantları fiyatın ortalama etrafındaki standart sapma zarfıdır. Fiyatın banda değmesi otomatik dönüş anlamına gelmez; trend güçlü ise fiyat bandın kenarında yürüyebilir. Band genişliği ve fiyat davranışı birlikte okunmalıdır.",
    "bb_width": "Bollinger Genişliği, sıkışma mı genişleme mi olduğunu gösterir. Düşük genişlik enerji birikimi, hızla yükselen genişlik ise hareketin açıldığını anlatır.",
    "stoch_rsi": "Stochastic RSI, RSI'ın kendi aralığındaki konumunu ölçerek klasik RSI'dan daha hızlı tepki verir. Bu hız avantaj olduğu kadar gürültü riskini de artırır.",
    "volume_ratio": "Hacim oranı, son hacmin kendi ortalamasına göre ne kadar güçlü olduğunu gösterir. 1'in üzeri normalin üstü ilgi, çok yüksek değerler ise kırılım, panik veya spekülatif hareket olasılığı anlamına gelebilir.",
    "obv": "OBV, fiyat yönünü hacimle çarparak kümülatif para akışı sezgisi verir. Fiyat yatayken OBV yükseliyorsa birikim, fiyat yükselirken OBV zayıfsa hareketin iç gücü sorgulanabilir.",
    "ema": "EMA, son barlara daha yüksek ağırlık veren hareketli ortalamadır. Kısa EMA'nın uzun EMA üzerinde olması çoğunlukla yukarı yapıya, altında olması aşağı yapıya işaret eder.",
    "adx": "ADX trendin yönünü değil gücünü ölçer. Yüksek ADX güçlü trend, düşük ADX yatay/kararsız piyasa anlamına gelebilir.",
    "stochastic": "Stokastik osilatör, fiyatın seçilen periyottaki yüksek-düşük aralığı içinde nereye kapandığını gösterir. Hızlı sinyal üretir ama trend piyasalarında çok erken karşı sinyal verebilir.",
    "force_index": "Force Index, fiyat değişimini hacimle birleştirir ve hareketin ne kadar güçle desteklendiğini gösterir.",
    "elder_ray": "Elder-Ray, EMA çevresindeki boğa ve ayı baskısını Bull Power ve Bear Power ile gösterir. Uyumsuzluk analizinde özellikle değerlidir.",
    "divergence": "Uyumsuzluk, fiyat yeni dip/tepe yaparken indikatörün aynı şeyi yapmaması durumudur. Hareketin yorulduğunu veya dönüş ihtimalini düşündürebilir; ancak tek başına yeterli sinyal değildir.",
    "vpvr": "VPVR, hacmi zaman ekseni yerine fiyat seviyelerine dağıtır. Böylece hangi seviyede ne kadar maliyet biriktiğini ve kurumsal ilginin nerede yoğunlaştığını gösterir.",
    "poc": "POC, hacim profilindeki en yüksek hacim düğümüdür; yani incelenen pencerede en çok işlemin geçtiği fiyat bölgesidir.",
    "poc_distance": "POC Uzaklık %, mevcut fiyatın ana hacim merkezinden ne kadar saptığını gösterir. Bu uzaklık hem güç hem kısa vadeli geri test riski anlamına gelebilir.",
    "support_resistance": "Destek/direnç, fiyatın daha önce tepki verdiği veya zorlandığı seviyelerdir. En iyi seviyeler çok dokunulan, hacim alan ve zaman içinde çalışan bölgelerdir.",
    "target_band": "Hedef fiyat bandı, ATR ve önemli seviyelerden türetilen olası hareket alanıdır. Bu band kesin tahmin değil, senaryo planlaması ve risk/ödül hesabı için yol haritasıdır.",
    "backtest": "Backtest, stratejinin geçmişte yaklaşık nasıl davranacağını gösterir. Komisyon ve slippage eklense bile gerçek piyasayı birebir kopyalamaz.",
    "monte_carlo": "Monte Carlo simülasyonu, getirilerin farklı sıralamalar altında nasıl davranabileceğini göstererek tek equity eğrisinin ötesine geçer.",
    "sharpe": "Sharpe oranı, riske göre düzeltilmiş getiriyi ölçer. Daha yüksek Sharpe daha verimli getiri anlamına gelebilir.",
    "sortino": "Sortino, sadece aşağı yönlü oynaklığı ceza olarak gördüğü için Sharpe'dan daha seçici olabilir.",
    "calmar": "Calmar oranı, yıllık getiriyi maksimum drawdown ile karşılaştırır.",
    "ulcer": "Ulcer Index, yalnızca oynaklığı değil, sermaye düşüşlerinin derinlik ve süresini de hissettirir.",
    "kelly": "Kelly yüzdesi, teorik optimal pozisyon büyüklüğünü tahmin eder. Pratikte tam Kelly çoğu zaman fazla agresiftir.",
    "beta": "Beta, varlığın veya stratejinin benchmark'a göre ne kadar hassas hareket ettiğini gösterir.",
    "information_ratio": "Information Ratio, benchmark'a göre ek getirinin ne kadar istikrarlı üretildiğini ölçer.",
    "future_price": "Future Price modülü, seçilen zaman dilimi için ileri bar kapanışı tahmin etmeye çalışan çoklu model katmanıdır. Bu bir kehanet değil, karar destek katmanıdır.",
    "mae": "MAE, modelin ortalama mutlak hata büyüklüğüdür. Daha düşük MAE daha tutarlı tahmin anlamına gelir.",
    "rmse": "RMSE, büyük hataları daha sert cezalandıran hata ölçüsüdür.",
    "mape": "MAPE, hatayı yüzde bazında ifade ederek farklı fiyat seviyelerindeki varlıkları kıyaslamayı kolaylaştırır.",
    "direction_acc": "Yön doğruluğu, modelin fiyatın tam seviyesinden çok yönünü doğru tahmin etme başarısını ölçer.",
    "confidence": "Güven skoru, modelin hata ve olasılık davranışlarından türetilen bileşik bir kalite puanıdır.",
    "train_test": "Eğitim/Test satır sayısı, modelin kaç örnekle öğrenip kaç örnek üzerinde sınandığını gösterir.",
    "trend_regime": "Trend rejimi, yapının daha çok yükseliş, düşüş veya nötr karakterde olup olmadığını özetler.",
    "vol_regime": "Volatilite rejimi, piyasanın sakin mi gergin mi olduğunu gösterir.",
    "triple_screen": "Triple Screen, büyük zaman diliminde trendi, orta zaman diliminde düzeltmeyi ve küçük zaman diliminde tetikleyiciyi birlikte okur.",
    "roe": "ROE, şirketin özkaynağını ne kadar verimli kullandığını gösterir.",
    "revenue_growth": "Net satış büyümesi, şirketin üst satırının ne kadar hızlı genişlediğini gösterir.",
    "ebitda": "FAVÖK/EBITDA, şirketin esas faaliyet üretim gücünü görmeye yarar.",
    "ebitda_margin": "FAVÖK marjı, satışların ne kadarının operasyonel kazanca döndüğünü gösterir.",
    "debt_equity": "Borç/Özsermaye oranı, büyümenin ne kadarının kaldıraçla finanse edildiğini gösterir.",
    "current_ratio": "Cari oran, kısa vadeli yükümlülüklerin dönen varlıklarla karşılanabilme gücünü ölçer.",
    "net_margin": "Net kâr marjı, tüm giderler sonrası satışların ne kadarının kâra dönüştüğünü gösterir.",
    "fcf": "Serbest nakit akışı, şirketin yatırım harcamaları sonrası kasada bıraktığı gerçek nakit üretimidir.",
    "pe": "F/K oranı, piyasanın şirket kârını kaç katla fiyatladığını gösterir.",
    "pb": "PD/DD, piyasa değerinin defter değerine oranıdır.",
    "net_debt_ebitda": "Net Borç / FAVÖK, şirketin borcunu faaliyet gücüyle ne kadar rahat taşıdığını gösterir.",
    "altman_z": "Altman Z-Skoru, bilanço ve gelir tablosundan türetilen iflas/stres risk göstergesidir.",
    "piotroski_f": "Piotroski F-Skoru, kârlılık, kaldıraç ve verimlilikten oluşan kalite puanıdır.",
    "dcf": "DCF, gelecekteki nakit akımlarını bugüne indirerek teorik içsel değer tahmini yapar.",
    "sector_relative": "Sektöre göre pahalı/ucuz göstergesi, şirket çarpanlarının sektör ortalamasına göre primli mi iskontolu mu olduğunu gösterir.",
    "trend_patt": "Trend with Patt Entry yaklaşımı, ana yönü trend filtresiyle belirleyip giriş zamanlamasını uygun pattern ile yapmaya çalışır.",
    "donchian": "Donchian Kanalları, belirli periyottaki en yüksek ve en düşük seviyeleri kanal olarak gösterir.",
    "donchian_520": "Donchian 5&20 yaklaşımı, kısa ve uzun kanal bilgisini birlikte okuyarak erken tetik ile daha sağlam trend teyidini birleştirir.",
    "richard_dennis": "Richard Dennis / Turtle mantığı, sistematik trend takibi ve kırılım temelli girişlere dayanır."
}

def _ensure_help_badge_css():
    if st.session_state.get("_edu_help_css_loaded"):
        return
    st.session_state["_edu_help_css_loaded"] = True
    st.markdown("""
    <style>
    .edu-help-wrap {
        display:flex;
        flex-wrap:wrap;
        gap:8px;
        margin:6px 0 12px 0;
        align-items:flex-start;
    }
    .edu-help-badge {
        border:1px solid rgba(120,120,120,.35);
        border-radius:999px;
        background:rgba(125,125,125,.08);
        overflow:visible;
    }
    .edu-help-summary {
        list-style:none;
        cursor:pointer;
        display:inline-flex;
        align-items:center;
        gap:6px;
        padding:4px 10px;
        font-size:12px;
        line-height:1.2;
        user-select:none;
    }
    .edu-help-summary::-webkit-details-marker {display:none;}
    .edu-help-q {
        width:18px;
        height:18px;
        border-radius:50%;
        display:inline-flex;
        align-items:center;
        justify-content:center;
        font-weight:700;
        font-size:11px;
        border:1px solid rgba(120,120,120,.45);
        background:rgba(255,255,255,.08);
        flex:0 0 auto;
    }
    .edu-help-tip {
        width:min(820px, 88vw);
        white-space:normal;
        padding:12px 14px;
        margin:0 8px 10px 8px;
        border-radius:12px;
        background:#111827;
        color:#f9fafb;
        border:1px solid rgba(255,255,255,.12);
        box-shadow:0 8px 24px rgba(0,0,0,.25);
        font-size:12px;
        line-height:1.55;
    }
    </style>
    """, unsafe_allow_html=True)

def render_help_badges(items: List[Any], title: str = ""):
    _ensure_help_badge_css()
    badges = []
    for item in items:
        if isinstance(item, (tuple, list)) and len(item) >= 2:
            key = str(item[0]); label = str(item[1])
        else:
            key = str(item); label = str(item)
        tip = APP_EDUCATION_TEXTS.get(key, "")
        if not tip:
            continue
        safe_label = html.escape(label)
        safe_tip = html.escape(tip).replace("\n", "<br><br>")
        badges.append(
            '<details class="edu-help-badge">'
            f'<summary class="edu-help-summary">{safe_label}<span class="edu-help-q">?</span></summary>'
            f'<div class="edu-help-tip">{safe_tip}</div>'
            '</details>'
        )
    if badges:
        st.markdown('<div class="edu-help-wrap">' + "".join(badges) + '</div>', unsafe_allow_html=True)



def render_page_education_expander(items: List[Any], label: str = "↘ Bu sayfanın eğitim rehberini aç"):
    with st.expander(label, expanded=False):
        for item in items:
            if isinstance(item, (tuple, list)) and len(item) >= 2:
                key = str(item[0]); title = str(item[1])
            else:
                key = str(item); title = str(item)
            tip = APP_EDUCATION_TEXTS.get(key, "")
            if not tip:
                continue
            st.markdown(f"### {title}")
            st.markdown(tip)
            st.markdown("---")


APP_EDUCATION_TEXTS.update({
    "ema": """EMA (Exponential Moving Average), son barlara daha fazla ağırlık veren hareketli ortalamadır. Bu yüzden fiyatın güncel ritmine klasik ortalamalardan daha hızlı uyum sağlar. Kısa EMA yukarı dönüyor ve uzun EMA'nın üzerindeyse trend ivmesi çoğunlukla boğa lehine kabul edilir; kısa EMA aşağı dönüyor ve uzun EMA'nın altındaysa ayı lehine baskı artar. Ancak EMA'lar gecikmeli çalışır; ani haber akışlarında veya yatay piyasalarda fiyat çizgiyi sık sık kesip tekrar üzerine çıkabilir. Bu nedenle EMA'yı tek başına sinyal değil, trend filtresi ve dinamik destek/direnç alanı olarak okumak en sağlıklı yaklaşımdır.""",
    "sma_bias": """SMA bias, hızlı SMA ile yavaş SMA'nın ilişkisinden türetilen yapısal yön özetidir. Hızlı SMA yavaş SMA'nın üzerindeyse piyasa kısa-orta vadede LONG bias taşır; altındaysa SHORT bias öne çıkar. Bu bilgi, her barda al-sat komutu vermekten çok, mevcut rüzgârın hangi yönden estiğini anlamak için kullanılır. Yatay piyasalarda SMA'lar birbirini sık kesebilir ve bias kısa sürede değişebilir; bu yüzden ADX, hacim ve fiyatın ana destek/direnç bölgelerine göre konumu ile birlikte değerlendirilmelidir.""",
    "rsi": """RSI, fiyat değişimlerinin hızını ve kalıcılığını 0 ile 100 arasında ölçen momentum osilatörüdür. Klasik eşikler 70 üzeri aşırı alım ve 30 altı aşırı satım olarak anlatılır; fakat bu seviyeler tek başına dönüş garantisi vermez. Güçlü boğa trendlerinde RSI 70 üzerinde uzun süre kalabilir, güçlü ayı trendlerinde ise 30 altı bölge uzayabilir. Bu nedenle RSI'ı en sağlıklı okumak için şu üç soruya bakılır: trend yönü nedir, fiyat önemli destek/direnç bölgesinde mi, indikatör fiyatla uyumsuzluk veriyor mu? Orta çizgi olan 50'nin üstü genellikle boğa momentumunu, altı ise ayı momentumunu yansıtır.""",
    "macd": """MACD, iki EMA arasındaki farktan oluşan trend-momentum göstergesidir. MACD çizgisi ile sinyal çizgisi arasındaki mesafe büyüdükçe momentum güçlenir; histogramın sıfır çizgisini geçmesi ise ivme rejimindeki değişimi gösterir. Sıfır üstünde büyüyen histogram çoğu zaman yükseliş ivmesinin arttığını, sıfır altında derinleşen histogram ise satış baskısının kuvvetlendiğini anlatır. Fakat yatay ve dalgalı piyasalarda histogram sık sık yön değiştirerek yalancı sinyal üretebilir. Bu yüzden MACD en iyi, ana trend filtresi ve fiyat yapısıyla birlikte kullanıldığında işe yarar.""",
    "atr_pct": """ATR%, ortalama gerçek aralığın fiyata oranlanmış halidir ve volatiliteyi yüzdesel olarak ölçer. Aynı ATR değeri 20 liralık bir hissede çok büyük, 500 liralık bir hissede küçük olabilir; bu yüzden ATR%'yi kullanmak farklı fiyat ölçeklerini karşılaştırmayı kolaylaştırır. Yüksek ATR% daha geniş dalga boyu, daha büyük stop mesafesi ve daha dikkatli pozisyon boyutu anlamına gelir. Düşük ATR% ise sakin fiyat yapısını gösterebilir ama bu her zaman fırsatsız piyasa demek değildir; bazen güçlü trendler de kontrollü, düşük ATR% ile ilerler. ATR%, sinyal üretmekten çok risk yönetimi, stop mesafesi ve rejim analizi için kullanılır.""",
    "bollinger": """Bollinger Bantları, hareketli ortalama etrafına standart sapma ekleyerek fiyatın istatistiksel zarfını çizer. Fiyat üst banda dokundu diye otomatik pahalı, alt banda indi diye otomatik ucuz kabul edilmez; güçlü trendlerde fiyat bandın kenarında yürüyebilir. Esas yorum, bandın genişliği, fiyatın orta banda göre davranışı ve hacimle birlikte yapılır. Orta bant çoğu zaman kısa vadeli denge çizgisidir; fiyat bunun üzerinde kalıyorsa yukarı yönlü yapı, altında kalıyorsa zayıflama eğilimi okunabilir. Bantların sıkışması enerji birikimini, genişlemesi ise hareketin açıldığını gösterir.""",
    "bb_width": """Bollinger Genişliği, üst ve alt bandın birbirinden ne kadar ayrıldığını ölçer. Çok düşük genişlik, piyasanın sıkıştığını ve enerjinin biriktiğini gösterebilir; bu dönemlerin ardından sert kırılımlar gelebilir. Çok hızlı genişleyen genişlik ise ya trend başlangıcını ya da panik hareketini anlatabilir. Yön belirtmediği için tek başına alım-satım kararı üretmez. En iyi kullanım, band sıkışması sonrası fiyatın hangi yöne kırıldığı, hacmin bu kırılımı destekleyip desteklemediği ve ana trendin ne söylediği ile birlikte değerlendirmektir.""",
    "stochastic": """Stokastik osilatör, fiyatın seçilen periyottaki yüksek-düşük aralığı içinde nereye kapandığını gösterir. Bu sayede kapanışların aralık içinde üst tarafta mı, alt tarafta mı yoğunlaştığını anlarsın. Hızlı tepki verdiği için dönüş arayan sistemlerde faydalıdır, ancak güçlü trendlerde erken karşı sinyal verme eğilimi yüksektir. 80 üzeri ve 20 altı bölgeler tek başına emir üretmez; asıl değer, kesişimlerin trend bağlamında ve destek/direnç bölgelerinde okunmasındadır.""",
    "stoch_rsi": """Stochastic RSI, RSI'ın kendi iç aralığı içindeki konumunu ölçerek klasik RSI'dan daha hassas ve hızlı hale gelir. Bu sayede kısa vadeli yön değişimlerine daha erken tepki verebilir, ancak aynı nedenle daha gürültülüdür. Özellikle kısa vadeli aşırılaşmaları ve mini dönüş bölgelerini görmekte kullanışlıdır. Trend yönü ile birlikte kullanılmadığında sık yalancı sinyal üretebilir; bu yüzden Stochastic RSI en iyi, EMA yapısı ve hacimle teyit edilerek kullanılır.""",
    "volume_ratio": """Hacim oranı, mevcut barın hacmini kendi ortalama hacmine böler ve hareketin ne kadar ilgi çektiğini gösterir. 1'in üzeri değerler son barın normalden daha canlı geçtiğini, belirgin yüksek değerler ise kırılım, haber etkisi, panik veya spekülatif ilgi ihtimalini işaret eder. Ancak yüksek hacim her zaman sağlıklı yükseliş değildir; dağıtım veya zorunlu çıkış da olabilir. Bu nedenle hacim oranını fiyatın yönü, mum yapısı ve destek/direnç bölgeleriyle birlikte okumak gerekir.""",
    "obv": """OBV (On Balance Volume), fiyat yönünü hacimle kümülatif bir şekilde birleştirerek para akışının genel yönü hakkında fikir verir. Fiyat yatay seyrederken OBV yükseliyorsa gizli birikim, fiyat yükselirken OBV yatay/negatif gidiyorsa hareketin iç gücünde sorun olabilir. Kırılım öncesi hacim davranışını görmek için çok faydalıdır. Ancak tüm hacmi aynı kalitede kabul ettiği için tek başına nihai sinyal motoru değil, doğrulama aracıdır.""",
    "force_index": """Force Index, fiyat değişimi ile hacmi çarparak hareketin ne kadar kuvvetli gerçekleştiğini gösterir. Sadece fiyatın yönünü değil, bu yönün ne kadar enerjiyle desteklendiğini anlamaya yarar. Sıfır çizgisi üzerindeki güçlü değerler alıcı baskısını, altındaki güçlü değerler satıcı baskısını anlatır. Force Index özellikle trend içindeki düzeltmelerin sona erip ana hareketin devam edip etmeyeceğini anlamada faydalıdır.""",
    "elder_ray": """Elder-Ray, fiyatı 13 EMA etrafında değerlendirerek Bull Power ve Bear Power üretir. Bull Power, boğaların EMA üzerine ne kadar çıkabildiğini; Bear Power ise ayıların EMA altına ne kadar inebildiğini gösterir. Fiyat yeni dip yaparken Bear Power daha az zayıflıyorsa pozitif uyumsuzluk, fiyat yeni tepe yaparken Bull Power eşlik etmiyorsa negatif uyumsuzluk okunabilir. Elder yaklaşımında bu göstergeler, trend filtresi ve giriş zamanlaması arasında köprü görevi görür.""",
    "adx": """ADX, trendin yönünü değil gücünü ölçer. +DI ve -DI hangi tarafın baskın olduğunu söylerken, ADX bu baskının ne kadar organize ve kuvvetli olduğunu gösterir. 20-25 altı değerler çoğu zaman yatay veya kararsız yapıya, daha yüksek değerler trendleşen piyasaya işaret eder. Bu nedenle RSI gibi osilatörler yatay dönemde daha iyi çalışırken, trend piyasasında ADX'in yükselmesi karşı sinyallerin daha tehlikeli olabileceğini anlatır.""",
    "divergence": """Uyumsuzluk, fiyatın yeni dip veya tepe yapmasına rağmen indikatörün bunu teyit etmemesi durumudur. Bu, mevcut hareketin yorulduğunu, iç momentumun zayıfladığını veya dönüş olasılığının arttığını düşündürebilir. Fakat uyumsuzluk tek başına giriş emri değildir; bazen fiyat uzun süre uyumsuzlukla devam eder. En güçlü kullanım, ana trend, önemli seviye, hacim ve teyit mumları ile birlikte okumaktır.""",
    "triple_screen": """Triple Screen yaklaşımı, piyasayı tek bir zaman diliminden değil, üç farklı perspektiften okumayı hedefler. Büyük zaman dilimi ana trendi, orta zaman dilimi düzeltmeyi, küçük zaman dilimi ise tetikleyiciyi verir. Böylece yatırımcı yanlış yönde agresif işlem açmak yerine, ana yönü arkasına almış fırsatları seçmeye çalışır. Bu sistemin gücü, göstergeleri üst üste koymasından değil, bağlam kurmasından gelir.""",
    "vpvr": """VPVR, hacmi zaman yerine fiyat seviyelerine dağıtarak hangi bölgelerde gerçek maliyet kümelenmesi oluştuğunu gösterir. Zaman bazlı destek/direnç çizgileri bazen seviyenin arkasındaki gerçek ilgiyi anlatamaz; VPVR bu boşluğu doldurur. Hacim düğümleri, piyasanın fiyatı hangi bölgede daha fazla kabul ettiğini; hacim boşlukları ise fiyatın hızla geçme eğiliminde olabileceği alanları gösterir. Özellikle kurumsal maliyetlenme ve denge alanlarını okumada çok değerlidir.""",
    "poc": """POC (Point of Control), seçilen hacim profilindeki en yüksek hacim düğümüdür. Yani incelenen pencerede en fazla işlemin geçtiği ve piyasanın en çok kabul ettiği fiyat seviyesini temsil eder. Fiyat POC'nin üzerinde kalıyorsa bu, kabul alanının üstüne taşmış bir yapı anlamına gelebilir; altında kalıyorsa piyasanın ana maliyet merkezinin altına sarkılmış olabilir. Ancak POC tek başına destek/direnç çizgisi değildir; çevresindeki hacim dağılımı ve trend yönü ile birlikte okunmalıdır.""",
    "poc_distance": """POC uzaklığı, mevcut fiyatın ana hacim merkezinden yüzde olarak ne kadar koptuğunu gösterir. Fiyat POC'nin üstündeyse çoğu zaman boğalar lehine kabul bölgesinden yukarı taşmış bir yapı vardır; fakat POC'den çok uzaklaşmak kısa vadeli geri test riskini de artırır. Çok küçük mesafe denge, çok büyük mesafe ise ya güçlü trend ya da aşırı uzama anlamına gelebilir. Bu nedenle POC uzaklığı tek başına iyi-kötü değil, 'güç ile geri dönüş riski arasındaki denge' olarak okunmalıdır.""",
    "support_resistance": """Destek ve direnç, fiyatın daha önce tepki verdiği, durduğu, reddedildiği veya kırılmakta zorlandığı bölgelerdir. İyi bir seviye tek dokunuşla değil; zaman içinde korunması, hacim çekmesi ve tekrar tekrar piyasayı etkilemesiyle güç kazanır. Teknik analizde seviyeler çoğu zaman çizgi değil bölgedir. Bu yüzden birkaç kuruş/sent taşma hemen seviye bozuldu demek değildir; asıl önemli olan fiyatın bölgeyi kabul edip etmemesidir.""",
    "target_band": """Hedef fiyat bandı, ATR ve önemli seviye kümelerini birlikte kullanarak olası boğa ve ayı senaryolarını aralık olarak sunar. Bu, 'kesin hedef fiyat' vermekten çok, pozisyon planlarken nerede nefes alınabileceğini, stop ve ödül oranının nasıl şekilleneceğini görmeye yarar. İyi yatırımcılar tek rakama değil, aralığa çalışır; çünkü piyasa doğrusal hareket etmez.""",
    "risk_reward": """Risk/ödül oranı, bir işlemde kaybetmeyi göze aldığın tutarla kazanmayı hedeflediğin tutar arasındaki oranı gösterir. Yüksek isabet oranı kadar güçlü bir risk/ödül yapısı da uzun vadede kritik öneme sahiptir. Düşük isabetli ama yüksek ödüllü trend takip sistemleri kârlı olabilirken, çok sık kazanan fakat düşük ödüllü sistemler birkaç büyük kayıpla bozulabilir.""",
    "backtest": """Backtest, stratejinin geçmiş veri üzerindeki yaklaşık davranışını ölçer. Amaç geleceği garanti etmek değil, sistemin hangi koşullarda güçlendiğini, hangi koşullarda kırılganlaştığını öğrenmektir. Komisyon, slippage ve stop kuralları eklemek backtest'i daha dürüst hale getirir; yine de gerçek piyasadaki likidite, haber akışı, gap ve psikoloji etkisini birebir yansıtmaz. Bu nedenle backtest sonucu bir 'kanıt' değil, karar desteği olarak kullanılmalıdır.""",
    "monte_carlo": """Monte Carlo simülasyonu, aynı stratejinin getiri dizisinin farklı sıralanışlarda nasıl davranabileceğini göstererek sonuçların ne kadar oynak olabileceğini ortaya koyar. Tek bir equity eğrisi bazen fazla güven verir; Monte Carlo ise iyi, orta ve kötü senaryo aralığını görmeni sağlar. Özellikle maksimum düşüş, beklenen sonuç dağılımı ve sermaye dayanıklılığı açısından çok öğreticidir.""",
    "sharpe": """Sharpe oranı, riskten arındırılmış getiriyi toplam volatiliteye göre ölçer. Aynı getiriyi üreten iki sistemden daha düşük oynak olan sistemin Sharpe'ı daha yüksek çıkar. Ancak Sharpe yukarı yönlü büyük hareketleri de volatilite saydığı için bazı trend takip sistemlerinde gereğinden sert olabilir. Bu yüzden başka risk metrikleriyle birlikte okunmalıdır.""",
    "sortino": """Sortino oranı, Sharpe'tan farklı olarak yalnızca aşağı yönlü oynaklığı ceza olarak görür. Bu da yatırımcı açısından daha psikolojik ve pratik bir risk yaklaşımı sunar. Özellikle yukarı yönlü sıçramalar yapan sistemlerde Sharpe düşük görünürken, Sortino daha anlamlı bir tablo verebilir.""",
    "calmar": """Calmar oranı, yıllıklaştırılmış getiriyi maksimum drawdown'a böler. Yani sistemin getirisini, yaşattığı en büyük acı ile karşılaştırır. Özellikle sermaye korumasını önemseyen yatırımcılar için çok değerlidir. Yüksek Calmar, genellikle hem güçlü getiri hem de kontrollü düşüş yapısı anlamına gelir.""",
    "ulcer": """Ulcer Index, sadece standart sapmayı değil, yatırımcının yaşadığı drawdown derinlik ve süresini de dikkate alır. Aynı getiriye sahip iki sistemden biri sermayeyi daha az yıpratıyorsa Ulcer Index bunu daha iyi yakalayabilir. Bu yönüyle yatırımcı psikolojisine daha yakın bir risk ölçüsüdür.""",
    "kelly": """Kelly kriteri, teorik olarak en verimli pozisyon büyüklüğünü bulmaya çalışır. Ancak tam Kelly çoğu zaman pratikte fazla agresiftir; veri hatasına ve model yanılgısına çok duyarlıdır. Bu yüzden birçok profesyonel yarım Kelly veya daha muhafazakâr bir oran tercih eder. Kelly'yi emir büyüklüğünü körlemesine belirleyen kural değil, üst sınır uyarısı gibi düşünmek daha güvenlidir.""",
    "beta": """Beta, varlığın veya stratejinin benchmark'a göre ne kadar hassas hareket ettiğini ölçer. 1'in üzerindeki beta, benchmark hareketlerinin büyütülerek yaşanabileceğini; 1'in altı daha sakin bir yapıyı anlatır. Negatif beta ise benchmark ile ters yönlü ilişki anlamına gelebilir. Beta riski anlatır ama kaliteyi anlatmaz; bu yüzden alfa ve bilgi oranıyla birlikte değerlidir.""",
    "information_ratio": """Information Ratio, benchmark'ı ne kadar geçtiğini değil, bunu ne kadar istikrarlı geçtiğini ölçer. Yani fazla getirinin kalite kontrolüdür. Benchmark'ı ara sıra çok geçen ama düzensiz performans gösteren stratejilerde düşük kalabilir. Düzenli ve sürdürülebilir ekstra getiri arayan yatırımcı için anlamlı bir metriktir.""",
    "slippage": """Slippage, planlanan fiyat ile fiili işlem fiyatı arasındaki farktır. Backtest'te küçük görünen bu kalem, özellikle hızlı hareketlerde, düşük likiditede ve stop çalışırken ciddi performans farkı yaratabilir. Strateji ne kadar sık işlem açıyorsa slippage o kadar kritik hale gelir.""",
    "commission": """Komisyon, görünen işlem maliyetidir ve özellikle yüksek frekanslı sistemlerde birikerek büyük aşındırma yaratır. Küçük oranlar bile çok sayıda işlemde stratejinin net performansını ciddi biçimde azaltabilir. Bu yüzden brüt değil, net sonuç düşünmek gerekir.""",
    "future_price": """Future Price modülü, makine öğrenmesi ile seçilen bar ufkunda ileriye dönük fiyat/göreli yön tahmini üretir. Bu çıktı, 'olacak fiyat' kehaneti değil; belirli veri, özellik seti ve model varsayımları altında üretilmiş olasılıklı bir projeksiyondur. En sağlıklı kullanım, bunu trend, seviye, hacim ve temel yapı ile birlikte değerlendirmektir.""",
    "mae": """MAE, tahminlerin ortalama mutlak sapmasını gösterir. Hataları sade ve anlaşılır biçimde ölçtüğü için pratikte çok kullanışlıdır. Ancak fiyatı yüksek hisselerde doğal olarak daha büyük sayı görünür; bu yüzden yüzde bazlı metriklerle birlikte okunmalıdır.""",
    "rmse": """RMSE, büyük hataları daha sert cezalandıran hata ölçüsüdür. Model çoğu zaman iyi ama bazen çok kötü tahmin yapıyorsa RMSE bunu hemen ortaya çıkarır. Bu yönüyle 'kötü gün performansı' için değerli bir metriktir.""",
    "mape": """MAPE, model hatasını yüzde olarak ölçer ve farklı fiyat ölçeklerindeki sembolleri kıyaslamayı kolaylaştırır. Çok düşük fiyatlı hisselerde baz etkisi nedeniyle yanıltıcı olabilir. Bu yüzden MAE ve RMSE ile birlikte okunmalıdır.""",
    "direction_acc": """Yön doğruluğu, modelin tam fiyatı değil, fiyatın yönünü doğru tahmin etme başarısını ölçer. Trading açısından bazen bu bilgi, fiyat tahmin hatasından daha değerli olabilir. Fakat yüksek yön doğruluğu bile kötü risk/ödül veya geniş hata bandı ile birleşirse tek başına yeterli olmaz.""",
    "confidence": """Güven skoru, hata metrikleri, yön başarısı ve bant genişliği gibi sinyallerden türetilen bileşik bir rahatlık göstergesidir. Kesin doğruluk garantisi vermez. En iyi kullanım, hangi sembol ve ufuk kombinasyonunda model sonucuna daha temkinli yaklaşmak gerektiğini anlamaktır.""",
    "train_test": """Eğitim/Test satırı, modelin kaç gözlemle öğrenip kaç gözlem üzerinde sınandığını gösterir. Çok küçük test setleri, sonucu olduğundan iyi veya kötü gösterebilir. Zaman serisi yapısında veri bölme disiplini çok önemlidir; geleceğin geçmişe sızmaması gerekir.""",
    "trend_regime": """Trend rejimi, mevcut piyasanın yukarı, aşağı veya nötr karakterini özetler. Aynı model farklı rejimlerde farklı kalite üretebilir. Bu nedenle model sonucunu mutlak değil, rejime duyarlı okumak daha güvenli olur.""",
    "vol_regime": """Volatilite rejimi, piyasanın sakin mi gergin mi olduğunu anlatır. Yüksek volatilitede hem fırsat hem hata bandı büyür; düşük volatilitede ise hareketler daha kontrollü ama sınırlı olabilir. Tahmin ve stop yönetimi mutlaka bu bağlam içinde yorumlanmalıdır.""",
    "roe": """ROE, özkaynağın ne kadar verimli kâra dönüştürüldüğünü gösterir. Yüksek ROE cazip görünür; ancak aşırı borç kullanımı bu oranı yapay olarak yükseltebilir. Bu yüzden borçluluk ve nakit üretimi ile birlikte okunmalıdır. Sürdürülebilir yüksek ROE, çoğu zaman kaliteli iş modeli ve verimli sermaye kullanımı işaretidir.""",
    "revenue_growth": """Net satış büyümesi, şirketin üst satırının ne kadar hızlandığını gösterir. Hızlı büyüme heyecan vericidir ama kârlılığı bozuyorsa kalitesi düşebilir. Bu nedenle marjlar, nakit akışı ve borç dinamiği ile birlikte incelenmelidir.""",
    "ebitda": """FAVÖK (EBITDA), şirketin faaliyet üretim gücünü faiz, vergi ve amortisman etkilerinden arındırılmış olarak gösterir. Özellikle sermaye yoğun sektörlerde operasyonel performansı daha temiz görmeye yardımcı olur. Ancak EBITDA nakdin aynısı değildir; yatırım harcamaları ve işletme sermayesi etkileri ayrıca izlenmelidir.""",
    "ebitda_margin": """FAVÖK marjı, satışların ne kadarının operasyonel kârlılığa döndüğünü gösterir. Marjın yükselmesi verimlilik, fiyatlama gücü veya daha kaliteli ürün karması anlamına gelebilir. Tek başına değil, sektör ortalaması ve zaman içindeki eğilim ile birlikte okunmalıdır.""",
    "debt_equity": """Borç/Özsermaye oranı, büyümenin ve faaliyetlerin ne kadarının kaldıraçla taşındığını gösterir. Aşırı yüksek değerler faiz, refinansman ve bilanço kırılganlığı riskini artırabilir. Bazı sektörlerde yüksek oranlar normal olsa da, nakit üretimi zayıfsa ciddi risk oluşturur.""",
    "current_ratio": """Cari oran, şirketin kısa vadeli yükümlülüklerini dönen varlıklarla karşılama gücünü gösterir. Çok düşük değerler likidite stresi işareti olabilir. Aşırı yüksek değerler ise bazen verimsiz işletme sermayesi kullanımını düşündürebilir.""",
    "net_margin": """Net kâr marjı, tüm giderler sonrası satışların ne kadarının net kâra dönüştüğünü gösterir. Düşük marjlı sektörlerde küçük artışlar bile büyük iyileşme anlamına gelebilir. Tek seferlik gelir/gider etkilerini ayırmak önemlidir.""",
    "fcf": """Serbest nakit akışı, şirketin faaliyetlerinden ve gerekli yatırımlarından sonra kasada gerçekten ne kadar nakit kaldığını gösterir. Muhasebe kârı yüksek ama nakit üretimi zayıf şirketleri ayıklamak için en kritik metriklerden biridir. Değerleme açısından çok güçlü bir temeldir.""",
    "pe": """F/K oranı, piyasanın şirketin mevcut veya beklenen kârını kaç katla fiyatladığını gösterir. Düşük F/K her zaman ucuzluk değildir; bazen büyüme zayıflığı veya risk primi yüksektir. Bu oran sektör, büyüme ve kalite ile birlikte anlam kazanır.""",
    "pb": """PD/DD, piyasa değerinin defter değerine oranıdır. Özellikle banka ve varlık yoğun iş modellerinde önemlidir. Çok düşük PD/DD bazen ucuzluk, bazen de bilanço kalitesine yönelik güvensizlik anlamına gelebilir.""",
    "net_debt_ebitda": """Net Borç / FAVÖK oranı, şirketin net borcunu operasyonel kazancıyla ne kadar sürede taşıyabileceğine dair pratik bir kaldıraç göstergesidir. Düşük oran genellikle daha güvenli finansal yapı anlamına gelir. Döngüsel sektörlerde bu oran, kazançlar hızla değiştiği için dikkatle okunmalıdır.""",
    "altman_z": """Altman Z-Skoru, farklı bilanço ve gelir tablosu kalemlerini birleştirerek finansal stres/iflas riski hakkında erken sinyal verir. Düşük skorlar özellikle borçlu ve kırılgan şirketlerde uyarıcı olabilir. Tek başına hüküm vermez ama ucuz görünen riskli hisseleri ayıklamak için çok yararlıdır.""",
    "piotroski_f": """Piotroski F-Skoru, kârlılık, verimlilik ve bilanço sağlığına dair 9 ayrı kalite testinin toplamıdır. Yüksek skor, değer hisseleri içinde daha sağlıklı adayları ayırmaya yardım eder. En iyi kullanım, ucuzluk metrikleriyle birlikte kalite filtresi olarak kullanmaktır.""",
    "dcf": """DCF, gelecekte oluşması beklenen nakit akımlarını bugüne indirger ve şirketin teorik içsel değerini hesaplamaya çalışır. Gücü buradan gelir; fakat büyüme oranı, marj, sermaye maliyeti ve terminal değer varsayımlarına son derece hassastır. Bu yüzden DCF sonucu tek bir 'mutlak doğru fiyat' değil, senaryo aralığı olarak ele alınmalıdır.""",
    "sector_relative": """Sektöre göre pahalı/ucuz analizi, şirket çarpanlarının sektörel ortalamaya göre primli mi iskontolu mu olduğunu gösterir. Böylece bir şirketin sadece kendi başına ucuz görünmesini değil, ait olduğu grubun içinde nerede durduğunu anlarsın. Yorum yapılırken büyüme kalitesi, borçluluk ve sektör döngüsü mutlaka hesaba katılmalıdır.""",
    "trend_patt": """Trend with Patt Entry yaklaşımı, ana trendin yönünü filtrelerle belirleyip giriş zamanlamasını pattern ve fiyat davranışı ile yapmaya odaklanır. Amaç, en dipten veya en tepeden yakalamak değil; hareketin sağlıklı devam fazına disiplinli biçimde katılmaktır. Pattern yalnız başına değil, trend bağlamında değer üretir.""",
    "donchian": """Donchian kanalları, seçilen periyottaki en yüksek tepe ile en düşük dibi kanal olarak çizer. Bu sayede piyasanın mevcut kırılım sınırları görünür hale gelir. Klasik trend takip ve breakout sistemlerinde çok kullanılır; çünkü fiyat kanal dışına çıktığında rejim değişimi veya hareket hızlanması ihtimali artar.""",
    "donchian_520": """Donchian 5&20 yaklaşımı, daha kısa kanal ile tetikleyici, daha uzun kanal ile trend doğrulama mantığını birleştirir. Kısa kanal erken hareketleri, uzun kanal ise daha olgun trend teyidini göstermede kullanılabilir. Uygun stop ve pozisyon yönetimi olmadan tek başına yeterli değildir.""",
    "richard_dennis": """Richard Dennis / Turtle yaklaşımı, sistematik trend takibinin klasik örneklerinden biridir. Mantık, tahmin yapmak değil; güçlü kırılımları kurallı şekilde izlemek ve büyük trendleri taşımaktır. Zayıf yanı yatay piyasalarda bir dizi küçük zararı kabul etmeyi gerektirmesidir; güçlü yanı ise büyük trendler geldiğinde bu küçük kayıpları telafi edebilmesidir.""",
    "ema_fast": """Hızlı EMA periyodu, trendin kısa tarafını belirleyen parametredir. Periyot küçüldükçe gösterge fiyata daha duyarlı hale gelir ve sinyaller hızlanır; fakat gürültü ve yalancı sinyal de artar. Daha büyük periyot ise daha temiz ama daha geç tepki verir. Bu nedenle hızlı EMA, işlem ufkun ve sembolün volatilitesine göre seçilmelidir.""",
    "ema_slow": """Yavaş EMA periyodu, yapısal trendin omurgasını belirler. Büyük periyotlar ana yönü daha sakin ve daha güvenli gösterirken, dönüşleri geç yakalar. Küçük periyotlar daha erken sinyal üretir ama kısa dalgalanmalara daha açık hale gelir. Hızlı EMA ile yavaş EMA arasındaki ilişki, sistemin trend filtresinin temelidir.""",
    "sma_fast": """Hızlı SMA, kısa vadeli ortalama eğilimi gösterir ve bias üretiminde kullanılır. Kısa pencere nedeniyle fiyat değişimlerine daha erken tepki verir. Ancak özellikle yatay piyasada sık yön değişimi gösterebilir. Bu yüzden tek başına değil, yavaş SMA ve trend gücü filtreleriyle birlikte kullanılmalıdır.""",
    "sma_slow": """Yavaş SMA, orta/uzun vadeli yönü daha sakin şekilde özetler. Hızlı SMA ile karşılaştırıldığında ana eğilimin hangi tarafta olduğunu anlamanı sağlar. Geç tepki verme pahasına daha temiz bir yapı sunar.""",
    "rsi_period": """RSI periyodu, göstergenin hangi zaman penceresindeki momentum değişimlerini dikkate alacağını belirler. Küçük periyot daha hızlı ve daha gürültülü, büyük periyot daha yavaş ama daha dengeli sonuç üretir. Stratejinin ufkuna uygun olmayan periyot, gereksiz erken veya gereksiz geç sinyal yaratabilir.""",
    "bb_period": """Bollinger periyodu, orta bandın hangi pencere üzerinden hesaplanacağını belirler. Kısa periyot bantları daha çevik hale getirir; uzun periyot daha dengeli ama yavaş davranır. Sıkışma ve genişleme davranışını yorumlarken bu parametre çok etkilidir.""",
    "bb_std": """Bollinger standart sapma katsayısı, bantların orta ortalamadan ne kadar uzaklaşacağını belirler. Yüksek katsayı bantları genişletir ve daha az ama daha seçici temas üretir. Düşük katsayı ise daha sık temas ve daha hassas alarm yaratır.""",
    "atr_period": """ATR periyodu, volatilite ölçümünün hangi uzunlukta ortalanacağını belirler. Kısa ATR periyodu son hareketlere daha hızlı uyum sağlar, uzun periyot ise volatiliteyi daha dengeli yansıtır. Stop tasarımı ve risk yönetiminde bu parametrenin etkisi büyüktür.""",
    "vol_sma": """Hacim SMA periyodu, mevcut hacmin hangi ortalamaya göre güçlü ya da zayıf sayılacağını belirler. Kısa ortalama son ilgi değişimini daha hızlı yakalar, uzun ortalama ise daha stabil kıyas yapar. Hacim oranı ve spike tespitlerinde ana referans budur.""",
    "rsi_entry_level": """RSI giriş seviyesi, sistemin momentumu hangi eşikten sonra yeterli kabul edeceğini belirler. Seviye yükseldikçe daha seçici ama daha geç; düştükçe daha erken ama daha gürültülü girişler oluşur. Trend piyasası ile range piyasasında aynı eşik aynı kaliteyi vermez.""",
    "rsi_exit_level": """RSI çıkış seviyesi, momentumun ne kadar zayıflaması halinde sistemin risk azaltacağını belirler. Çok yüksek çıkış eşiği erken çıkışa, çok düşük eşik geç çıkışa neden olabilir. Stratejinin doğası ve hedeflediği hareket boyu bu parametrede önemlidir.""",
    "atr_pct_max": """ATR% üst sınırı, sistemin aşırı oynak gördüğü koşullarda yeni girişleri filtrelemek için kullanılır. Bu filtre, fırsatları azaltırken riskin kontrol altında tutulmasına yardım eder. Çok düşük ayarlanırsa güçlü trendler de kaçabilir; çok yüksek ayarlanırsa filtre anlamını yitirir.""",
    "initial_capital": """Başlangıç sermayesi, backtest ve risk metriklerinin parasal ölçeğini belirler. Yüzdesel performans aynı kalsa bile parasal drawdown ve işlem büyüklüğü bu değere göre değişir. Gerçekçi sermaye ile test yapmak psikolojik uygunluk açısından önemlidir.""",
    "commission_bps": """Komisyon baz puanı, her işlemde düşülecek doğrudan maliyet oranını ifade eder. Küçük görünebilir ama çok işlemli sistemlerde birikerek stratejinin net kalitesini belirgin şekilde etkiler. Gerçek hayata en yakın maliyet varsayımını kullanmak gerekir.""",
    "slippage_bps": """Slippage baz puanı, beklenen emir fiyatı ile fiili gerçekleşme arasındaki kaymayı simüle eder. Özellikle kırılım, stop ve düşük likidite senaryolarında çok önemlidir. Backtest performansını dürüstleştiren ana kalemlerden biridir.""",
    "atr_stop_mult": """ATR stop katsayısı, stop mesafesini volatiliteye göre ayarlamak için kullanılır. Düşük katsayı, sık ama küçük kayıplar; yüksek katsayı daha geniş stop ve daha az sıkışma anlamına gelebilir. Stratejinin trend takip mi yoksa kısa vade dönüş mü olduğuna göre uygun katsayı değişir.""",
    "risk_per_trade": """İşlem başına risk yüzdesi, her pozisyonda toplam sermayenin ne kadarını kaybetmeyi kabul ettiğini belirler. Doğru sistem bile kötü pozisyon boyutuyla batabilir. Bu parametre, stratejinin finansal sürdürülebilirliği kadar psikolojik taşınabilirliği için de kritiktir.""",
    "take_profit_mult": """Kâr alma katsayısı, giriş ile stop mesafesinin kaç katında parsiyel veya tam realize düşünüldüğünü tanımlar. Düşük hedefler daha sık kazanç, yüksek hedefler daha seyrek ama büyük kazanç yaratabilir. Risk/ödül yapısının en görünür parametrelerinden biridir.""",
    "time_stop_bars": """Zaman stopu, işlem belirli sayıda bar içinde beklenen performansı üretmezse pozisyonun kapanmasını sağlar. Böylece sermayenin verimsiz şekilde uzun süre bağlanması azaltılır. Özellikle ivme ve breakout stratejilerinde etkili bir disiplin aracıdır.""",
    "horizon_bars": """Tahmin ufku (horizon bars), modelin kaç bar sonrasını tahmin etmeye çalıştığını belirler. Kısa ufuklar daha gürültülü ama daha taktiksel, uzun ufuklar daha zor ama daha stratejik olabilir. Model başarısı ve hata büyüklüğü bu parametreye çok duyarlıdır."""
})


APP_EDUCATION_TEXTS.update({
    "dashboard_page": """Dashboard sekmesi, seçilen hissenin o anki teknik özetini tek bakışta okumak için tasarlanmıştır. Buradaki amaç tek bir göstergeden emir üretmek değil; fiyat, trend, volatilite, hacim, risk/ödül, backtest ve hacim profili gibi farklı katmanları aynı ekranda birleştirerek bağlam kurmaktır. Bu sayfa en sağlıklı şekilde yukarıdan aşağı okunur: önce genel skor ve trend durumu, sonra fiyat grafiği, sonra momentum/volatilite panelleri, ardından backtest ve VPVR. Eğer göstergeler aynı yönde kümeleniyorsa karar kalitesi artar; birbirleriyle çelişiyorsa temkin artmalıdır.""",
    "triple_page": """3 Ekranlı Sistem sekmesinde büyük zaman dilimi ana trendi, orta zaman dilimi düzeltmeyi ve küçük zaman dilimi tetikleyiciyi anlamak için kullanılır. Bu sayfa, tek zaman dilimine bakarak acele karar vermeyi azaltır. Haftalık görünüm yönü, günlük görünüm setup kalitesini, saatlik görünüm ise hassas zamanlamayı destekler. En iyi okuma, üç ekranın da aynı yönde hizalanıp hizalanmadığına bakmaktır.""",
    "future_page": """Future Price sekmesi, makine öğrenmesi tabanlı tahmin katmanıdır. Buradaki temel amaç 'kesin gelecek fiyatı' bulmak değil; farklı modellerin belirli bir bar ufkunda ne kadar tutarlı sonuç verdiğini görmek ve bunu mevcut trend, volatilite ve hata bandı ile birlikte yorumlamaktır. Model tahmini güçlü teknik bağlamla aynı yöne bakıyorsa yardımcı olabilir; teknik yapı ile zıt düşüyorsa tek başına belirleyici olmamalıdır.""",
    "indicator_stats_page": """İndikatör İstatistik sekmesi, tek tek göstergelerin geçmişte ne kadar sık oluştuğunu ve ne ölçüde işe yaradığını istatistiksel olarak incelemek için kullanılır. Bu sayfa, 'bu sinyal oluşunca tarihte ne olmuş?' sorusunu cevaplamaya çalışır. Ancak geçmiş frekans her zaman gelecek başarı garantisi vermez; piyasa rejimi değiştiğinde istatistikler de anlam değiştirebilir.""",
    "chart_patterns_page": """Grafik Formasyonları sekmesi, klasik teknik formasyonların varlığını ve bulunduğu konumu görmeye yarar. Formasyonlar tek başına kesin emir değildir; en iyi sonuç, trend filtresi, hacim ve seviye yapısı ile birlikte okunduğunda elde edilir. Özellikle kırılım beklenen yapılarda hacim teyidi çok önemlidir.""",
    "history_page": """Tarih Aralığı Analizi sekmesi, geçmişte seçilen bir dönem içinde fiyatın ve teknik göstergelerin birlikte nasıl davrandığını inceleme aracıdır. Bu sayfa geriye dönük öğrenme için çok değerlidir; belirli kriz, rallİ veya yatay dönemlerde sistemin nasıl çalıştığını görmeyi sağlar. Geçmişte oluşan sinyal kümelerini görerek stratejinin güçlü ve zayıf rejimlerini keşfetmek mümkündür.""",
    "financials_page": """Bilanço Analizi sekmesi, şirketin finansal kalitesini, büyümesini, borçluluğunu, nakit üretimini ve göreli değerlemesini bir arada okumayı amaçlar. Bu sayfada oranların tek tek iyi görünmesinden çok, birbirleriyle uyumu önemlidir. Yüksek büyüme ama zayıf nakit akışı, yüksek ROE ama aşırı borç veya düşük F/K ama kötü kalite gibi çelişkiler burada ayıklanmalıdır.""",
    "index_center_page": """BIST Endeks Merkezi, tek hisse yerine endeks düzeyinde bağlam üretmek için tasarlanmıştır. Amaç, seçilen endeksin genel yönünü, momentumunu ve tahmin yapısını görmek; böylece tekil hisseleri endeks rüzgârından bağımsız düşünmemektir. Endeks olumlu değilse hisse bazlı sinyallerin başarı ihtimali de düşebilir.""",
    "calendar_page": """Ekonomik Takvim sekmesi, makro veri akışının piyasayı nasıl etkileyebileceğini görmek için kullanılır. Burada veri başlığının kendisi kadar sürpriz potansiyeli, hangi sektörleri etkileyebileceği ve piyasadaki mevcut beklenti rejimi önemlidir. Makro veri günlerinde teknik seviyeler daha kolay delinip sahte hareketler de artabilir.""",
    "social_page": """X + YouTube Trends sekmesi, sosyal ilgi yoğunluğunu fiyat davranışıyla birlikte değerlendirmeye yarar. Artan dijital ilgi bazen erken momentum göstergesi olabilir, bazen de geç kalınmış spekülatif coşkuyu anlatabilir. Bu nedenle sosyal ilgi her zaman olumlu kabul edilmez; zamanlaması ve fiyat yapısıyla uyumu önemlidir.""",
    "heatmap_page": """Sektörel Heatmap sekmesi, hisseleri tek tek değil, ait oldukları sektör bağlamında okumaya yarar. Aynı piyasa içinde bazı sektörler lider, bazıları zayıf kalabilir. Göreli güç, pahalı/ucuzluk ve kalite karşılaştırmaları burada daha anlamlı hale gelir.""",
    "export_page": """Rapor sekmesi, uygulamanın farklı modüllerinden gelen özet bilgileri saklamak, paylaşmak ve daha sistemli incelemek için kullanılır. Bu sayfa kararın kendisini değil, kararın belgesini üretir. Düzenli raporlama, stratejinin zaman içindeki tutarlılığını izlemeye yardım eder.""",
    "scan_page": """Tarama sekmesi, evrendeki hisseler arasında seçilen şartlara göre hızlı filtreleme yapmak için kullanılır. Bu sayfa nihai karar yeri değil, aday bulma motorudur. Tarama sonrası en güçlü adaylar mutlaka Dashboard, Triple Screen, seviye yapısı ve finansal kalite ile tekrar doğrulanmalıdır.""",
    "trend_donchian_page": """Trend + Donchian Sistemleri sekmesi, trend takip ve breakout mantığını sistematik hale getiren bir eğitim/uygulama alanıdır. Buradaki sistemler en dipten veya tepeden yakalamaya değil, yön netleştikten sonra disiplinli biçimde katılmaya odaklanır. Güçlü tarafları büyük trendleri taşıyabilmeleri, zayıf tarafları ise yatay piyasalarda sık küçük sinyaller üretebilmeleridir.""",
    "twp_long_setup": """Trend with Patt Entry LONG setup, ana trend yukarıyken fiyat davranışının yeniden boğa yönünde hizalanmasını arar. Bu pratik sürümde temel mantık şudur: yapısal trend filtreleri olumlu olacak, kısa-orta vadeli ortalamalar boğa tarafını destekleyecek ve fiyat aksiyonunda boğa lehine bir giriş paternı görülecektir. LONG setup 'hemen al' demek değildir; özellikle hacim, seviye ve genel piyasa bağlamı ile teyit edildiğinde daha anlamlıdır.""",
    "twp_short_setup": """Trend with Patt Entry SHORT setup, ana yön aşağıyken fiyatın ayı lehine yeniden ivme kazandığı bölgeleri arar. Bu pratik sürümde trend filtresi negatif, ortalama yapısı baskı yönünde ve fiyat davranışı da short lehine olmalıdır. SHORT setup özellikle zayıf piyasa rejiminde ve önemli direnç bölgelerine yakınken daha güçlü anlam taşır.""",
    "twp_trend_filter": """Trend with Patt Entry içindeki trend filtresi, işlemin rüzgâra karşı mı yoksa rüzgârla birlikte mi açıldığını anlamak için kullanılır. Trend filtresi pozitifse boğa patternleri daha ciddiye alınır; negatifse ayı patternleri öncelik kazanır. En önemli işlevi, tek mum veya tek pattern kaynaklı acele işlemleri azaltmaktır.""",
    "twp_sma_relation": """TWP içindeki SMA50 / SMA200 ilişkisi, yapısal trendin çoğunlukla hangi tarafta olduğunu özetler. SMA50'nin SMA200 üzerinde olması trend tarafının daha çok LONG lehine olduğunu, altında olması ise SHORT lehine baskıyı anlatır. Ancak bu ilişki gecikmeli çalışır; bu yüzden pattern ve fiyat yapısıyla birlikte teyit edilmelidir.""",
    "donchian_upper_band": """Donchian üst bandı, seçilen periyottaki en yüksek tepeyi gösterir. Bu seviye, fiyatın mevcut işlem aralığının üst sınırıdır ve breakout sistemlerinde çok önemlidir. Fiyat bu bandın üzerine taşarsa yeni yüksek bölgeye geçiş sinyali oluşabilir; fakat hacim ve kabul davranışı teyit için çok önemlidir.""",
    "donchian_mid_band": """Donchian orta bandı, üst ve alt kanalın ortalamasıdır ve denge çizgisi gibi düşünülebilir. Fiyat orta bandın üstünde kaldığında üst bölgeye eğilim, altında kaldığında alt bölgeye baskı okunabilir. Tek başına sinyal değil, kanal içi konum yorumunda yardımcı referanstır.""",
    "donchian_lower_band": """Donchian alt bandı, seçilen periyottaki en düşük dibi gösterir. Bu seviye mevcut işlem aralığının alt sınırıdır. Fiyat bu bandın altına sarkarsa zayıflama veya aşağı kırılım sinyali oluşabilir; ancak sahte kırılımlar özellikle yatay dönemde sık görülebilir.""",
    "donchian_position": """Donchian konum metrikleri, fiyatın kanalın üstüne mi altına mı daha yakın olduğunu gösterir. Üst banda yakınlık çoğu zaman göreli güç, alt banda yakınlık ise zayıflama veya baskı anlatır. Yine de kanalın tam neresinde olduğu kadar, oraya nasıl geldiği de önemlidir.""",
    "d520_state": """5&20 sistemindeki 'durum' metriği, mevcut yapının kısa ortalama ile orta ortalama arasındaki ilişkiye göre LONG, SHORT veya nötr tarafta olup olmadığını özetler. Bu, tetik değil yön filtresidir. Sistem en iyi trend başlatan değil, trendi devam ederken izleyen araçlardan biridir.""",
    "d520_buy_sig": """5&20 sistemindeki AL sinyali, kısa tarafın orta tarafı yukarı kestiği veya boğa lehine hizalandığı anı temsil eder. Fakat yatay piyasalarda bu tür sinyaller sık sık bozulabilir. Hacim, seviye ve trend gücü ile teyit edildiğinde kalite artar.""",
    "d520_sell_sig": """5&20 sistemindeki SAT sinyali, kısa tarafın orta tarafı aşağı kestiği veya ayı lehine hizalandığı anı temsil eder. Güçlü ayı rejimlerinde etkili olabilir; ancak boğa piyasasında sık erken uyarı verip bozulabilir. Bu yüzden bağlam çok önemlidir.""",
    "rd_long_entry": """Richard Dennis / Turtle LONG giriş mantığı, fiyatın belirli bir lookback içindeki üst kırılım seviyesini aşması üzerine kuruludur. Buradaki amaç ucuzluk aramak değil, güç gösterisini disiplinle takip etmektir. Sistem, güçlü trendlerin başını kaçırsa bile devam eden hareketi taşımaya çalışır.""",
    "rd_short_entry": """Richard Dennis / Turtle SHORT giriş mantığı, fiyatın alt kırılım seviyesinin altına inmesiyle ayı tarafına katılmayı amaçlar. Bu yaklaşım özellikle zayıf ve trendli piyasalarda etkilidir. Ancak sert haber ve gap ortamında kırılımlar daha riskli hale gelir.""",
    "rd_exit_filter": """Richard Dennis sistemindeki çıkış filtresi, trend bozulduğunda pozisyonda kalmaya devam etmemek için kullanılır. Klasik mantık, giriş kanalından daha kısa bir ters kanal veya belirli kırılım seviyesi ile çıkışı yönetmektir. Bu, büyük trendleri taşırken gereksiz inadı azaltır.""",
    "rd_upper20": """Richard Dennis ekranındaki 20 günlük üst seviye, klasik turtle giriş üst bandını temsil eder. Fiyat bu seviyeyi yukarı geçiyorsa boğa yönlü breakout ihtimali doğar. Fakat hacim ve genel piyasa bağlamı kırılım kalitesi için hâlâ kritiktir.""",
    "rd_lower20": """Richard Dennis ekranındaki 20 günlük alt seviye, aşağı yönlü breakout referansıdır. Fiyat bu seviyenin altına iniyorsa satış baskısının yeni bir faza geçtiği düşünülebilir. Ancak yatay ve haber bazlı piyasada sahte kırılım riski unutulmamalıdır.""",
})
EDUCATION_TAB_SECTIONS: List[Tuple[str, List[Tuple[str, str]]]] = [
    ("Trend, Ortalama ve Yapı", [
        ("ema", "EMA"), ("sma_bias", "SMA Bias"), ("ema_fast", "Hızlı EMA Periyodu"),
        ("ema_slow", "Yavaş EMA Periyodu"), ("sma_fast", "Hızlı SMA"), ("sma_slow", "Yavaş SMA")
    ]),
    ("Momentum, Osilatörler ve Uyumsuzluklar", [
        ("rsi", "RSI"), ("rsi_period", "RSI Periyodu"), ("rsi_entry_level", "RSI Giriş Eşiği"),
        ("rsi_exit_level", "RSI Çıkış Eşiği"), ("macd", "MACD"), ("stochastic", "Stokastik"),
        ("stoch_rsi", "Stochastic RSI"), ("force_index", "Force Index"), ("elder_ray", "Elder-Ray"),
        ("adx", "ADX"), ("divergence", "Uyumsuzluk")
    ]),
    ("Volatilite, Bantlar ve Kanallar", [
        ("atr_pct", "ATR%"), ("atr_period", "ATR Periyodu"), ("atr_pct_max", "ATR% Üst Limiti"),
        ("bollinger", "Bollinger Bantları"), ("bb_width", "Bollinger Genişliği"),
        ("bb_period", "Bollinger Periyodu"), ("bb_std", "Bollinger Std Katsayısı"),
        ("donchian", "Donchian Kanalları"), ("donchian_520", "Donchian 5&20")
    ]),
    ("Hacim, Seviye ve Piyasa Yapısı", [
        ("volume_ratio", "Hacim Oranı"), ("vol_sma", "Hacim SMA Periyodu"), ("obv", "OBV"),
        ("vpvr", "VPVR"), ("poc", "POC"), ("poc_distance", "POC Uzaklık %"),
        ("support_resistance", "Destek / Direnç"), ("target_band", "Hedef Bandı"),
        ("risk_reward", "Risk / Ödül")
    ]),
    ("Sistemler ve Strateji Çerçevesi", [
        ("triple_screen", "Triple Screen"), ("trend_patt", "Trend with Patt Entry"),
        ("richard_dennis", "Richard Dennis / Turtle"), ("future_price", "Future Price"),
        ("horizon_bars", "Tahmin Ufku")
    ]),
    ("Backtest, Risk ve Portföy Disiplini", [
        ("backtest", "Backtest"), ("monte_carlo", "Monte Carlo"), ("sharpe", "Sharpe"),
        ("sortino", "Sortino"), ("calmar", "Calmar"), ("ulcer", "Ulcer Index"),
        ("kelly", "Kelly"), ("beta", "Beta"), ("information_ratio", "Information Ratio"),
        ("initial_capital", "Başlangıç Sermayesi"), ("commission", "Komisyon"),
        ("commission_bps", "Komisyon (bps)"), ("slippage", "Slippage"),
        ("slippage_bps", "Slippage (bps)"), ("atr_stop_mult", "ATR Stop Katsayısı"),
        ("risk_per_trade", "İşlem Başına Risk %"), ("take_profit_mult", "Kâr Alma Katsayısı"),
        ("time_stop_bars", "Zaman Stopu (bar)")
    ]),
    ("Future Price Model Kalitesi ve Rejimler", [
        ("mae", "MAE"), ("rmse", "RMSE"), ("mape", "MAPE"), ("direction_acc", "Yön Doğruluğu"),
        ("confidence", "Güven Skoru"), ("train_test", "Eğitim/Test"), ("trend_regime", "Trend Rejimi"),
        ("vol_regime", "Volatilite Rejimi")
    ]),
    ("Finansal Analiz ve Değerleme", [
        ("roe", "ROE"), ("revenue_growth", "Net Satış Büyümesi"), ("ebitda", "FAVÖK"),
        ("ebitda_margin", "FAVÖK Marjı"), ("debt_equity", "Borç / Özsermaye"),
        ("current_ratio", "Cari Oran"), ("net_margin", "Net Kâr Marjı"), ("fcf", "Serbest Nakit Akışı"),
        ("pe", "F/K"), ("pb", "PD/DD"), ("net_debt_ebitda", "Net Borç / FAVÖK"),
        ("altman_z", "Altman Z"), ("piotroski_f", "Piotroski F"), ("dcf", "DCF"),
        ("sector_relative", "Sektöre Göre Pahalı / Ucuz")
    ]),
]


def render_education_center_tab():
    st.header("📚 Eğitim Merkezi")
    st.caption("Bu sekme, uygulamadaki göstergeler, osilatörler, parametreler, risk metrikleri, finansal oranlar ve sistemler için kapsamlı eğitim notlarını tek yerde toplar.")
    st.info("Önemli not: Buradaki açıklamalar eğitim amaçlıdır. Hiçbir gösterge, oran veya model tek başına kesin yatırım kararı verdirmez; en sağlıklı yaklaşım, birden çok kanıtı aynı yönde toplamaktır.")

    with st.expander("↘ Eğitim merkezinin kapsamlı rehberini aç", expanded=False):
        for section_title, items in EDUCATION_TAB_SECTIONS:
            st.markdown(f"## {section_title}")
            for key, label in items:
                st.markdown(f"### {label}")
                st.markdown(APP_EDUCATION_TEXTS.get(key, "Bu başlık için açıklama bulunamadı."))
                st.markdown("---")



# =============================
# Indicators
# =============================


def ema(s: pd.Series, span: int) -> pd.Series:
    return s.ewm(span=span, adjust=False).mean()


def sma(s: pd.Series, window: int) -> pd.Series:
    return s.rolling(int(window), min_periods=int(window)).mean()


def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """
    Wilder RSI. İlk ortalama SMA ile başlatılır, devamı Wilder RMA mantığıyla yürür.
    Bu, ewm(alpha=1/period) yaklaşımına çok yakındır ama başlangıç etkisini daha doğru yönetir.
    """
    close = pd.to_numeric(close, errors="coerce").astype(float)
    period = max(1, int(period))
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)

    avg_gain = gain.rolling(period, min_periods=period).mean()
    avg_loss = loss.rolling(period, min_periods=period).mean()

    # Wilder recursive smoothing
    for i in range(period + 1, len(close)):
        if pd.notna(avg_gain.iloc[i - 1]):
            avg_gain.iloc[i] = ((avg_gain.iloc[i - 1] * (period - 1)) + gain.iloc[i]) / period
        if pd.notna(avg_loss.iloc[i - 1]):
            avg_loss.iloc[i] = ((avg_loss.iloc[i - 1] * (period - 1)) + loss.iloc[i]) / period

    rs = avg_gain / avg_loss.replace(0, np.nan)
    out = 100 - (100 / (1 + rs))
    return out.replace([np.inf, -np.inf], np.nan).fillna(50)


def macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    macd_line = ema(close, fast) - ema(close, slow)
    signal_line = ema(macd_line, signal)
    hist = macd_line - signal_line
    return macd_line, signal_line, hist


def bollinger(close: pd.Series, period: int = 20, std_mult: float = 2.0):
    mid = close.rolling(period).mean()
    sd = close.rolling(period).std()
    upper = mid + std_mult * sd
    lower = mid - std_mult * sd
    return mid, upper, lower


def true_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    prev_close = close.shift(1)
    tr = pd.concat(
        [(high - low), (high - prev_close).abs(), (low - prev_close).abs()],
        axis=1,
    ).max(axis=1)
    return tr


def atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    tr = true_range(high, low, close)
    return tr.ewm(alpha=1 / period, adjust=False).mean()


def obv(close: pd.Series, volume: pd.Series) -> pd.Series:
    direction = np.sign(close.diff()).fillna(0)
    return (direction * volume).cumsum()


def max_drawdown(eq: pd.Series) -> float:
    if eq is None or len(eq) == 0:
        return 0.0
    peak = eq.cummax()
    dd = (eq / peak) - 1.0
    return float(dd.min())


# =============================
# YENİ EKLENEN İNDİKATÖRLER (TRIPLE SCREEN İÇİN)
# =============================
def force_index(close: pd.Series, volume: pd.Series) -> pd.Series:
    return volume * (close - close.shift(1))

def stochastic(high: pd.Series, low: pd.Series, close: pd.Series, k_period: int = 5, d_period: int = 3):
    lowest_low = low.rolling(window=k_period).min()
    highest_high = high.rolling(window=k_period).max()
    k = 100 * ((close - lowest_low) / (highest_high - lowest_low))
    d = k.rolling(window=d_period).mean()
    return k.fillna(50), d.fillna(50)

def elder_ray(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 13):
    e = ema(close, period)
    bull_power = high - e
    bear_power = low - e
    return e, bull_power, bear_power


def _find_pivot_positions(
    series: pd.Series,
    mode: str = "low",
    left: int = 2,
    right: int = 2,
    plateau_tol_pct: float = 0.001,
    volume: Optional[pd.Series] = None,
    volume_sma_period: int = 20,
) -> List[int]:
    """
    Tek bar pivot yerine önce plato bölgesini tanır:
    1) Yakın/eşit fiyatlı barları plato olarak gruplar.
    2) Divergence için platonun temsilcisini seçer:
       - volume verisi varsa en yüksek volume_ratio barı
       - yoksa orta bar
    3) Pivot testini plato DIŞINDA kalan sol/sağ barlara uygular.
    """
    s = pd.to_numeric(series, errors="coerce")
    vals = s.values
    pivots: List[int] = []

    if len(vals) < (left + right + 3):
        return pivots

    vol_ratio = None
    if volume is not None:
        try:
            v = pd.to_numeric(volume.reindex(series.index), errors="coerce")
            v_sma = v.rolling(int(volume_sma_period), min_periods=1).mean().replace(0, np.nan)
            vol_ratio = (v / v_sma).replace([np.inf, -np.inf], np.nan)
        except Exception:
            vol_ratio = None

    seen_plateaus = set()

    for i in range(left, len(vals) - right):
        center = vals[i]
        if not np.isfinite(center):
            continue

        tol = max(abs(center) * float(plateau_tol_pct), 1e-12)

        plateau_left = i
        plateau_right = i

        while plateau_left > 0 and np.isfinite(vals[plateau_left - 1]) and abs(vals[plateau_left - 1] - center) <= tol:
            plateau_left -= 1
        while plateau_right < len(vals) - 1 and np.isfinite(vals[plateau_right + 1]) and abs(vals[plateau_right + 1] - center) <= tol:
            plateau_right += 1

        plateau_key = (plateau_left, plateau_right, mode)
        if plateau_key in seen_plateaus:
            continue
        seen_plateaus.add(plateau_key)

        plateau_positions = list(range(plateau_left, plateau_right + 1))
        plateau_mid = (plateau_left + plateau_right) // 2
        rep_idx = plateau_mid

        if vol_ratio is not None:
            plateau_scores = vol_ratio.iloc[plateau_positions]
            if plateau_scores.notna().any():
                max_score = plateau_scores.max()
                best_positions = [
                    plateau_positions[j]
                    for j, val in enumerate(plateau_scores.values)
                    if pd.notna(val) and val == max_score
                ]
                if best_positions:
                    rep_idx = min(best_positions, key=lambda x: (abs(x - plateau_mid), x))

        local_left = vals[max(0, rep_idx - left):plateau_left]
        local_right = vals[plateau_right + 1:min(len(vals), rep_idx + right + 1)]

        if len(local_left) == 0 or len(local_right) == 0:
            continue
        if not np.isfinite(local_left).all() or not np.isfinite(local_right).all():
            continue

        plateau_value = float(np.nanmean(vals[plateau_left:plateau_right + 1]))

        if mode == "low":
            if plateau_value < local_left.min() and plateau_value < local_right.min():
                pivots.append(int(rep_idx))
        else:
            if plateau_value > local_left.max() and plateau_value > local_right.max():
                pivots.append(int(rep_idx))

    return sorted(set(pivots))

def _series_extreme_near(series: pd.Series, pos: int, mode: str = "low", radius: int = 2) -> float:
    s = pd.to_numeric(series, errors="coerce")
    start = max(0, int(pos) - int(radius))
    end = min(len(s), int(pos) + int(radius) + 1)
    window = s.iloc[start:end].dropna()
    if window.empty:
        return np.nan
    return float(window.min()) if mode == "low" else float(window.max())


def _pivot_divergence_core(
    close: pd.Series,
    indicator: pd.Series,
    lookback: int = 30,
    mode: str = "bull",
    volume: Optional[pd.Series] = None,
) -> Tuple[bool, int]:
    if close is None or indicator is None:
        return False, 0

    lb = max(int(lookback), 12)
    c = pd.to_numeric(close.tail(lb), errors="coerce")
    ind = pd.to_numeric(indicator.reindex(c.index), errors="coerce")
    vol = None
    if volume is not None:
        try:
            vol = pd.to_numeric(volume.reindex(c.index), errors="coerce")
        except Exception:
            vol = None

    if len(c) < 8 or len(ind) < 8:
        return False, 0

    pivot_mode = "low" if mode == "bull" else "high"
    price_pivots = _find_pivot_positions(
        c,
        mode=pivot_mode,
        left=2,
        right=2,
        plateau_tol_pct=0.001,
        volume=vol,
        volume_sma_period=20,
    )

    if len(price_pivots) < 2:
        return False, 0

    recent_pivots = price_pivots[-6:]

    for newer_idx in range(len(recent_pivots) - 1, 0, -1):
        newer = recent_pivots[newer_idx]

        for older_idx in range(newer_idx - 1, -1, -1):
            older = recent_pivots[older_idx]
            sep = newer - older

            if sep < 4:
                continue
            if sep > lb - 3:
                continue

            p1 = float(c.iloc[older])
            p2 = float(c.iloc[newer])

            i1 = _series_extreme_near(ind, older, mode=pivot_mode, radius=2)
            i2 = _series_extreme_near(ind, newer, mode=pivot_mode, radius=2)

            if not (np.isfinite(p1) and np.isfinite(p2) and np.isfinite(i1) and np.isfinite(i2)):
                continue

            if mode == "bull":
                if p2 < p1 and i2 > i1:
                    bars_ago = (len(c) - 1) - newer
                    return True, int(bars_ago)
            else:
                if p2 > p1 and i2 < i1:
                    bars_ago = (len(c) - 1) - newer
                    return True, int(bars_ago)

    return False, 0


def check_bullish_divergence(
    close: pd.Series,
    indicator: pd.Series,
    lookback: int = 30,
    volume: Optional[pd.Series] = None,
) -> Tuple[bool, int]:
    return _pivot_divergence_core(close, indicator, lookback=lookback, mode="bull", volume=volume)


def check_bearish_divergence(
    close: pd.Series,
    indicator: pd.Series,
    lookback: int = 30,
    volume: Optional[pd.Series] = None,
) -> Tuple[bool, int]:
    return _pivot_divergence_core(close, indicator, lookback=lookback, mode="bear", volume=volume)

def adx_indicator(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14):
    up = high - high.shift(1)
    down = low.shift(1) - low
    
    plus_dm = pd.Series(np.where((up > down) & (up > 0), up, 0.0), index=high.index)
    minus_dm = pd.Series(np.where((down > up) & (down > 0), down, 0.0), index=high.index)
    
    tr = true_range(high, low, close)
    
    tr_smooth = pd.Series(tr, index=high.index).ewm(alpha=1/period, adjust=False).mean()
    pdm_smooth = plus_dm.ewm(alpha=1/period, adjust=False).mean()
    mdm_smooth = minus_dm.ewm(alpha=1/period, adjust=False).mean()
    
    pdi = 100 * (pdm_smooth / tr_smooth.replace(0, np.nan))
    mdi = 100 * (mdm_smooth / tr_smooth.replace(0, np.nan))
    
    dx = 100 * (abs(pdi - mdi) / (pdi + mdi).replace(0, np.nan))
    adx = dx.ewm(alpha=1/period, adjust=False).mean()
    
    return adx.fillna(0), pdi.fillna(0), mdi.fillna(0)


# =============================
# KANGAROO TAIL (KANGURU KUYRUĞU)
# =============================

def add_kangaroo_tails(df: pd.DataFrame, lookback: int = 20) -> pd.DataFrame:
    df = df.copy()
    df["KANGAROO_BULL"] = 0
    df["KANGAROO_BEAR"] = 0

    body = (df["Close"] - df["Open"]).abs()
    trange = df["High"] - df["Low"]
    lower_wick = df[["Open", "Close"]].min(axis=1) - df["Low"]
    upper_wick = df["High"] - df[["Open", "Close"]].max(axis=1)

    rolling_min = df["Low"].rolling(window=lookback).min()
    rolling_max = df["High"].rolling(window=lookback).max()

    atr_approx = trange.rolling(10).mean()
    valid_trange = trange > 0
    pivot_tol = 0.001

    bull_cond = valid_trange & (df["Low"] <= (rolling_min * (1.0 + pivot_tol))) & ((body / trange) <= 0.3) & ((lower_wick / trange) >= 0.6) & (trange >= atr_approx * 0.8)
    bear_cond = valid_trange & (df["High"] >= (rolling_max * (1.0 - pivot_tol))) & ((body / trange) <= 0.3) & ((upper_wick / trange) >= 0.6) & (trange >= atr_approx * 0.8)

    df.loc[bull_cond, "KANGAROO_BULL"] = 1
    df.loc[bear_cond, "KANGAROO_BEAR"] = 1
    return df


# =============================
# PRICE ACTION PATTERNS (CANDLESTICKS)
# =============================
def add_candlestick_patterns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    O = df["Open"]
    H = df["High"]
    L = df["Low"]
    C = df["Close"]
    EMA50_REF = df["EMA50"] if "EMA50" in df.columns else ema(C, 50)

    Body = (C - O).abs()
    Range = H - L
    UpperWick = H - df[["Open", "Close"]].max(axis=1)
    LowerWick = df[["Open", "Close"]].min(axis=1) - L
    AvgRange = Range.rolling(10).mean()

    is_bull = C > O
    is_bear = C < O

    # 4. Doji
    df["PATTERN_DOJI"] = (Body <= 0.1 * Range) & (Range > 0)
    
    # 5. Long-Legged Doji
    df["PATTERN_LL_DOJI"] = df["PATTERN_DOJI"] & (UpperWick >= 0.35 * Range) & (LowerWick >= 0.35 * Range) & (Range > AvgRange * 0.8)

    # 1. Hammer / Hanging Man
    shape_hammer = (LowerWick >= 2 * Body) & (UpperWick <= 0.2 * Range) & (Body > 0.02 * Range)
    df["PATTERN_HAMMER"] = shape_hammer & (C < EMA50_REF)
    df["PATTERN_HANGING_MAN"] = shape_hammer & (C > EMA50_REF)

    # 2. Shooting Star / Inverted Hammer
    shape_star = (UpperWick >= 2 * Body) & (LowerWick <= 0.2 * Range) & (Body > 0.02 * Range)
    df["PATTERN_SHOOTING_STAR"] = shape_star & (C > EMA50_REF)
    df["PATTERN_INV_HAMMER"] = shape_star & (C < EMA50_REF)

    # 7. Marubozu
    df["PATTERN_MARUBOZU_BULL"] = is_bull & (Body >= 0.85 * Range) & (Range > AvgRange * 0.5)
    df["PATTERN_MARUBOZU_BEAR"] = is_bear & (Body >= 0.85 * Range) & (Range > AvgRange * 0.5)

    # Shifting values for multi-day patterns
    prev_is_bear = is_bear.shift(1)
    prev_is_bull = is_bull.shift(1)
    prev_O = O.shift(1)
    prev_C = C.shift(1)

    # 3. Engulfing
    df["PATTERN_ENGULFING_BULL"] = is_bull & prev_is_bear & (O <= prev_C) & (C >= prev_O) & (Body > (prev_O - prev_C))
    df["PATTERN_ENGULFING_BEAR"] = is_bear & prev_is_bull & (O >= prev_C) & (C <= prev_O) & (Body > (prev_C - prev_O))

    # 6. Harami
    df["PATTERN_HARAMI_BULL"] = is_bull & prev_is_bear & (O > prev_C) & (C < prev_O) & ((prev_O - prev_C) > AvgRange * 0.5)
    df["PATTERN_HARAMI_BEAR"] = is_bear & prev_is_bull & (O < prev_C) & (C > prev_O) & ((prev_C - prev_O) > AvgRange * 0.5)

    # 8. Tweezer Top / Bottom
    prev_H = H.shift(1)
    prev_L = L.shift(1)
    df["PATTERN_TWEEZER_TOP"] = (abs(H - prev_H) <= 0.002 * C) & is_bear & prev_is_bull & (H > df.get("EMA50", C))
    df["PATTERN_TWEEZER_BOTTOM"] = (abs(L - prev_L) <= 0.002 * C) & is_bull & prev_is_bear & (L < df.get("EMA50", C))

    # 10. Piercing Pattern / Dark Cloud Cover
    df["PATTERN_PIERCING"] = is_bull & prev_is_bear & (O < L.shift(1)) & (C > (prev_O + prev_C)/2) & (C < prev_O)
    df["PATTERN_DARK_CLOUD"] = is_bear & prev_is_bull & (O > H.shift(1)) & (C < (prev_O + prev_C)/2) & (C > prev_O)

    # 9. Morning Star / Evening Star (3-day pattern)
    prev2_is_bear = is_bear.shift(2)
    prev2_is_bull = is_bull.shift(2)
    prev2_O = O.shift(2)
    prev2_C = C.shift(2)

    df["PATTERN_MORNING_STAR"] = is_bull & prev2_is_bear & (prev_C < prev2_C) & (O > prev_C) & (C > (prev2_O + prev2_C)/2)
    df["PATTERN_EVENING_STAR"] = is_bear & prev2_is_bull & (prev_C > prev2_C) & (O < prev_C) & (C < (prev2_O + prev2_C)/2)

    return df


# =============================
# Overbought / Speculation Indicators
# =============================
def add_overbought_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["RSI_OVERBOUGHT"] = (df["RSI"] > 70).astype(int)
    df["RSI_OVERSOLD"] = (df["RSI"] < 30).astype(int)

    bb_den = (df["BB_upper"] - df["BB_lower"]).replace(0, np.nan)
    df["BB_PERCENT_B"] = ((df["Close"] - df["BB_lower"]) / bb_den).replace([np.inf, -np.inf], np.nan)
    df["BB_OVERBOUGHT"] = (df["Close"] > df["BB_upper"]).astype(int)
    df["BB_OVERSOLD"] = (df["Close"] < df["BB_lower"]).astype(int)

    df["VOLUME_SMA20"] = df["Volume"].rolling(20).mean()
    df["VOLUME_SPIKE"] = (df["Volume"] > df["VOLUME_SMA20"] * 1.5).astype(int)

    df["PRICE_TO_EMA50"] = (df["Close"] / df["EMA50"] - 1) * 100
    df["PRICE_TO_EMA200"] = (df["Close"] / df["EMA200"] - 1) * 100
    df["PRICE_EXTREME"] = ((df["PRICE_TO_EMA50"] > 20) | (df["PRICE_TO_EMA200"] > 30)).astype(int)

    def stoch_rsi(series, period=14, smooth_k=3, smooth_d=3):
        rsi_vals = series
        min_rsi = rsi_vals.rolling(period).min()
        max_rsi = rsi_vals.rolling(period).max()
        den = (max_rsi - min_rsi).replace(0, np.nan)
        stoch = 100 * (rsi_vals - min_rsi) / den
        stoch = stoch.replace([np.inf, -np.inf], np.nan).fillna(50)
        k = stoch.rolling(smooth_k).mean()
        d = k.rolling(smooth_d).mean()
        return k, d

    df["STOCH_RSI_K"], df["STOCH_RSI_D"] = stoch_rsi(df["RSI"])
    df["STOCH_OVERBOUGHT"] = (df["STOCH_RSI_K"] > 80).astype(int)

    df["VOLUME_DIR"] = np.sign(df["Volume"].diff()).fillna(0)
    df["PRICE_DIR"] = np.sign(df["Close"].diff()).fillna(0)
    df["WEAK_UPTREND"] = ((df["PRICE_DIR"] > 0) & (df["VOLUME_DIR"] < 0)).astype(int)

    return df


def detect_speculation(df: pd.DataFrame) -> Dict[str, Any]:
    last = df.iloc[-1]
    result = {
        "overbought_score": 0,
        "oversold_score": 0,
        "speculation_score": 0,
        "details": {},
    }

    if last["RSI"] > 70:
        result["overbought_score"] += 40
        result["details"]["rsi"] = f"Aşırı alım (RSI: {last['RSI']:.1f})"
    elif last["RSI"] < 30:
        result["oversold_score"] += 50
        result["details"]["rsi"] = f"Aşırı satım (RSI: {last['RSI']:.1f})"

    if bool(last["BB_OVERBOUGHT"]):
        result["overbought_score"] += 20
        result["details"]["bb"] = "Fiyat Bollinger üst bandında"
    elif bool(last["BB_OVERSOLD"]):
        result["oversold_score"] += 50
        result["details"]["bb"] = "Fiyat Bollinger alt bandında"

    if bool(last["STOCH_OVERBOUGHT"]):
        result["overbought_score"] += 20
        result["details"]["stoch"] = "Stokastik RSI aşırı alımda"

    if bool(last["VOLUME_SPIKE"]):
        result["speculation_score"] += 60
        result["details"]["volume"] = "Ani hacim artışı (spekülasyon)"

    if bool(last["PRICE_EXTREME"]):
        result["overbought_score"] += 20
        result["details"]["price_extreme"] = f"Fiyat EMA'dan çok uzak (EMA50: %{last['PRICE_TO_EMA50']:.1f})"

    if bool(last["WEAK_UPTREND"]):
        result["speculation_score"] += 40
        result["details"]["weak_trend"] = "Fiyat yükselirken hacim düşüyor (zayıflama)"

    result["overbought_score"] = min(100, result["overbought_score"])
    result["oversold_score"] = min(100, result["oversold_score"])
    result["speculation_score"] = min(100, result["speculation_score"])

    if result["overbought_score"] >= 60:
        result["verdict"] = "AŞIRI DEĞERLİ (SAT bölgesi)"
    elif result["oversold_score"] >= 60:
        result["verdict"] = "AŞIRI DEĞERSİZ (AL bölgesi)"
    elif result["speculation_score"] >= 60:
        result["verdict"] = "SPEKÜLATİF HAREKET (dikkatli olunmalı)"
    else:
        result["verdict"] = "NÖTR (normal değer aralığı)"

    return result


# =============================
# Feature builder
# =============================
def build_features(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    df = df.copy()

    fast_len = int(cfg["ema_fast"])
    slow_len = int(cfg["ema_slow"])

    df["EMA50"] = ema(df["Close"], fast_len)
    df["EMA200"] = ema(df["Close"], slow_len)
    df["SMA_FAST"] = sma(df["Close"], fast_len)
    df["SMA_SLOW"] = sma(df["Close"], slow_len)
    df["RSI"] = rsi(df["Close"], int(cfg["rsi_period"]))
    df["MACD"], df["MACD_signal"], df["MACD_hist"] = macd(df["Close"], 12, 26, 9)
    df["BB_mid"], df["BB_upper"], df["BB_lower"] = bollinger(df["Close"], int(cfg["bb_period"]), float(cfg["bb_std"]))
    df["ATR"] = atr(df["High"], df["Low"], df["Close"], int(cfg["atr_period"]))
    df["OBV"] = obv(df["Close"], df["Volume"])
    df["OBV_EMA"] = ema(df["OBV"], 21)
    df["VOL_SMA"] = df["Volume"].rolling(int(cfg["vol_sma"])).mean()

    df["ATR_PCT"] = (df["ATR"] / df["Close"]).replace([np.inf, -np.inf], np.nan)

    bb_mid_safe = _safe_positive_denominator(df["BB_mid"])
    df["BB_WIDTH"] = ((df["BB_upper"] - df["BB_lower"]) / bb_mid_safe).replace([np.inf, -np.inf], np.nan)

    vol_sma_safe = _safe_positive_denominator(df["VOL_SMA"])
    df["VOL_RATIO"] = (df["Volume"] / vol_sma_safe).replace([np.inf, -np.inf], np.nan)

    df = add_overbought_indicators(df)
    df = add_kangaroo_tails(df)
    df = add_candlestick_patterns(df)
    return df


# =============================
# Market regime filters (GÜNCELLENDİ: Zaman Serisi Döndürür)
# =============================
@st.cache_data(ttl=6 * 3600, show_spinner=False)
def get_spy_regime_series() -> pd.Series:
    try:
        spy = yf.download("SPY", period="2y", interval="1d", auto_adjust=True, progress=False)
        spy = _flatten_yf(spy)
        if spy.empty or len(spy) < 260:
            return pd.Series(dtype=bool)
        spy["EMA200"] = ema(spy["Close"], 200)
        return spy["Close"] > spy["EMA200"]
    except Exception:
        return pd.Series(dtype=bool)


@st.cache_data(ttl=6 * 3600, show_spinner=False)
def get_bist_regime_series() -> pd.Series:
    try:
        xu100 = yf.download("XU100.IS", period="2y", interval="1d", auto_adjust=True, progress=False)
        xu100 = _flatten_yf(xu100)
        if xu100.empty or len(xu100) < 200:
            return pd.Series(dtype=bool)
        xu100["EMA200"] = ema(xu100["Close"], 200)
        return xu100["Close"] > xu100["EMA200"]
    except Exception:
        return pd.Series(dtype=bool)


# =============================
# Higher timeframe trend filter (GÜNCELLENDİ: Zaman Serisi Döndürür)
# =============================
@st.cache_data(ttl=6 * 3600, show_spinner=False)
def get_higher_tf_trend_series(ticker: str, higher_tf_interval: str = "1wk", ema_period: int = 200) -> pd.Series:
    try:
        df = yf.download(ticker, period="5y", interval=higher_tf_interval, auto_adjust=True, progress=False)
        df = _flatten_yf(df)
        if df.empty or len(df) < min(ema_period, 100):
            return pd.Series(dtype=bool)
        df["EMA"] = ema(df["Close"], ema_period)
        return df["Close"] > df["EMA"]
    except Exception:
        return pd.Series(dtype=bool)


# =============================
# Strategy: scoring + checkpoints (GÜNCELLENDİ: Seriler ile Look-ahead önleme)
# =============================
def signal_with_checkpoints(
    df: pd.DataFrame,
    cfg: dict,
    market_filter_series: pd.Series = None,
    higher_tf_filter_series: pd.Series = None,
):
    df = df.copy()

    liq_ok = (df["Volume"] > df["VOL_SMA"]).fillna(False)
    trend_ok = (df["Close"] > df["EMA200"]) & (df["EMA50"] > df["EMA200"])

    if market_filter_series is not None and not market_filter_series.empty:
        aligned_market = market_filter_series.reindex(df.index).ffill().fillna(True)
    else:
        aligned_market = pd.Series(True, index=df.index)

    if higher_tf_filter_series is not None and not higher_tf_filter_series.empty:
        aligned_htf = higher_tf_filter_series.reindex(df.index).ffill().fillna(True)
    else:
        aligned_htf = pd.Series(True, index=df.index)

    rsi_ok = df["RSI"] > cfg["rsi_entry_level"]
    rsi_cross = (df["RSI"] > cfg["rsi_entry_level"]) & (df["RSI"].shift(1) <= cfg["rsi_entry_level"])

    macd_ok = df["MACD_hist"] > 0
    macd_turn = (df["MACD_hist"] > 0) & (df["MACD_hist"].shift(1) <= 0)

    atr_pct = (df["ATR"] / df["Close"]).replace([np.inf, -np.inf], np.nan)
    vol_ok = atr_pct < cfg["atr_pct_max"]

    bb_ok = df["Close"] > df["BB_mid"]
    bb_break = (df["Close"] > df["BB_upper"]) & trend_ok

    obv_ok = df["OBV"] > df["OBV_EMA"]

    w = {"liq": 10, "trend": 25, "rsi": 15, "macd": 15, "vol": 10, "bb": 15, "obv": 10}
    score = (
        w["liq"] * liq_ok.astype(int)
        + w["trend"] * trend_ok.astype(int)
        + w["rsi"] * rsi_ok.astype(int)
        + w["macd"] * macd_ok.astype(int)
        + w["vol"] * vol_ok.astype(int)
        + w["bb"] * (bb_ok | bb_break).astype(int)
        + w["obv"] * obv_ok.astype(int)
    ).astype(float)

    entry_triggers = (rsi_cross.astype(int) + macd_turn.astype(int) + bb_break.astype(int)) >= 1
    
    # Burada aligned_market ve aligned_htf mevcut günün bilgisidir. Backtest tarafında shift(1) yapılacağı için
    # geleceği görme hatası tamamen ortadan kalkar.
    entry = trend_ok & vol_ok & liq_ok & entry_triggers & aligned_market & aligned_htf

    exit_ = (
        (df["Close"] < df["EMA50"])
        | (df["MACD_hist"] < 0)
        | (df["RSI"] < cfg["rsi_exit_level"])
        | (df["Close"] < df["BB_mid"])
    )

    df["SCORE"] = score
    df["ENTRY"] = entry.astype(int)
    df["EXIT"] = exit_.astype(int)

    last = df.iloc[-1]
    cp = {
        "Market Filter OK": bool(aligned_market.iloc[-1]),
        "Higher TF Filter OK": bool(aligned_htf.iloc[-1]),
        "Liquidity (Volume > VolSMA)": bool(last["Volume"] > last["VOL_SMA"]) if pd.notna(last["VOL_SMA"]) else False,
        "Trend (Close>EMA200 & EMA50>EMA200)": bool((last["Close"] > last["EMA200"]) and (last["EMA50"] > last["EMA200"]))
        if pd.notna(last["EMA200"]) else False,
        f"RSI > {cfg['rsi_entry_level']}": bool(last["RSI"] > cfg["rsi_entry_level"]) if pd.notna(last["RSI"]) else False,
        "MACD Hist > 0": bool(last["MACD_hist"] > 0) if pd.notna(last["MACD_hist"]) else False,
        f"ATR% < {cfg['atr_pct_max']:.2%}": bool((last["ATR"] / last["Close"]) < cfg["atr_pct_max"])
        if pd.notna(last["ATR"]) and pd.notna(last["Close"]) else False,
        "Bollinger (Close>BB_mid or Breakout)": bool((last["Close"] > last["BB_mid"]) or (last["Close"] > last["BB_upper"]))
        if pd.notna(last["BB_mid"]) else False,
        "OBV > OBV_EMA": bool(last["OBV"] > last["OBV_EMA"]) if pd.notna(last["OBV_EMA"]) else False,
    }
    return df, cp


# =============================
# Backtest (long-only) + advanced exits
# =============================

def backtest_long_only(
    df: pd.DataFrame,
    cfg: dict,
    risk_free_annual: float,
    benchmark_returns: Optional[pd.Series] = None,
):
    df = df.copy()

    # Sinyal bar kapanışında oluşur, işlem bir sonraki barın açılışında yapılır.
    # Bu nedenle ENTRY/EXIT bir bar kaydırılır. Aynı işlem barında stop/target çalıştırılmaz.
    entry_sig = df["ENTRY"].shift(1).fillna(0).astype(int).values
    exit_sig = df["EXIT"].shift(1).fillna(0).astype(int).values

    cash = float(cfg["initial_capital"])
    shares = 0.0
    stop = np.nan
    entry_price = np.nan
    target_price = np.nan
    bars_held = 0
    half_sold = False

    trades = []
    equity_curve = []

    commission = cfg["commission_bps"] / 10000.0
    slippage = cfg["slippage_bps"] / 10000.0
    time_stop_bars = cfg.get("time_stop_bars", 10)
    tp_mult = cfg.get("take_profit_mult", 2.0)
    risk_pct = float(cfg.get("risk_per_trade", 0.01))

    idx_arr = df.index.to_list()
    close_arr = pd.to_numeric(df["Close"], errors="coerce").astype(float).values
    open_arr = pd.to_numeric(df["Open"], errors="coerce").astype(float).fillna(pd.Series(close_arr, index=df.index)).values
    high_arr = pd.to_numeric(df["High"], errors="coerce").astype(float).fillna(pd.Series(close_arr, index=df.index)).values
    low_arr = pd.to_numeric(df["Low"], errors="coerce").astype(float).fillna(pd.Series(close_arr, index=df.index)).values
    atr_arr = pd.to_numeric(df.get("ATR", pd.Series(np.nan, index=df.index)), errors="coerce").astype(float).values
    vol_ratio_arr = pd.to_numeric(df.get("VOL_RATIO", pd.Series(1.0, index=df.index)), errors="coerce").fillna(1.0).astype(float).values
    kangaroo_arr = pd.to_numeric(df.get("KANGAROO_BULL", pd.Series(0, index=df.index)), errors="coerce").fillna(0).astype(int).values

    for i in range(len(df)):
        date = idx_arr[i]
        close_px = float(close_arr[i]) if np.isfinite(close_arr[i]) else np.nan
        if not np.isfinite(close_px) or close_px <= 0:
            equity_curve.append((date, cash))
            continue

        open_px = float(open_arr[i]) if np.isfinite(open_arr[i]) else close_px
        high_px = float(high_arr[i]) if np.isfinite(high_arr[i]) else close_px
        low_px = float(low_arr[i]) if np.isfinite(low_arr[i]) else close_px
        atrv = float(atr_arr[i]) if np.isfinite(atr_arr[i]) else np.nan

        atr_pct = float(atrv / close_px) if np.isfinite(atrv) and close_px > 0 else 0.0
        vol_ratio = float(vol_ratio_arr[i]) if np.isfinite(vol_ratio_arr[i]) else 1.0
        dynamic_penalty = max(0.0, min(0.004, atr_pct * 0.25 + max(0.0, (1.0 - vol_ratio)) * 0.001))
        eff_slippage = slippage + dynamic_penalty

        entered_this_bar = False

        # 1) Var olan pozisyon için trailing stop güncellemesi. Girişten önceki pozisyonlara uygulanır.
        if shares > 0 and np.isfinite(atrv) and atrv > 0:
            new_stop = close_px - cfg["atr_stop_mult"] * atrv
            stop = max(stop, new_stop) if pd.notna(stop) else new_stop

        # 2) Pozisyon yoksa, bir önceki bar sinyaline göre bu bar açılışında giriş.
        if shares == 0 and entry_sig[i] == 1:
            if np.isfinite(atrv) and atrv > 0:
                risk_amount = cash * risk_pct
                exec_entry_px = open_px * (1 + eff_slippage)

                is_kangaroo = int(kangaroo_arr[i]) == 1
                if is_kangaroo:
                    stop_price = low_px - (0.5 * atrv)
                    stop_dist = exec_entry_px - stop_price
                    stop_type = "KANGAROO_ATR"
                else:
                    stop_dist = cfg["atr_stop_mult"] * atrv
                    stop_price = exec_entry_px - stop_dist
                    stop_type = "ATR"

                if stop_dist > 0:
                    potential_shares = risk_amount / stop_dist
                    max_shares = cash / (exec_entry_px * (1 + commission))
                    shares_to_buy = min(potential_shares, max_shares)

                    if shares_to_buy > 0.001:
                        shares = shares_to_buy
                        entry_price = exec_entry_px
                        fee = (shares * entry_price) * commission
                        cash -= ((shares * entry_price) + fee)

                        stop = stop_price
                        target_price = entry_price + (tp_mult * stop_dist)
                        bars_held = 0
                        half_sold = False
                        entered_this_bar = True

                        trades.append({
                            "entry_date": date,
                            "entry_price": entry_price,
                            "stop_type": stop_type,
                            "equity_before": cash + (shares * close_px),
                        })

        # 3) Aynı barda yeni girilen işlem için stop/target çalıştırılmaz.
        # Bu, sinyal sonrası next-open işlem varsayımında intrabar look-ahead/iyimserliği azaltır.
        if shares > 0 and not entered_this_bar:
            bars_held += 1

            stop_hit = pd.notna(stop) and (low_px <= stop)
            target_hit = (not half_sold) and pd.notna(target_price) and (high_px >= target_price)
            time_stop_hit = (bars_held >= time_stop_bars) and (close_px < entry_price)

            # Aynı barda hem stop hem target varsa muhafazakâr kural: long için stop önce kabul edilir.
            if stop_hit:
                sell_price = float(stop) * (1 - eff_slippage)
                reason = "STOP"
                gross = shares * sell_price
                fee = gross * commission
                cash += (gross - fee)

                trades[-1]["exit_date"] = date
                trades[-1]["exit_price"] = sell_price
                trades[-1]["exit_reason"] = reason
                trades[-1]["pnl"] = cash - trades[-1]["equity_before"]

                shares = 0.0
                stop = np.nan
                entry_price = np.nan
                target_price = np.nan
                bars_held = 0
                half_sold = False

            else:
                if target_hit:
                    sell_shares = shares * 0.5
                    sell_price = float(target_price) * (1 - eff_slippage)
                    gross = sell_shares * sell_price
                    fee = gross * commission
                    cash += (gross - fee)
                    shares -= sell_shares
                    half_sold = True
                    stop = max(stop, entry_price)

                    if len(trades) > 0:
                        trades[-1]["pnl"] = cash + (shares * close_px * (1 - eff_slippage)) - trades[-1]["equity_before"]

                if exit_sig[i] == 1 or time_stop_hit:
                    sell_price = open_px * (1 - eff_slippage)
                    reason = "TIME_STOP" if time_stop_hit else "RULE_EXIT"

                    gross = shares * sell_price
                    fee = gross * commission
                    cash += (gross - fee)

                    trades[-1]["exit_date"] = date
                    trades[-1]["exit_price"] = sell_price
                    trades[-1]["exit_reason"] = reason
                    trades[-1]["pnl"] = cash - trades[-1]["equity_before"]

                    shares = 0.0
                    stop = np.nan
                    entry_price = np.nan
                    target_price = np.nan
                    bars_held = 0
                    half_sold = False

        position_value = shares * close_px * (1 - eff_slippage)
        equity = cash + position_value
        equity_curve.append((date, equity))

    eq = pd.Series([v for _, v in equity_curve], index=[d for d, _ in equity_curve], name="equity").astype(float)
    eq = eq.replace([np.inf, -np.inf], np.nan).dropna()

    ret = eq.pct_change().dropna()
    total_return = (eq.iloc[-1] / eq.iloc[0] - 1) if len(eq) > 1 else 0.0
    ann_return = (1 + total_return) ** (252 / max(1, len(ret))) - 1 if len(ret) > 0 else 0.0
    ann_vol = float(ret.std() * np.sqrt(252)) if len(ret) > 1 else 0.0

    rf_daily = (1 + float(risk_free_annual)) ** (1 / 252) - 1
    excess = ret - rf_daily

    sharpe = float((excess.mean() * 252) / (excess.std() * np.sqrt(252))) if len(ret) > 1 and excess.std() > 0 else 0.0
    downside = excess.copy()
    downside[downside > 0] = 0
    downside_dev = float(np.sqrt((downside**2).mean()) * np.sqrt(252)) if len(downside) > 1 else 0.0
    sortino = float((excess.mean() * 252) / downside_dev) if downside_dev > 0 else 0.0

    mdd = max_drawdown(eq)
    calmar = float(ann_return / abs(mdd)) if mdd < 0 else 0.0

    if benchmark_returns is not None:
        common_dates = ret.index.intersection(benchmark_returns.index)
        if len(common_dates) > 5:
            r_aligned = ret.loc[common_dates]
            b_aligned = benchmark_returns.loc[common_dates]
            cov = np.cov(r_aligned, b_aligned)[0, 1]
            var_b = np.var(b_aligned)
            beta = cov / var_b if var_b != 0 else 1.0
            mean_r = r_aligned.mean() * 252
            mean_b = b_aligned.mean() * 252
            alpha = (mean_r - risk_free_annual) - beta * (mean_b - risk_free_annual)
            diff = r_aligned - b_aligned
            info_ratio = (diff.mean() * 252) / (diff.std() * np.sqrt(252)) if diff.std() > 0 else 0.0
        else:
            beta = 1.0
            alpha = 0.0
            info_ratio = 0.0
    else:
        beta = 1.0
        alpha = 0.0
        info_ratio = 0.0

    peak = eq.cummax()
    drawdown_pct = (eq - peak) / peak.replace(0, np.nan)
    ulcer_index = np.sqrt((drawdown_pct.dropna()**2).mean()) if len(drawdown_pct.dropna()) > 0 else 0.0

    tdf = pd.DataFrame(trades)
    if not tdf.empty:
        if "pnl" not in tdf.columns:
            tdf["pnl"] = np.nan
        if "exit_date" not in tdf.columns:
            tdf["exit_date"] = pd.NaT
        tdf["pnl"] = tdf["pnl"].astype(float)
        tdf["return_%"] = (tdf["pnl"] / tdf["equity_before"]) * 100
        tdf["holding_days"] = (pd.to_datetime(tdf["exit_date"]) - pd.to_datetime(tdf["entry_date"])).dt.days

    profit_factor = 0.0
    if not tdf.empty and "pnl" in tdf.columns:
        gross_profit = float(tdf.loc[tdf["pnl"] > 0, "pnl"].sum())
        gross_loss = float(-tdf.loc[tdf["pnl"] < 0, "pnl"].sum())
        if gross_loss > 0:
            profit_factor = gross_profit / gross_loss
        elif gross_profit > 0 and gross_loss == 0:
            profit_factor = float("inf")
        else:
            profit_factor = 0.0

    if not tdf.empty and len(tdf) >= 20 and "pnl" in tdf.columns:
        win_rate = (tdf["pnl"] > 0).mean()
        avg_win = tdf.loc[tdf["pnl"] > 0, "pnl"].mean() if win_rate > 0 else 0
        avg_loss = -tdf.loc[tdf["pnl"] < 0, "pnl"].mean() if win_rate < 1 else 0
        if avg_loss > 0 and win_rate > 0 and win_rate < 1:
            b = avg_win / avg_loss
            p = win_rate
            q = 1 - p
            kelly = (p * b - q) / b
            kelly = max(0, min(kelly, 0.10))
        else:
            kelly = 0.0
    else:
        kelly = 0.0

    metrics = {
        "Total Return": float(total_return),
        "Annualized Return": float(ann_return),
        "Annualized Volatility": float(ann_vol),
        "Sharpe": float(sharpe),
        "Sortino": float(sortino),
        "Calmar": float(calmar),
        "Max Drawdown": float(mdd),
        "Trades": int(len(tdf)) if not tdf.empty else 0,
        "Win Rate": float((tdf["pnl"] > 0).mean()) if not tdf.empty and "pnl" in tdf.columns else 0.0,
        "Profit Factor": float(profit_factor) if np.isfinite(profit_factor) else float("inf"),
        "Beta": float(beta),
        "Alpha": float(alpha),
        "Information Ratio": float(info_ratio),
        "Ulcer Index": float(ulcer_index),
        "Kelly % (öneri)": float(kelly * 100),
        "Execution Mode": "next_open_conservative_intrabar",
    }
    return eq, tdf, metrics


def fundamental_score_row(row: dict, mode: str, thresholds: dict) -> Tuple[float, dict, bool]:
    b = {}

    def ok(name, cond, weight, available: bool):
        b[name] = {
            "ok": bool(cond) if available else False,
            "weight": weight,
            "available": bool(available),
        }
        return (weight if (available and cond) else 0.0), (weight if available else 0.0), (1 if available else 0)

    score = 0.0
    total_w = 0.0
    avail_cnt = 0
    ok_cnt = 0

    def A(x):
        return pd.notna(x)

    if mode == "Quality":
        s, tw, ac = ok("ROE", A(row["returnOnEquity"]) and row["returnOnEquity"] >= thresholds["roe"], 20, A(row["returnOnEquity"]))
        score += s
        total_w += tw
        avail_cnt += ac
        ok_cnt += (1 if (A(row["returnOnEquity"]) and row["returnOnEquity"] >= thresholds["roe"]) else 0)

        s, tw, ac = ok("Op Margin", A(row["operatingMargins"]) and row["operatingMargins"] >= thresholds["op_margin"], 15, A(row["operatingMargins"]))
        score += s
        total_w += tw
        avail_cnt += ac
        ok_cnt += (1 if (A(row["operatingMargins"]) and row["operatingMargins"] >= thresholds["op_margin"]) else 0)

        s, tw, ac = ok("Debt/Equity", A(row["debtToEquity"]) and row["debtToEquity"] <= thresholds["dte"], 20, A(row["debtToEquity"]))
        score += s
        total_w += tw
        avail_cnt += ac
        ok_cnt += (1 if (A(row["debtToEquity"]) and row["debtToEquity"] <= thresholds["dte"]) else 0)

        s, tw, ac = ok("Profit Margin", A(row["profitMargins"]) and row["profitMargins"] >= thresholds["profit_margin"], 15, A(row["profitMargins"]))
        score += s
        total_w += tw
        avail_cnt += ac
        ok_cnt += (1 if (A(row["profitMargins"]) and row["profitMargins"] >= thresholds["profit_margin"]) else 0)

        s, tw, ac = ok("FCF", A(row["freeCashflow"]) and row["freeCashflow"] > 0, 30, A(row["freeCashflow"]))
        score += s
        total_w += tw
        avail_cnt += ac
        ok_cnt += (1 if (A(row["freeCashflow"]) and row["freeCashflow"] > 0) else 0)

    elif mode == "Value":
        s, tw, ac = ok("Forward P/E", A(row["forwardPE"]) and row["forwardPE"] <= thresholds["fpe"], 30, A(row["forwardPE"]))
        score += s
        total_w += tw
        avail_cnt += ac
        ok_cnt += (1 if (A(row["forwardPE"]) and row["forwardPE"] <= thresholds["fpe"]) else 0)

        s, tw, ac = ok("PEG", A(row["pegRatio"]) and row["pegRatio"] <= thresholds["peg"], 20, A(row["pegRatio"]))
        score += s
        total_w += tw
        avail_cnt += ac
        ok_cnt += (1 if (A(row["pegRatio"]) and row["pegRatio"] <= thresholds["peg"]) else 0)

        s, tw, ac = ok(
            "P/S",
            A(row["priceToSalesTrailing12Months"]) and row["priceToSalesTrailing12Months"] <= thresholds["ps"],
            20,
            A(row["priceToSalesTrailing12Months"]),
        )
        score += s
        total_w += tw
        avail_cnt += ac
        ok_cnt += (1 if (A(row["priceToSalesTrailing12Months"]) and row["priceToSalesTrailing12Months"] <= thresholds["ps"]) else 0)

        s, tw, ac = ok("P/B", A(row["priceToBook"]) and row["priceToBook"] <= thresholds["pb"], 15, A(row["priceToBook"]))
        score += s
        total_w += tw
        avail_cnt += ac
        ok_cnt += (1 if (A(row["priceToBook"]) and row["priceToBook"] <= thresholds["pb"]) else 0)

        s, tw, ac = ok("ROE", A(row["returnOnEquity"]) and row["returnOnEquity"] >= thresholds["roe"], 15, A(row["returnOnEquity"]))
        score += s
        total_w += tw
        avail_cnt += ac
        ok_cnt += (1 if (A(row["returnOnEquity"]) and row["returnOnEquity"] >= thresholds["roe"]) else 0)

    else:  # Growth
        s, tw, ac = ok("Revenue Growth", A(row["revenueGrowth"]) and row["revenueGrowth"] >= thresholds["rev_g"], 35, A(row["revenueGrowth"]))
        score += s
        total_w += tw
        avail_cnt += ac
        ok_cnt += (1 if (A(row["revenueGrowth"]) and row["revenueGrowth"] >= thresholds["rev_g"]) else 0)

        s, tw, ac = ok("Earnings Growth", A(row["earningsGrowth"]) and row["earningsGrowth"] >= thresholds["earn_g"], 35, A(row["earningsGrowth"]))
        score += s
        total_w += tw
        avail_cnt += ac
        ok_cnt += (1 if (A(row["earningsGrowth"]) and row["earningsGrowth"] >= thresholds["earn_g"]) else 0)

        s, tw, ac = ok("Op Margin", A(row["operatingMargins"]) and row["operatingMargins"] >= thresholds["op_margin"], 15, A(row["operatingMargins"]))
        score += s
        total_w += tw
        avail_cnt += ac
        ok_cnt += (1 if (A(row["operatingMargins"]) and row["operatingMargins"] >= thresholds["op_margin"]) else 0)

        s, tw, ac = ok("Debt/Equity", A(row["debtToEquity"]) and row["debtToEquity"] <= thresholds["dte"], 15, A(row["debtToEquity"]))
        score += s
        total_w += tw
        avail_cnt += ac
        ok_cnt += (1 if (A(row["debtToEquity"]) and row["debtToEquity"] <= thresholds["dte"]) else 0)

    score_pct = (score / total_w) * 100 if total_w > 0 else 0.0
    min_coverage = int(thresholds.get("min_coverage", 3))
    min_ok = int(thresholds["min_ok"])
    pass_bool = (score_pct >= thresholds["min_score"]) and (ok_cnt >= min_ok) and (avail_cnt >= min_coverage)
    return float(score_pct), b, bool(pass_bool)


# =============================
# Target price band / SR Levels (GÜÇLÜ S/R FİLTRESİ)
# =============================
def _swing_points(high: pd.Series, low: pd.Series, left: int = 2, right: int = 2):
    hs = []
    ls = []
    n = len(high)
    for i in range(left, n - right):
        hwin = high.iloc[i - left : i + right + 1]
        lwin = low.iloc[i - left : i + right + 1]
        if high.iloc[i] == hwin.max():
            hs.append((high.index[i], float(high.iloc[i])))
        if low.iloc[i] == lwin.min():
            ls.append((low.index[i], float(low.iloc[i])))
    return hs, ls



@st.cache_data(ttl=PRICE_CACHE_TTL_SECONDS, show_spinner=False)
def _analyze_sr_levels_cached(df_in: pd.DataFrame, lookback: int = 200, tol: float = 0.02, cache_version: str = SR_CACHE_VERSION) -> List[dict]:
    df = df_in.copy()
    h = df["High"].tail(lookback).dropna()
    l = df["Low"].tail(lookback).dropna()
    c = df["Close"].tail(lookback).dropna()
    if len(c) < 10:
        return []

    v = df["Volume"].tail(lookback) if "Volume" in df.columns else pd.Series(dtype=float)
    atr_ref = atr(df["High"], df["Low"], df["Close"], 14).tail(lookback)
    atr_pct_med = float((atr_ref / c.reindex(atr_ref.index)).replace([np.inf, -np.inf], np.nan).median()) if len(atr_ref) > 0 else np.nan
    adaptive_tol = float(np.clip(np.nanmedian([tol, (atr_pct_med * 1.2) if pd.notna(atr_pct_med) else tol]), 0.008, 0.04))

    hs, ls = _swing_points(h, l, left=3, right=3)
    raw_levels = [val for _, val in hs] + [val for _, val in ls]
    raw_levels += [float(c.tail(20).max()), float(c.tail(20).min())]
    raw_levels = [float(x) for x in raw_levels if np.isfinite(x)]
    if not raw_levels:
        return []

    center_sources: List[Tuple[float, str, float]] = []

    # 1) 1D KMeans: doğal kümelenme yakalama
    try:
        arr = np.array(raw_levels, dtype=float).reshape(-1, 1)
        unique_count = len(np.unique(arr.flatten()))
        if unique_count >= 3:
            n_clusters = int(min(max(3, round(np.sqrt(len(raw_levels)))), 8, unique_count))
            km = KMeans(n_clusters=n_clusters, n_init=10, random_state=42)
            labels = km.fit_predict(arr)
            for k, center in enumerate(km.cluster_centers_.flatten()):
                count = int(np.sum(labels == k))
                if np.isfinite(center):
                    center_sources.append((float(center), "kmeans", min(1.0, count / max(1, len(raw_levels)))))
    except Exception as e:
        log_app_error("analyze_sr_levels.kmeans", e, {"raw_levels": len(raw_levels)}, level="DEBUG")

    # 2) Histogram peak yaklaşımı
    try:
        bins = int(min(max(8, round(np.sqrt(len(raw_levels)) * 2)), 24))
        hist, edges = np.histogram(np.array(raw_levels, dtype=float), bins=bins)
        max_hist = max(1, int(np.max(hist))) if len(hist) else 1
        for i, val in enumerate(hist):
            if val <= 0:
                continue
            left_ok = (i == 0) or (hist[i] >= hist[i - 1])
            right_ok = (i == len(hist) - 1) or (hist[i] >= hist[i + 1])
            if left_ok and right_ok:
                center = float((edges[i] + edges[i + 1]) / 2.0)
                if np.isfinite(center):
                    center_sources.append((center, "histogram", float(val) / max_hist))
    except Exception as e:
        log_app_error("analyze_sr_levels.histogram", e, {"raw_levels": len(raw_levels)}, level="DEBUG")

    # 3) Fallback: tolerans bazlı eski mantık, ama dinamik merkezli
    adaptive_clusters = []
    for rl in sorted(set(round(float(x), 2) for x in raw_levels)):
        placed = False
        for cl in adaptive_clusters:
            if cl["center"] != 0 and abs(rl - cl["center"]) / abs(cl["center"]) <= adaptive_tol:
                cl["points"].append(float(rl))
                cl["center"] = float(np.mean(cl["points"]))
                placed = True
                break
        if not placed:
            adaptive_clusters.append({"center": float(rl), "points": [float(rl)]})
    for cl in adaptive_clusters:
        center_sources.append((float(cl["center"]), "adaptive", min(1.0, len(cl["points"]) / max(1, len(raw_levels)))))

    if not center_sources:
        return []

    # 4) Kaynak ağırlıklı merge: KMeans + histogram + adaptive merkezleri tek normalize skorla birleştir.
    source_weight = {"kmeans": 1.05, "histogram": 1.00, "adaptive": 0.90}
    center_sources = sorted([(p, s, w) for p, s, w in center_sources if np.isfinite(p)], key=lambda x: x[0])
    merged: List[Dict[str, Any]] = []
    for price, source, local_weight in center_sources:
        if not merged:
            merged.append({"points": [price], "weights": [source_weight.get(source, 1.0) * max(0.05, local_weight)], "sources": {source}})
            continue
        ref = float(np.average(merged[-1]["points"], weights=merged[-1]["weights"]))
        if ref != 0 and abs(price - ref) / abs(ref) <= (adaptive_tol * 0.65):
            merged[-1]["points"].append(price)
            merged[-1]["weights"].append(source_weight.get(source, 1.0) * max(0.05, local_weight))
            merged[-1]["sources"].add(source)
        else:
            merged.append({"points": [price], "weights": [source_weight.get(source, 1.0) * max(0.05, local_weight)], "sources": {source}})

    cluster_centers = []
    center_meta = {}
    for m in merged:
        center = float(np.average(m["points"], weights=m["weights"]))
        center_round = round(center, 2)
        cluster_centers.append(center_round)
        center_meta[center_round] = {
            "source_count": len(m["sources"]),
            "source_weight": float(np.sum(m["weights"])),
            "sources": ",".join(sorted(m["sources"])),
        }

    cluster_centers = sorted(set(x for x in cluster_centers if np.isfinite(x)))
    if not cluster_centers:
        return []

    avg_vol_normal = float(v.mean()) if not v.empty else 1.0
    if avg_vol_normal <= 0:
        avg_vol_normal = 1.0

    details = []
    df_lookback = df.tail(lookback)

    for level_px in cluster_centers:
        lower_bound = level_px * (1 - adaptive_tol / 2)
        upper_bound = level_px * (1 + adaptive_tol / 2)
        touches = df_lookback[(df_lookback["High"] >= lower_bound) & (df_lookback["Low"] <= upper_bound)]
        num_touches = len(touches)
        if num_touches == 0:
            continue

        first_touch_idx = touches.index[0]
        first_idx_num = df_lookback.index.get_loc(first_touch_idx)
        duration_bars = len(df_lookback) - first_idx_num

        if "Volume" in df_lookback.columns and not touches.empty:
            vol_at_level = float(touches["Volume"].mean())
        else:
            vol_at_level = avg_vol_normal

        vol_diff_pct = (vol_at_level / avg_vol_normal - 1.0) * 100.0
        meta = center_meta.get(level_px, {"source_count": 1, "source_weight": 1.0, "sources": ""})
        score_touches = min(num_touches * 10, 40)
        score_vol = min(max(vol_diff_pct / 2.0, 0), 30)
        score_dur = min(duration_bars / 2.0, 20)
        score_sources = min(meta.get("source_count", 1) * 4.0 + meta.get("source_weight", 1.0) * 2.0, 12)
        strength_pct = min(score_touches + score_vol + score_dur + score_sources, 99.0)

        details.append({
            "price": round(level_px, 2),
            "duration_bars": int(duration_bars),
            "vol_at_level": float(vol_at_level),
            "vol_diff_pct": float(vol_diff_pct),
            "strength_pct": float(strength_pct),
            "touches": int(num_touches),
            "source_count": int(meta.get("source_count", 1)),
            "sources": str(meta.get("sources", "")),
        })

    return sorted(details, key=lambda x: x["price"])


def analyze_sr_levels(df: pd.DataFrame, lookback: int = 200, tol=0.02) -> List[dict]:
    try:
        if df is None or df.empty or not {"High", "Low", "Close"}.issubset(df.columns):
            return []
        return _analyze_sr_levels_cached(df.tail(int(lookback)).copy(), int(lookback), float(tol), SR_CACHE_VERSION)
    except Exception as e:
        log_app_error("analyze_sr_levels", e, {"rows": len(df) if df is not None else 0, "lookback": lookback})
        return []


def target_price_band(df: pd.DataFrame):
    last = df.iloc[-1]
    px_close = float(last["Close"])
    atrv = float(last["ATR"]) if pd.notna(last.get("ATR", np.nan)) else np.nan

    lv_details = analyze_sr_levels(df)

    if not np.isfinite(atrv) or atrv <= 0:
        return {"base": px_close, "bull": None, "bear": None, "levels": lv_details, "r1_dict": None, "s1_dict": None}

    bull1 = px_close + 1.5 * atrv
    bull2 = px_close + 3.0 * atrv
    bear1 = px_close - 1.5 * atrv
    bear2 = px_close - 3.0 * atrv

    above = [x for x in lv_details if x["price"] >= px_close * 1.005]
    below = [x for x in lv_details if x["price"] <= px_close * 0.995]

    valid_above = [x for x in above if x["duration_bars"] >= 10 and x["touches"] >= 2]
    valid_below = [x for x in below if x["duration_bars"] >= 10 and x["touches"] >= 2]

    # Simetrik ve açıklanabilir eşik: destek/direnç için aynı minimum güç.
    # Ayrı eşik gerekiyorsa ileride cfg üzerinden yönetilecek şekilde tek sabite indirildi.
    sr_strength_min = 30.0
    strong_above = [x for x in valid_above if x["strength_pct"] >= sr_strength_min]
    strong_below = [x for x in valid_below if x["strength_pct"] >= sr_strength_min]

    r1_dict = min(strong_above, key=lambda x: x["price"]) if strong_above else (min(valid_above, key=lambda x: x["price"]) if valid_above else None)
    s1_dict = max(strong_below, key=lambda x: x["price"]) if strong_below else (max(valid_below, key=lambda x: x["price"]) if valid_below else None)

    r1 = r1_dict["price"] if r1_dict else None
    s1 = s1_dict["price"] if s1_dict else None

    if r1 is None:
        pivot = (float(last["High"]) + float(last["Low"]) + px_close) / 3.0
        synth_r1 = (2 * pivot) - float(last["Low"])
        if synth_r1 > px_close:
            r1 = synth_r1
            r1_dict = {"price": synth_r1, "duration_bars": 0, "vol_diff_pct": 0, "strength_pct": 100, "is_synthetic": True}

    if s1 is None:
        pivot = (float(last["High"]) + float(last["Low"]) + px_close) / 3.0
        synth_s1 = (2 * pivot) - float(last["High"])
        if synth_s1 < px_close and synth_s1 > 0:
            s1 = synth_s1
            s1_dict = {"price": synth_s1, "duration_bars": 0, "vol_diff_pct": 0, "strength_pct": 100, "is_synthetic": True}

    return {
        "base": px_close,
        "bull": (bull1, bull2, r1),
        "bear": (bear1, bear2, s1),
        "levels": lv_details,
        "r1_dict": r1_dict,
        "s1_dict": s1_dict,
        "sr_strength_min": sr_strength_min,
    }



# =============================
# Volume Profile / VPVR Helpers
# =============================
def compute_volume_profile(df: pd.DataFrame, bins: int = 24, lookback: int = 220) -> Tuple[pd.DataFrame, float]:
    if df is None or df.empty or not {"High", "Low", "Close", "Volume"}.issubset(df.columns):
        return pd.DataFrame(), np.nan

    try:
        use = df.tail(min(len(df), int(lookback))).copy()
        use = use.dropna(subset=["High", "Low", "Close", "Volume"])
        if use.empty:
            return pd.DataFrame(), np.nan

        high = pd.to_numeric(use["High"], errors="coerce").astype(float)
        low = pd.to_numeric(use["Low"], errors="coerce").astype(float)
        close = pd.to_numeric(use["Close"], errors="coerce").astype(float)
        volumes = pd.to_numeric(use["Volume"], errors="coerce").fillna(0.0).astype(float)

        typical_price = ((high + low + close) / 3.0).replace([np.inf, -np.inf], np.nan)
        mask = typical_price.notna() & volumes.notna() & (volumes >= 0)
        typical_price = typical_price[mask]
        volumes = volumes[mask]

        if typical_price.empty:
            return pd.DataFrame(), np.nan

        price_min = float(low[mask].min())
        price_max = float(high[mask].max())
        if not np.isfinite(price_min) or not np.isfinite(price_max) or price_max <= price_min:
            return pd.DataFrame(), np.nan

        # Dinamik bin: aşırı geniş fiyat aralığında log ölçek, normalde Freedman-Diaconis + sınır.
        requested_bins = max(8, int(bins))
        use_log = price_min > 0 and (price_max / max(price_min, 1e-9)) > 1.8

        if use_log:
            tp_work = np.log(typical_price.values)
            range_work = (np.log(price_min), np.log(price_max))
        else:
            tp_work = typical_price.values
            range_work = (price_min, price_max)

        try:
            q75, q25 = np.nanpercentile(tp_work, [75, 25])
            iqr = q75 - q25
            if np.isfinite(iqr) and iqr > 0:
                bin_width = 2 * iqr / (len(tp_work) ** (1 / 3))
                fd_bins = int(np.ceil((np.nanmax(tp_work) - np.nanmin(tp_work)) / max(bin_width, 1e-12)))
                final_bins = int(np.clip(fd_bins, 12, max(18, requested_bins * 2)))
            else:
                final_bins = requested_bins
        except Exception:
            final_bins = requested_bins

        final_bins = int(np.clip(final_bins, 8, 80))
        hist, edges = np.histogram(tp_work, bins=final_bins, range=range_work, weights=volumes.values)

        if use_log:
            centers = np.exp((edges[:-1] + edges[1:]) / 2.0)
        else:
            centers = (edges[:-1] + edges[1:]) / 2.0

        vp_df = pd.DataFrame({"Price": centers, "Volume": hist.astype(float)})
        vp_df = vp_df[vp_df["Volume"] > 0].copy()
        if vp_df.empty:
            return pd.DataFrame(), np.nan

        poc_price = float(vp_df.loc[vp_df["Volume"].idxmax(), "Price"])
        return vp_df.sort_values("Price"), poc_price
    except Exception as e:
        log_app_error("compute_volume_profile", e, {"rows": len(df) if df is not None else 0, "bins": bins, "lookback": lookback})
        return pd.DataFrame(), np.nan


def build_volume_profile_figure(df: pd.DataFrame, bins: int = 24, lookback: int = 220) -> Tuple[go.Figure, float]:
    vp_df, poc_price = compute_volume_profile(df, bins=bins, lookback=lookback)
    fig = go.Figure()

    if vp_df.empty:
        fig.update_layout(height=360, title="Hacim Profili (VPVR / POC)")
        return fig, np.nan

    fig.add_trace(go.Bar(
        x=vp_df["Volume"],
        y=vp_df["Price"],
        orientation="h",
        name="Volume Profile",
    ))

    if np.isfinite(poc_price):
        fig.add_hline(y=poc_price, line_dash="dash", line_color="red", annotation_text=f"POC: {poc_price:.2f}")

    fig.update_layout(
        height=360,
        title="Hacim Profili (VPVR / POC)",
        xaxis_title="Hacim",
        yaxis_title="Fiyat",
        showlegend=False,
        margin=dict(l=0, r=0, t=40, b=0),
    )
    return fig, poc_price



def _get_line_body_arrays(subset: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
    body_low = subset[["Open", "Close"]].min(axis=1).astype(float) if {"Open", "Close"}.issubset(subset.columns) else subset["Close"].astype(float)
    body_high = subset[["Open", "Close"]].max(axis=1).astype(float) if {"Open", "Close"}.issubset(subset.columns) else subset["Close"].astype(float)
    return body_low, body_high


def _count_line_touches(
    subset: pd.DataFrame,
    pivot_points: List[Tuple[pd.Timestamp, float]],
    slope: float,
    intercept: float,
    touch_tol_pct: float = 0.01,
) -> List[Tuple[pd.Timestamp, float, int, float]]:
    touched = []
    for tp, pp in pivot_points:
        try:
            xp = subset.index.get_loc(tp)
        except Exception:
            continue
        y_hat = float(intercept) + float(slope) * float(xp)
        denom = max(abs(float(pp)), 1e-9)
        if abs(float(pp) - y_hat) / denom <= touch_tol_pct:
            touched.append((tp, float(pp), int(xp), float(y_hat)))
    return touched


def _line_break_stats(
    subset: pd.DataFrame,
    slope: float,
    intercept: float,
    line_kind: str = "support",
    start_x: int = 0,
    break_tol_pct: float = 0.003,
) -> Dict[str, Any]:
    try:
        body_low, body_high = _get_line_body_arrays(subset)
        wick_low = pd.to_numeric(subset["Low"], errors="coerce").astype(float)
        wick_high = pd.to_numeric(subset["High"], errors="coerce").astype(float)

        x_arr = np.arange(len(subset), dtype=float)
        y_line = float(intercept) + float(slope) * x_arr
        seg = slice(max(0, int(start_x)), len(subset))
        tol_arr = np.maximum(np.abs(y_line[seg]), 1e-9) * float(break_tol_pct)

        if line_kind == "support":
            wick_diff = wick_low.iloc[seg].values - y_line[seg]
            body_diff = body_low.iloc[seg].values - y_line[seg]
            wick_violations = wick_diff < (-tol_arr)
            body_violations = body_diff < (-tol_arr)
            primary_diff = wick_diff
        else:
            wick_diff = y_line[seg] - wick_high.iloc[seg].values
            body_diff = y_line[seg] - body_high.iloc[seg].values
            wick_violations = wick_diff < (-tol_arr)
            body_violations = body_diff < (-tol_arr)
            primary_diff = wick_diff

        return {
            "violation_count": int(np.sum(wick_violations)),
            "body_violation_count": int(np.sum(body_violations)),
            "wick_violation_count": int(np.sum(wick_violations)),
            "worst_break": float(np.nanmin(primary_diff)) if len(primary_diff) > 0 else 0.0,
        }
    except Exception as e:
        log_app_error("_line_break_stats", e, {"line_kind": line_kind, "rows": len(subset) if subset is not None else 0})
        return {"violation_count": 0, "body_violation_count": 0, "wick_violation_count": 0, "worst_break": 0.0}


def _project_future_index(index: pd.Index, bars_ahead: int = 8) -> Tuple[Any, float]:
    if len(index) == 0:
        return None, float(bars_ahead)

    future_x = float(max(0, len(index) - 1) + int(bars_ahead))

    try:
        if isinstance(index, pd.DatetimeIndex) and len(index) >= 2:
            idx = pd.DatetimeIndex(index)
            last = idx[-1]
            diffs = idx.to_series().diff().dropna()
            median_step = diffs.median() if not diffs.empty else pd.Timedelta(days=1)

            if pd.isna(median_step) or median_step <= pd.Timedelta(0):
                median_step = pd.Timedelta(days=1)

            # Günlük veride takvim günü yerine iş günü projeksiyonu kullan.
            if median_step >= pd.Timedelta(hours=20) and median_step <= pd.Timedelta(days=3):
                future_date = last + pd.offsets.BDay(int(bars_ahead))
            elif median_step >= pd.Timedelta(days=4):
                future_date = last + pd.DateOffset(weeks=int(bars_ahead))
            else:
                future_date = last + (median_step * int(bars_ahead))

            return future_date, future_x
    except Exception as e:
        log_app_error("_project_future_index", e, {"bars_ahead": bars_ahead, "index_type": str(type(index))})

    return index[-1], future_x


def _adaptive_trendline_params(subset: pd.DataFrame, line_kind: str = "support") -> Dict[str, Any]:
    if subset is None or subset.empty:
        return {"touch_tol_pct": 0.012, "break_tol_pct": 0.004, "max_pivots": 7, "min_required_touches": 3}

    atr_pct_series = (subset.get("ATR", pd.Series(index=subset.index, dtype=float)) / subset["Close"]).replace([np.inf, -np.inf], np.nan)
    atr_pct_med = float(atr_pct_series.median()) if atr_pct_series.notna().any() else 0.02
    base_touch = float(np.clip(max(0.008, atr_pct_med * 1.2), 0.008, 0.02))
    base_break = float(np.clip(base_touch * 0.35, 0.002, 0.008))

    min_touches = 3
    if len(subset) < 80 or atr_pct_med > 0.04:
        min_touches = 2

    max_pivots = int(min(max(6, round(len(subset) / 20)), 10))
    return {
        "touch_tol_pct": base_touch,
        "break_tol_pct": base_break,
        "max_pivots": max_pivots,
        "min_required_touches": min_touches,
    }



def _evaluate_line_candidate(
    subset: pd.DataFrame,
    pivot_points: List[Tuple[pd.Timestamp, float]],
    slope: float,
    intercept: float,
    line_kind: str,
    touch_tol_pct: float,
    break_tol_pct: float,
    source: str,
    anchor_t1: Optional[pd.Timestamp] = None,
    anchor_p1: Optional[float] = None,
    min_required_touches: int = 3,
    base_r2: Optional[float] = None,
    extra_score: float = 0.0,
) -> Optional[Dict[str, Any]]:
    if subset is None or subset.empty or len(pivot_points) < 2:
        return None

    try:
        touched = _count_line_touches(subset, pivot_points, slope, intercept, touch_tol_pct=touch_tol_pct)
        adaptive_min_touches = min_required_touches
        if source in {"linregress", "ransac"} and len(subset) >= 18:
            adaptive_min_touches = min(adaptive_min_touches, 2)

        if len(touched) < max(2, adaptive_min_touches):
            return None

        touched_x = np.array([x for _, _, x, _ in touched], dtype=float)
        touched_y = np.array([p for _, p, _, _ in touched], dtype=float)
        if len(np.unique(touched_x)) < 2:
            return None

        y_fit_touch = float(intercept) + float(slope) * touched_x
        ss_res_touch = float(np.sum((touched_y - y_fit_touch) ** 2))
        ss_tot_touch = float(np.sum((touched_y - np.mean(touched_y)) ** 2))
        touch_r2 = 1.0 if ss_tot_touch <= 1e-12 else max(0.0, 1.0 - (ss_res_touch / ss_tot_touch))

        start_seg = int(np.min(touched_x))
        end_seg = int(np.max(touched_x))
        x_seg = np.arange(start_seg, end_seg + 1, dtype=float)
        y_line_seg = float(intercept) + float(slope) * x_seg

        wick_low = pd.to_numeric(subset["Low"], errors="coerce").astype(float)
        wick_high = pd.to_numeric(subset["High"], errors="coerce").astype(float)
        ref_seg = wick_low.iloc[start_seg:end_seg + 1].values if line_kind == "support" else wick_high.iloc[start_seg:end_seg + 1].values

        if len(ref_seg) > 1 and np.isfinite(ref_seg).all():
            ss_res_seg = float(np.sum((ref_seg - y_line_seg) ** 2))
            ss_tot_seg = float(np.sum((ref_seg - np.mean(ref_seg)) ** 2))
            segment_r2 = 1.0 if ss_tot_seg <= 1e-12 else max(0.0, 1.0 - (ss_res_seg / ss_tot_seg))
            ref_scale = max(float(np.nanmax(ref_seg) - np.nanmin(ref_seg)), float(np.nanmean(np.abs(ref_seg))) * 0.01, 1e-9)
            mean_dist = float(np.mean(np.abs(ref_seg - y_line_seg)))
            segment_fit = max(0.0, 1.0 - (mean_dist / ref_scale))
        else:
            segment_r2 = 0.0
            segment_fit = 0.0

        base_r2_clean = float(base_r2) if base_r2 is not None and np.isfinite(base_r2) else np.nan
        r2_components = [touch_r2, segment_r2]
        if np.isfinite(base_r2_clean):
            r2_components.append(base_r2_clean)
        r2_final = float(np.nanmean(r2_components)) if r2_components else 0.0

        # R² ve segment_fit aynı ölçek gibi max() ile kıyaslanmaz; ayrı eşik + birleşik kalite skoru kullanılır.
        quality_score = (
            0.40 * max(0.0, min(1.0, touch_r2))
            + 0.25 * max(0.0, min(1.0, segment_r2))
            + 0.25 * max(0.0, min(1.0, segment_fit))
            + 0.10 * (max(0.0, min(1.0, base_r2_clean)) if np.isfinite(base_r2_clean) else 0.5)
        )

        if source in {"linregress", "ransac"}:
            if quality_score < 0.58 or (segment_fit < 0.45 and touch_r2 < 0.70):
                return None

        break_stats = _line_break_stats(
            subset,
            slope=slope,
            intercept=intercept,
            line_kind=line_kind,
            start_x=start_seg,
            break_tol_pct=break_tol_pct,
        )
        # Fitil kırılımı stricter. Çok küçük tolerans kaynaklı tek fitil kırılımını, gövde kırılmıyorsa esnet.
        if break_stats["wick_violation_count"] > 1 or (break_stats["wick_violation_count"] == 1 and break_stats.get("body_violation_count", 0) > 0):
            return None

        span_ratio = (int(np.max(touched_x)) - int(np.min(touched_x))) / max(1.0, float(len(subset) - 1))
        recent_bonus = float(np.max(touched_x)) / max(1.0, float(len(subset) - 1))

        score = (
            len(touched) * 70.0
            + quality_score * 75.0
            + span_ratio * 20.0
            + recent_bonus * 8.0
            + float(extra_score)
            - break_stats.get("wick_violation_count", 0) * 10.0
        )

        first_touch = min(touched, key=lambda x: x[2])
        return {
            "x1": int(first_touch[2]),
            "x2": int(np.max(touched_x)),
            "t1": anchor_t1 if anchor_t1 is not None else first_touch[0],
            "t2": max(touched, key=lambda x: x[2])[0],
            "p1": float(anchor_p1) if anchor_p1 is not None else float(first_touch[1]),
            "p2": float(max(touched, key=lambda x: x[2])[1]),
            "slope": float(slope),
            "intercept": float(intercept),
            "touches": int(len(touched)),
            "r2": float(r2_final),
            "segment_fit": float(segment_fit),
            "quality_score": float(quality_score),
            "wick_breaks": int(break_stats.get("wick_violation_count", 0)),
            "body_breaks": int(break_stats.get("body_violation_count", 0)),
            "score": float(score),
            "source": source,
        }
    except Exception as e:
        log_app_error("_evaluate_line_candidate", e, {"source": source, "line_kind": line_kind, "rows": len(subset) if subset is not None else 0})
        return None


def _build_trendline_candidates(
    subset: pd.DataFrame,
    pivot_points: List[Tuple[pd.Timestamp, float]],
    line_kind: str = "support",
    max_pivots: int = 7,
    touch_tol_pct: float = 0.01,
    break_tol_pct: float = 0.003,
) -> Optional[Dict[str, Any]]:
    if subset is None or subset.empty or len(pivot_points) < 2:
        return None

    params = _adaptive_trendline_params(subset, line_kind=line_kind)
    use_touch_tol = float(params["touch_tol_pct"])
    use_break_tol = float(params["break_tol_pct"])
    use_max_pivots = int(max(max_pivots, params["max_pivots"]))
    min_required_touches = int(params["min_required_touches"])

    pivots = pivot_points[-use_max_pivots:]
    if len(pivots) < 2:
        return None

    candidates: List[Dict[str, Any]] = []

    for i in range(len(pivots) - 1):
        for j in range(i + 1, len(pivots)):
            t1, p1 = pivots[i]
            t2, p2 = pivots[j]
            try:
                x1 = subset.index.get_loc(t1)
                x2 = subset.index.get_loc(t2)
            except Exception as e:
                log_app_error("_build_trendline_candidates.loc", e, {"line_kind": line_kind, "t1": str(t1), "t2": str(t2)}, level="DEBUG")
                continue
            if x2 <= x1:
                continue

            slope = (float(p2) - float(p1)) / float(x2 - x1)
            intercept = float(p1) - slope * float(x1)

            cand = _evaluate_line_candidate(
                subset, pivots,
                slope=slope, intercept=intercept, line_kind=line_kind,
                touch_tol_pct=use_touch_tol, break_tol_pct=use_break_tol,
                source="pivot_combo", anchor_t1=t1, anchor_p1=float(p1),
                min_required_touches=min_required_touches,
            )
            if cand is not None:
                candidates.append(cand)

    reg_pivots = pivots[-5:] if len(pivots) >= 3 else pivots
    if len(reg_pivots) >= 3:
        try:
            rx = np.array([float(subset.index.get_loc(t)) for t, _ in reg_pivots], dtype=float)
            ry = np.array([float(p) for _, p in reg_pivots], dtype=float)
            slope_lr, intercept_lr, r_value, _, _ = linregress(rx, ry)
            base_r2 = float((r_value ** 2)) if np.isfinite(r_value) else np.nan
            extra_score = max(0.0, (base_r2 - 0.5) * 20.0) if pd.notna(base_r2) else 0.0
            cand = _evaluate_line_candidate(
                subset, pivots,
                slope=float(slope_lr), intercept=float(intercept_lr), line_kind=line_kind,
                touch_tol_pct=use_touch_tol, break_tol_pct=use_break_tol,
                source="linregress", anchor_t1=reg_pivots[0][0], anchor_p1=float(reg_pivots[0][1]),
                min_required_touches=min_required_touches, base_r2=base_r2, extra_score=extra_score,
            )
            if cand is not None:
                candidates.append(cand)
        except Exception as e:
            log_app_error("_build_trendline_candidates.ransac", e, {"line_kind": line_kind, "reg_pivots": len(reg_pivots)}, level="DEBUG")

        try:
            rx = np.array([float(subset.index.get_loc(t)) for t, _ in reg_pivots], dtype=float).reshape(-1, 1)
            ry = np.array([float(p) for _, p in reg_pivots], dtype=float)
            ransac = RANSACRegressor(random_state=42)
            ransac.fit(rx, ry)
            slope_rc = float(ransac.estimator_.coef_[0]) if hasattr(ransac.estimator_, "coef_") else np.nan
            intercept_rc = float(ransac.estimator_.intercept_) if hasattr(ransac.estimator_, "intercept_") else np.nan
            if np.isfinite(slope_rc) and np.isfinite(intercept_rc):
                base_r2 = float(ransac.score(rx, ry))
                extra_score = max(0.0, (base_r2 - 0.5) * 20.0) if pd.notna(base_r2) else 0.0
                cand = _evaluate_line_candidate(
                    subset, pivots,
                    slope=slope_rc, intercept=intercept_rc, line_kind=line_kind,
                    touch_tol_pct=use_touch_tol, break_tol_pct=use_break_tol,
                    source="ransac", anchor_t1=reg_pivots[0][0], anchor_p1=float(reg_pivots[0][1]),
                    min_required_touches=min_required_touches, base_r2=base_r2, extra_score=extra_score,
                )
                if cand is not None:
                    candidates.append(cand)
        except Exception:
            pass

    if not candidates:
        return None

    candidates = sorted(
        candidates,
        key=lambda x: (x.get("score", 0.0), x.get("touches", 0), x.get("segment_fit", 0.0), x.get("r2", 0.0)),
        reverse=True,
    )
    return candidates[0]




def add_support_resistance_trend_overlays(fig: go.Figure, plot_df: pd.DataFrame, lookback: int = 220) -> go.Figure:
    if fig is None:
        fig = go.Figure()
    if plot_df is None or plot_df.empty or not {"High", "Low", "Close"}.issubset(plot_df.columns):
        return fig

    use = plot_df.tail(min(len(plot_df), int(lookback))).copy()
    if use.empty:
        return fig

    try:
        base_px = float(use["Close"].iloc[-1])

        sr_levels = analyze_sr_levels(use, lookback=min(len(use), 200), tol=0.02)
        below = [lv for lv in sr_levels if lv["price"] < base_px]
        above = [lv for lv in sr_levels if lv["price"] > base_px]

        near_supports = sorted(below, key=lambda x: abs(base_px - x["price"]))[:2]
        near_resistances = sorted(above, key=lambda x: abs(x["price"] - base_px))[:2]

        for lv in near_supports:
            fig.add_hline(y=float(lv["price"]), line_dash="dot", line_color="green", annotation_text=f"Destek {lv['price']:.2f}", annotation_position="bottom left")
        for lv in near_resistances:
            fig.add_hline(y=float(lv["price"]), line_dash="dot", line_color="red", annotation_text=f"Direnç {lv['price']:.2f}", annotation_position="top left")

        subset = use.tail(min(len(use), 140))
        swing_highs, swing_lows = _swing_points(subset["High"], subset["Low"], left=3, right=3)
        future_date, future_x = _project_future_index(subset.index, bars_ahead=8)

        support_line = _build_trendline_candidates(subset, swing_lows, line_kind="support")
        if support_line is not None and future_date is not None:
            y_future = support_line["intercept"] + support_line["slope"] * float(future_x)
            fig.add_trace(go.Scatter(
                x=[support_line["t1"], future_date],
                y=[support_line["p1"], float(y_future)],
                mode="lines",
                name=f"Destek Trendi ({support_line.get('source','')})",
                line=dict(color="green", dash="dash", width=3 if support_line["touches"] >= 3 else 2),
            ))

        resistance_line = _build_trendline_candidates(subset, swing_highs, line_kind="resistance")
        if resistance_line is not None and future_date is not None:
            y_future = resistance_line["intercept"] + resistance_line["slope"] * float(future_x)
            fig.add_trace(go.Scatter(
                x=[resistance_line["t1"], future_date],
                y=[resistance_line["p1"], float(y_future)],
                mode="lines",
                name=f"Direnç Trendi ({resistance_line.get('source','')})",
                line=dict(color="red", dash="dash", width=3 if resistance_line["touches"] >= 3 else 2),
            ))
    except Exception as e:
        log_app_error("add_support_resistance_trend_overlays", e, {"rows": len(plot_df) if plot_df is not None else 0, "lookback": lookback})

    return fig


def apply_live_last_override_to_df(df: pd.DataFrame, live_price: float) -> pd.DataFrame:
    if df is None or df.empty or not np.isfinite(live_price):
        return df

    out = df.copy()
    last_idx = out.index[-1]

    try:
        if "Close" in out.columns:
            out.at[last_idx, "Close"] = float(live_price)
        if "High" in out.columns and pd.notna(out.at[last_idx, "High"]):
            out.at[last_idx, "High"] = max(float(out.at[last_idx, "High"]), float(live_price))
        elif "High" in out.columns:
            out.at[last_idx, "High"] = float(live_price)
        if "Low" in out.columns and pd.notna(out.at[last_idx, "Low"]):
            out.at[last_idx, "Low"] = min(float(out.at[last_idx, "Low"]), float(live_price))
        elif "Low" in out.columns:
            out.at[last_idx, "Low"] = float(live_price)
    except Exception:
        return df

    return out


@st.cache_data(ttl=6 * 3600, show_spinner=False)
def build_sector_peer_snapshot(universe_tuple: Tuple[str, ...], market: str) -> pd.DataFrame:
    if not universe_tuple:
        return pd.DataFrame()

    rows = []

    def _one(raw_ticker: str):
        norm = normalize_ticker(raw_ticker, market)
        try:
            row = fetch_fundamentals_generic(norm, market)
            if not isinstance(row, dict):
                raise ValueError("fetch_fundamentals_generic dict döndürmedi.")
            row["raw_ticker"] = raw_ticker
            return row
        except Exception as e:
            log_app_error("build_sector_peer_snapshot.fetch_one", e, {"ticker": raw_ticker, "normalized": norm}, level="DEBUG")
            return {"raw_ticker": raw_ticker, "ticker": norm, "error": str(e)}

    max_workers = min(12, max(1, min(4, len(universe_tuple)) if len(universe_tuple) <= 8 else max(4, len(universe_tuple) // 8)))

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {ex.submit(_one, raw): raw for raw in universe_tuple}
        for fut in as_completed(futures):
            raw = futures.get(fut, "")
            try:
                res = fut.result()
                if isinstance(res, dict) and not res.get("error"):
                    rows.append(res)
                elif isinstance(res, dict) and res.get("error"):
                    # Hata loglandı, tabloya dahil etmiyoruz.
                    continue
            except Exception as e:
                log_app_error("build_sector_peer_snapshot.thread", e, {"ticker": raw}, level="DEBUG")

    df_peers = pd.DataFrame(rows)
    if df_peers.empty:
        return df_peers

    for col in ["trailingPE", "priceToBook", "forwardPE", "marketCap"]:
        if col in df_peers.columns:
            df_peers[col] = pd.to_numeric(df_peers[col], errors="coerce")

    return df_peers


def get_sector_relative_value_summary(ticker: str, market: str, universe: List[str]) -> Dict[str, Any]:
    out = {"label": "N/A", "delta_pct": np.nan, "detail": "Sektör karşılaştırması yapılamadı."}
    if not universe:
        return out

    peers = build_sector_peer_snapshot(tuple(universe), market)
    if peers.empty:
        return out

    target_row = peers[peers["ticker"] == ticker]
    if target_row.empty:
        return out

    target_row = target_row.iloc[0]
    sector = str(target_row.get("sector", "") or "").strip()
    if not sector:
        return out

    sector_peers = peers[(peers["sector"].fillna("").astype(str).str.strip() == sector) & (peers["ticker"] != ticker)].copy()
    if sector_peers.empty:
        return out

    target_pe = safe_float(target_row.get("trailingPE"))
    target_pb = safe_float(target_row.get("priceToBook"))
    sec_pe = pd.to_numeric(sector_peers.get("trailingPE"), errors="coerce").replace([np.inf, -np.inf], np.nan)
    sec_pb = pd.to_numeric(sector_peers.get("priceToBook"), errors="coerce").replace([np.inf, -np.inf], np.nan)

    comps = []
    if pd.notna(target_pe) and sec_pe.notna().sum() >= 3:
        sec_pe_med = float(sec_pe.median())
        if np.isfinite(sec_pe_med) and sec_pe_med > 0:
            comps.append((target_pe / sec_pe_med - 1.0) * 100.0)
    if pd.notna(target_pb) and sec_pb.notna().sum() >= 3:
        sec_pb_med = float(sec_pb.median())
        if np.isfinite(sec_pb_med) and sec_pb_med > 0:
            comps.append((target_pb / sec_pb_med - 1.0) * 100.0)

    if not comps:
        return out

    delta_pct = float(np.mean(comps))
    if delta_pct >= 5:
        label = "Pahalı"
        detail = f"Sektör ortalamasına göre %{abs(delta_pct):.1f} daha pahalı"
    elif delta_pct <= -5:
        label = "Ucuz"
        detail = f"Sektör ortalamasına göre %{abs(delta_pct):.1f} daha ucuz"
    else:
        label = "Nötr"
        detail = f"Sektör ortalamasına göre fark %{abs(delta_pct):.1f}"

    out.update({"label": label, "delta_pct": delta_pct, "detail": f"{sector} | {detail}"})
    return out


def backtest_monte_carlo(eq: pd.Series, n_sims: int = 300, bars: Optional[int] = None) -> Dict[str, Any]:
    if eq is None or len(eq) < 5:
        return {"error": "Monte Carlo için yeterli sermaye eğrisi yok."}

    ret = pd.Series(eq).pct_change().dropna()
    if ret.empty:
        return {"error": "Monte Carlo için yeterli getiri serisi yok."}

    horizon = int(bars) if bars else int(len(ret))
    horizon = max(5, min(horizon, len(ret)))
    source = ret.tail(min(len(ret), max(40, horizon))).astype(float).values

    sims = []
    final_returns = []
    max_dds = []

    for _ in range(int(n_sims)):
        sampled = np.random.choice(source, size=horizon, replace=True)
        path = np.cumprod(1.0 + sampled)
        sims.append(path)
        final_returns.append(float(path[-1] - 1.0))
        peak = np.maximum.accumulate(path)
        dd = (path / peak) - 1.0
        max_dds.append(float(np.min(dd)))

    sims_arr = np.asarray(sims, dtype=float)
    p10 = np.nanpercentile(sims_arr, 10, axis=0)
    p50 = np.nanpercentile(sims_arr, 50, axis=0)
    p90 = np.nanpercentile(sims_arr, 90, axis=0)

    fig = go.Figure()
    x = list(range(1, horizon + 1))
    fig.add_trace(go.Scatter(x=x, y=p10, name="P10", mode="lines", line=dict(dash="dot")))
    fig.add_trace(go.Scatter(x=x, y=p50, name="Median", mode="lines"))
    fig.add_trace(go.Scatter(x=x, y=p90, name="P90", mode="lines", line=dict(dash="dot")))
    fig.update_layout(height=320, title="Monte Carlo Simülasyonu (Bootstrap)", xaxis_title="Bar", yaxis_title="Birikimli Getiri Katsayısı")

    return {
        "figure": fig,
        "median_return_pct": float(np.nanmedian(final_returns) * 100.0),
        "p10_return_pct": float(np.nanpercentile(final_returns, 10) * 100.0),
        "p90_return_pct": float(np.nanpercentile(final_returns, 90) * 100.0),
        "median_max_dd_pct": float(np.nanmedian(max_dds) * 100.0),
    }


def donchian_channels(high: pd.Series, low: pd.Series, window: int = 20) -> Tuple[pd.Series, pd.Series, pd.Series]:
    upper = high.rolling(int(window), min_periods=int(window)).max()
    lower = low.rolling(int(window), min_periods=int(window)).min()
    mid = (upper + lower) / 2.0
    return upper, mid, lower


def trend_with_pattern_entry_signals(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["TWP_SMA50"] = sma(out["Close"], 50)
    out["TWP_SMA200"] = sma(out["Close"], 200)
    out["TWP_TREND_UP"] = (out["TWP_SMA50"] > out["TWP_SMA200"]).fillna(False)
    out["TWP_TREND_DOWN"] = (out["TWP_SMA50"] < out["TWP_SMA200"]).fillna(False)

    bull_patterns = (
        out.get("PATTERN_ENGULFING_BULL", False).astype(bool)
        | out.get("PATTERN_HAMMER", False).astype(bool)
        | out.get("PATTERN_PIERCING", False).astype(bool)
        | out.get("KANGAROO_BULL", 0).astype(bool)
    )
    bear_patterns = (
        out.get("PATTERN_ENGULFING_BEAR", False).astype(bool)
        | out.get("PATTERN_SHOOTING_STAR", False).astype(bool)
        | out.get("PATTERN_DARK_CLOUD", False).astype(bool)
        | out.get("KANGAROO_BEAR", 0).astype(bool)
    )

    out["TWP_LONG_ENTRY"] = out["TWP_TREND_UP"] & bull_patterns & (out["Close"] > out.get("EMA50", out["Close"]))
    out["TWP_SHORT_ENTRY"] = out["TWP_TREND_DOWN"] & bear_patterns & (out["Close"] < out.get("EMA50", out["Close"]))
    return out


def donchian_5_20_system(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["D520_SMA5"] = sma(out["Close"], 5)
    out["D520_SMA20"] = sma(out["Close"], 20)
    out["D520_LONG"] = (out["D520_SMA5"] > out["D520_SMA20"]).fillna(False)
    out["D520_SHORT"] = (out["D520_SMA5"] < out["D520_SMA20"]).fillna(False)
    out["D520_BUY_SIG"] = out["D520_LONG"] & (~out["D520_LONG"].shift(1).fillna(False))
    out["D520_SELL_SIG"] = out["D520_SHORT"] & (~out["D520_SHORT"].shift(1).fillna(False))
    return out


def richard_dennis_system(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["RD_UPPER20"], out["RD_MID20"], out["RD_LOWER20"] = donchian_channels(out["High"], out["Low"], 20)
    out["RD_EXIT_UPPER10"], _, out["RD_EXIT_LOWER10"] = donchian_channels(out["High"], out["Low"], 10)
    out["RD_LONG_ENTRY"] = (out["Close"] > out["RD_UPPER20"].shift(1)).fillna(False)
    out["RD_LONG_EXIT"] = (out["Close"] < out["RD_EXIT_LOWER10"].shift(1)).fillna(False)
    out["RD_SHORT_ENTRY"] = (out["Close"] < out["RD_LOWER20"].shift(1)).fillna(False)
    out["RD_SHORT_EXIT"] = (out["Close"] > out["RD_EXIT_UPPER10"].shift(1)).fillna(False)
    return out


def build_system_overlay_chart(
    df: pd.DataFrame,
    title: str,
    line_specs: List[Tuple[str, str]],
    marker_specs: Optional[List[Tuple[str, str, str]]] = None,
) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=df.index, open=df["Open"], high=df["High"], low=df["Low"], close=df["Close"], name="Fiyat"
    ))
    for col, label in line_specs:
        if col in df.columns:
            fig.add_trace(go.Scatter(x=df.index, y=df[col], mode="lines", name=label))
    marker_specs = marker_specs or []
    for col, label, symbol in marker_specs:
        if col in df.columns:
            filt = pd.Series(df[col]).fillna(False).astype(bool)
            if filt.any():
                fig.add_trace(go.Scatter(
                    x=df.index[filt],
                    y=df["Close"][filt],
                    mode="markers",
                    name=label,
                    marker=dict(symbol=symbol, size=10),
                ))
    fig.update_layout(height=620, title=title, xaxis_rangeslider_visible=False)
    return fig

@st.cache_data(ttl=30, show_spinner=False)
def get_live_price(ticker: str) -> dict:
    out = {"last_price": np.nan, "currency": "", "exchange": "", "asof": ""}
    try:
        t = yf.Ticker(ticker)
        fi = getattr(t, "fast_info", None)
        if fi:
            out["last_price"] = safe_float(fi.get("last_price") or fi.get("lastPrice"))
            out["currency"] = fi.get("currency") or ""
            out["exchange"] = fi.get("exchange") or ""
            out["asof"] = str(fi.get("last_trade_time") or fi.get("lastTradeDate") or "")
    except Exception:
        pass
    return out

@st.cache_data(ttl=12 * 3600, show_spinner=False)
def get_short_info(ticker: str) -> dict:
    out = {"short_percent_float": np.nan, "short_ratio": np.nan}
    try:
        t = yf.Ticker(ticker)
        info = t.info or {}
        out["short_percent_float"] = safe_float(info.get("shortPercentOfFloat"))
        out["short_ratio"] = safe_float(info.get("shortRatio"))
    except Exception:
        pass
    return out



def fetch_fundamentals_generic(ticker: str, market: str = "BIST") -> Dict[str, Any]:
    """
    Güvenli temel veri fallback'i.
    Kodun farklı bölümleri bu fonksiyonu bekliyor; tanım yoksa screener/sector snapshot NameError üretir.
    Yahoo info/fast_info kaynaklarından minimum gerekli alanları doldurur.
    """
    row: Dict[str, Any] = {"ticker": ticker, "market": market}
    try:
        tk = yf.Ticker(ticker)
        info = {}
        try:
            info = tk.info or {}
        except Exception as e:
            log_app_error("fetch_fundamentals_generic.info", e, {"ticker": ticker}, level="DEBUG")

        fast = {}
        try:
            fast = dict(tk.fast_info or {})
        except Exception as e:
            log_app_error("fetch_fundamentals_generic.fast_info", e, {"ticker": ticker}, level="DEBUG")

        def pick(*keys):
            for k in keys:
                if k in info and info.get(k) is not None:
                    return info.get(k)
                if k in fast and fast.get(k) is not None:
                    return fast.get(k)
            return np.nan

        row.update({
            "company": info.get("shortName") or info.get("longName") or naked_ticker(ticker),
            "sector": info.get("sector", "N/A"),
            "industry": info.get("industry", "N/A"),
            "currency": info.get("currency", "TRY" if market == "BIST" else "USD"),
            "currentPrice": safe_float(pick("currentPrice", "last_price", "lastPrice")),
            "marketCap": safe_float(pick("marketCap", "market_cap")),
            "trailingPE": safe_float(pick("trailingPE")),
            "forwardPE": safe_float(pick("forwardPE")),
            "priceToBook": safe_float(pick("priceToBook")),
            "debtToEquity": safe_float(pick("debtToEquity")),
            "returnOnEquity": safe_float(pick("returnOnEquity")),
            "profitMargins": safe_float(pick("profitMargins")),
            "revenueGrowth": safe_float(pick("revenueGrowth")),
            "freeCashflow": safe_float(pick("freeCashflow")),
            "totalCash": safe_float(pick("totalCash")),
            "totalDebt": safe_float(pick("totalDebt")),
            "enterpriseValue": safe_float(pick("enterpriseValue")),
            "ebitda": safe_float(pick("ebitda")),
        })
        return row
    except Exception as e:
        log_app_error("fetch_fundamentals_generic", e, {"ticker": ticker, "market": market})
        row["error"] = str(e)
        return row


# =============================
# Gemini helpers
# =============================
def _get_secret(name: str, default: str = "") -> str:
    try:
        v = st.secrets.get(name, "")
        if v is None:
            return default
        return str(v).strip()
    except Exception:
        return default


def _http_post_json(url: str, payload: dict, headers: dict = None, timeout: int = 60) -> dict:
    r = requests.post(url, json=payload, headers=headers, timeout=timeout)
    try:
        data = r.json()
    except Exception:
        data = {"error": {"message": f"Non-JSON response (status={r.status_code})", "text": r.text[:500]}}
    if r.status_code >= 400:
        if "error" not in data:
            data["error"] = {"message": f"HTTP {r.status_code}", "text": str(data)[:500]}
    return data


def _extract_gemini_text(resp: dict) -> str:
    if not isinstance(resp, dict):
        return str(resp)
    if resp.get("error"):
        return f"Gemini API error: {resp['error'].get('message','')}"
    cands = resp.get("candidates") or []
    if not cands:
        return "Gemini: boş cevap döndü (candidates yok)."
    parts = (cands[0].get("content") or {}).get("parts") or []
    if not parts:
        return "Gemini: boş cevap döndü (parts yok)."
    texts = []
    for p in parts:
        if isinstance(p, dict) and "text" in p:
            texts.append(p["text"])
    return "\n".join(texts).strip() if texts else "Gemini: metin üretmedi."


def gemini_generate_text(
    *,
    prompt: str,
    model: str = "gemini-3.5-flash",
    temperature: float = 0.2,
    max_output_tokens: int = 2048,
    image_bytes: Optional[bytes] = None,
) -> str:
    api_key = _get_secret("GEMINI_API_KEY", "")
    if not api_key:
        return "GEMINI_API_KEY bulunamadı. Streamlit Cloud > Settings > Secrets içine GEMINI_API_KEY=... ekleyin."
    
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
    headers = {"x-goog-api-key": api_key}

    parts = [{"text": prompt}]
    if image_bytes:
        b64_img = base64.b64encode(image_bytes).decode("utf-8")
        parts.append(
            {
                "inlineData": {
                    "mimeType": "image/png",
                    "data": b64_img,
                }
            }
        )

    payload = {
        "contents": [{"role": "user", "parts": parts}],
        "generationConfig": {
            "temperature": float(temperature),
            "maxOutputTokens": int(max_output_tokens),
        },
    }
    resp = _http_post_json(url, payload, headers=headers, timeout=90)
    return _extract_gemini_text(resp)


# =============================
# Sentiment Analysis via Google News RSS + Gemini
# =============================
@st.cache_data(ttl=30 * 60, show_spinner=False)
def get_news_sentiment(
    ticker: str,
    company_name: str = "",
    gemini_model: str = "gemini-1.5-flash",
    gemini_temp: float = 0.2,
    max_tokens: int = 2048,
) -> Dict[str, Any]:
    
    try:
        if company_name and company_name != "":
            query = f"{company_name} stock"
        else:
            query = f"{ticker} stock"

        url = f"https://news.google.com/rss/search?q={requests.utils.quote(query)}&hl=en-US&gl=US&ceid=US:en"
        
        resp = requests.get(url, timeout=15)
        if resp.status_code != 200:
            return {"error": f"Haberler çekilemedi (HTTP {resp.status_code})", "sentiment": None, "summary": ""}
            
        root = ET.fromstring(resp.content)
        
        news_items = []
        for item in root.findall(".//item")[:10]:
            title_node = item.find("title")
            link_node = item.find("link")
            if title_node is not None and title_node.text:
                t = title_node.text
                l = link_node.text if (link_node is not None and link_node.text) else ""
                news_items.append({"title": t, "link": l})

        if not news_items:
            return {"error": "Haber bulunamadı", "sentiment": None, "summary": ""}

        prompt_titles = [item["title"] for item in news_items]
        prompt = f"""Aşağıdaki haber başlıklarının duygu analizini yap (pozitif, negatif, nötr).
Sonuçları şu formatta ver:
Pozitif: [sayı]
Negatif: [sayı]
Nötr: [sayı]
- Bileşik skor: (pozitif - negatif) / toplam (örneğin 0.35)
- Kısa bir özet (2 cümle)

Haber Başlıkları:
{chr(10).join([f"- {t}" for t in prompt_titles])}
"""

        response = gemini_generate_text(
            prompt=prompt,
            model=gemini_model,
            temperature=gemini_temp,
            max_output_tokens=max_tokens,
            image_bytes=None,
        )

        pos_match = re.search(r"Pozitif:?\s*(\d+)", response, re.IGNORECASE)
        neg_match = re.search(r"Negatif:?\s*(\d+)", response, re.IGNORECASE)
        neu_match = re.search(r"Nötr:?\s*(\d+)", response, re.IGNORECASE)

        pos = int(pos_match.group(1)) if pos_match else 0
        neg = int(neg_match.group(1)) if neg_match else 0
        neu = int(neu_match.group(1)) if neu_match else 0
        total = pos + neg + neu
        compound = (pos - neg) / total if total > 0 else 0

        return {
            "error": None,
            "sentiment": compound,
            "summary": response,
            "pos": pos / total if total > 0 else 0,
            "neg": neg / total if total > 0 else 0,
            "neu": neu / total if total > 0 else 0,
            "news_items": news_items[:5], 
        }
    except Exception as e:
        return {"error": str(e), "sentiment": None, "summary": ""}


# =============================
# Price Action
# =============================
def price_action_pack(df: pd.DataFrame, last_n: int = 20) -> dict:
    use = df.tail(last_n).copy()
    if use.empty or len(use) < 10:
        return {"note": "insufficient_bars", "last_n": int(len(use))}

    o = use["Open"].astype(float)
    h = use["High"].astype(float)
    l = use["Low"].astype(float)
    c = use["Close"].astype(float)

    swing_highs, swing_lows = _swing_points(h, l, left=2, right=2)

    q20 = float(np.quantile(c.values, 0.20))
    q50 = float(np.quantile(c.values, 0.50))
    q80 = float(np.quantile(c.values, 0.80))

    recent_highs = [v for _, v in swing_highs[-5:]] if swing_highs else []
    recent_lows = [v for _, v in swing_lows[-5:]] if swing_lows else []
    res = max(recent_highs) if recent_highs else float(h.max())
    sup = min(recent_lows) if recent_lows else float(l.min())

    last_close = float(c.iloc[-1])
    prev_close = float(c.iloc[-2]) if len(c) >= 2 else last_close
    last_high = float(h.iloc[-1])
    last_low = float(l.iloc[-1])

    bull_break = (last_close > res) and (prev_close <= res)
    bear_break = (last_close < sup) and (prev_close >= sup)

    vol_ok = None
    if "Volume" in use.columns:
        vol = use["Volume"].astype(float)
        vol_sma = float(vol.rolling(10).mean().iloc[-1]) if len(vol) >= 10 else float(vol.mean())
        vol_ok = float(vol.iloc[-1]) > vol_sma if np.isfinite(vol_sma) else None

    impulse_up = (c.diff().tail(3) > 0).all() and (last_close >= q80)
    impulse_dn = (c.diff().tail(3) < 0).all() and (last_close <= q20)

    ob = None
    if impulse_up:
        for i in range(len(use) - 4, -1, -1):
            if c.iloc[i] < o.iloc[i]:
                ob = {
                    "type": "bullish_order_block_proxy",
                    "index": str(use.index[i]),
                    "open": float(o.iloc[i]),
                    "high": float(h.iloc[i]),
                    "low": float(l.iloc[i]),
                    "close": float(c.iloc[i]),
                }
                break
    elif impulse_dn:
        for i in range(len(use) - 4, -1, -1):
            if c.iloc[i] > o.iloc[i]:
                ob = {
                    "type": "bearish_order_block_proxy",
                    "index": str(use.index[i]),
                    "open": float(o.iloc[i]),
                    "high": float(h.iloc[i]),
                    "low": float(l.iloc[i]),
                    "close": float(c.iloc[i]),
                }
                break

    pack = {
        "last_n": int(len(use)),
        "q20": q20,
        "q50": q50,
        "q80": q80,
        "support": sup,
        "resistance": res,
        "bull_breakout": bool(bull_break),
        "bear_breakout": bool(bear_break),
        "vol_confirm": (None if vol_ok is None else bool(vol_ok)),
        "last_bar": {
            "t": str(use.index[-1]),
            "open": float(o.iloc[-1]),
            "high": last_high,
            "low": last_low,
            "close": last_close,
        },
        "swing_highs": [{"t": str(t), "p": float(p)} for t, p in swing_highs[-6:]],
        "swing_lows": [{"t": str(t), "p": float(p)} for t, p in swing_lows[-6:]],
        "order_block_proxy": ob,
    }
    return pack


def df_snapshot_for_llm(df: pd.DataFrame, n: int = 25) -> dict:
    use_cols = [
        "Open",
        "High",
        "Low",
        "Close",
        "Volume",
        "EMA50",
        "EMA200",
        "RSI",
        "MACD",
        "MACD_signal",
        "MACD_hist",
        "BB_mid",
        "BB_upper",
        "BB_lower",
        "ATR",
        "ATR_PCT",
        "VOL_SMA",
        "VOL_RATIO",
        "BB_WIDTH",
        "SCORE",
        "ENTRY",
        "EXIT",
        "RSI_OVERBOUGHT",
        "BB_OVERBOUGHT",
        "BB_OVERSOLD",
        "VOLUME_SPIKE",
        "PRICE_EXTREME",
        "STOCH_OVERBOUGHT",
        "WEAK_UPTREND",
        "KANGAROO_BULL",
        "KANGAROO_BEAR"
    ]
    cols = [c for c in use_cols if c in df.columns]
    tail = df[cols].tail(n).copy()
    tail.index = tail.index.astype(str)
    
    summary = {}
    if not df.empty:
        summary["rsi_last"] = float(df["RSI"].iloc[-1]) if "RSI" in df else None
        summary["rsi_5d_avg"] = float(df["RSI"].tail(5).mean()) if "RSI" in df else None
        if "EMA50" in df and "EMA200" in df:
            summary["trend"] = "up" if df["EMA50"].iloc[-1] > df["EMA200"].iloc[-1] else "down"

    return {
        "cols": cols,
        "n": int(len(tail)),
        "last_index": str(tail.index[-1]) if len(tail) else None,
        "rows": tail.to_dict(orient="records"),
        "summary": summary
    }


# =============================
# Presets
# =============================
PRESETS = {
    "Defansif": {
        "rsi_entry_level": 52,
        "rsi_exit_level": 46,
        "atr_pct_max": 0.06,
        "atr_stop_mult": 2.0,
        "time_stop_bars": 15,
        "take_profit_mult": 2.5,
    },
    "Dengeli": {
        "rsi_entry_level": 50,
        "rsi_exit_level": 45,
        "atr_pct_max": 0.08,
        "atr_stop_mult": 1.5,
        "time_stop_bars": 10,
        "take_profit_mult": 2.0,
    },
    "Agresif": {
        "rsi_entry_level": 48,
        "rsi_exit_level": 43,
        "atr_pct_max": 0.10,
        "atr_stop_mult": 1.2,
        "time_stop_bars": 7,
        "take_profit_mult": 1.5,
    },
}


# =============================
# Screener row finder
# =============================
def find_screener_row(sdf: pd.DataFrame, ticker: str) -> Optional[Dict[str, Any]]:
    if sdf is None or sdf.empty or "ticker" not in sdf.columns:
        return None

    t = (ticker or "").upper().strip()
    t_naked = naked_ticker(t)

    tmp = sdf.copy()
    tmp["_tk"] = tmp["ticker"].astype(str).str.upper().str.strip()
    tmp["_tk_naked"] = tmp["_tk"].str.replace(".IS", "", regex=False)

    m = tmp[(tmp["_tk"] == t) | (tmp["_tk"] == f"{t_naked}.IS") | (tmp["_tk_naked"] == t_naked)]
    if m.empty:
        return None

    row = m.iloc[0].drop(labels=["_tk", "_tk_naked"], errors="ignore").to_dict()
    return row


def merge_fa_row(
    screener_row: Optional[Dict[str, Any]],
    fundamentals: Optional[Dict[str, Any]],
    fa_eval: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    if fundamentals:
        out.update(fundamentals)
    if screener_row:
        out.update(screener_row)
    if fa_eval:
        out["FA_mode"] = fa_eval.get("mode")
        out["FA_score"] = fa_eval.get("score")
        out["FA_pass"] = fa_eval.get("passed")
        out["FA_ok_count"] = fa_eval.get("ok_cnt")
        out["FA_coverage"] = fa_eval.get("coverage")
    return out


# =============================
# REPORT EXPORT
# =============================
def build_html_report(
    title: str,
    meta: dict,
    checkpoints: dict,
    metrics: dict,
    tp: dict,
    rr_info: dict,
    figs: Dict[str, go.Figure],
    fa_row: Optional[Dict[str, Any]] = None,
    gemini_insight: Optional[str] = None,
    pa_pack: Optional[Dict[str, Any]] = None,
    sentiment_summary: Optional[str] = None,
    sentiment_items: Optional[List[dict]] = None,
    overbought_result: Optional[Dict[str, Any]] = None,
) -> bytes:
    def esc(x):
        return str(x).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

    fig_blocks = []
    first = True
    for name, fig in (figs or {}).items():
        fig_html = fig.to_html(full_html=False, include_plotlyjs=("cdn" if first else False))
        first = False
        fig_blocks.append(f"<h3>{esc(name)}</h3>{fig_html}")

    cp_list = "".join([f"<li>{'✅' if v else '❌'} {esc(k)}</li>" for k, v in checkpoints.items()])

    bull = tp.get("bull")
    bear = tp.get("bear")
    levels = tp.get("levels", []) or []
    
    levels_txt = "<br>".join([
        f"{x['price']:.2f} (Güç: %{x['strength_pct']:.0f}, Uzunluk: {x['duration_bars']} Bar, Hacim: %{x['vol_diff_pct']:+.1f})"
        for x in levels[:120]
    ]) if levels else "N/A"

    show_cols = [
        ("ticker", "Ticker"),
        ("longName", "Name"),
        ("FA_pass", "FA_pass"),
        ("FA_score", "FA_score"),
        ("FA_ok_count", "FA_ok_count"),
        ("FA_coverage", "FA_coverage"),
        ("sector", "Sector"),
        ("industry", "Industry"),
        ("trailingPE", "Trailing PE"),
        ("forwardPE", "Forward PE"),
        ("pegRatio", "PEG"),
        ("priceToSalesTrailing12Months", "P/S"),
        ("priceToBook", "P/B"),
        ("returnOnEquity", "ROE"),
        ("operatingMargins", "Op Margin"),
        ("profitMargins", "Profit Margin"),
        ("debtToEquity", "Debt/Equity"),
        ("revenueGrowth", "Revenue Growth"),
        ("earningsGrowth", "Earnings Growth"),
        ("marketCap", "Market Cap"),
    ]

    fa_rows_html = ""
    if fa_row:
        for key, label in show_cols:
            val = fa_row.get(key, "")
            fa_rows_html += f"<tr><td><b>{esc(label)}</b></td><td>{esc(val)}</td></tr>"
    else:
        fa_rows_html = "<tr><td colspan='2'>Screener satırı bulunamadı (screener çalıştırılmamış olabilir).</td></tr>"

    overbought_html = ""
    if overbought_result:
        ob_details = "<ul>"
        for _, v in overbought_result.get("details", {}).items():
            ob_details += f"<li>{esc(v)}</li>"
        ob_details += "</ul>"
        
        sf_val = overbought_result.get("short_percent_float")
        sr_val = overbought_result.get("short_ratio")
        sf_str = f"{sf_val * 100:.2f}%" if pd.notna(sf_val) else "N/A"
        sr_str = f"{sr_val:.2f}" if pd.notna(sr_val) else "N/A"
        
        overbought_html = f"""
        <div class="card" style="margin-top:16px;">
            <h2>📊 Aşırı Alım / Spekülasyon Analizi</h2>
            <div><b>Karar:</b> {esc(overbought_result['verdict'])}</div>
            <div><b>Aşırı Alım Skoru:</b> {overbought_result['overbought_score']}/100</div>
            <div><b>Aşırı Satım Skoru:</b> {overbought_result['oversold_score']}/100</div>
            <div><b>Spekülasyon Skoru:</b> {overbought_result['speculation_score']}/100</div>
            <div><b>Kısa Poz. % (Short Float):</b> {sf_str}</div>
            <div><b>Kapatma Gün (Days to Cover):</b> {sr_str}</div>
            <div><b>Detaylar:</b> {ob_details}</div>
        </div>
        """

    gemini_block = ""
    if gemini_insight:
        gemini_block = f"""
        <div class="card" style="margin-top:16px;">
            <h2>Gemini — Chart & Price Action Insight</h2>
            <pre style="white-space:pre-wrap; font-family:inherit;">{esc(gemini_insight)}</pre>
        </div>
        """

    pa_block = ""
    if pa_pack:
        pa_block = f"""
        <div class="card" style="margin-top:16px;">
            <h2>Price Action Pack (Last {esc(pa_pack.get('last_n',''))} Bars)</h2>
            <pre style="white-space:pre-wrap; font-family:monospace; font-size:12px;">{esc(json.dumps(pa_pack, ensure_ascii=False, indent=2))}</pre>
        </div>
        """

    sentiment_block = ""
    if sentiment_summary:
        links_html = ""
        if sentiment_items:
            links_html = "<br><br><b>Kaynak Haberler:</b><ul>"
            for item in sentiment_items:
                links_html += f"<li><a href='{esc(item['link'])}' target='_blank'>{esc(item['title'])}</a></li>"
            links_html += "</ul>"

        sentiment_block = f"""
        <div class="card" style="margin-top:16px;">
            <h2>Haber Duygu Analizi (Google News + Gemini)</h2>
            <pre style="white-space:pre-wrap; font-family:inherit;">{esc(sentiment_summary)}</pre>
            {links_html}
        </div>
        """

    html = f"""
<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>{esc(title)}</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 24px; }}
    .muted {{ color: #666; font-size: 12px; }}
    .grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }}
    .card {{ border: 1px solid #ddd; border-radius: 10px; padding: 14px; }}
    h1,h2,h3 {{ margin: 0 0 8px 0; }}
    ul {{ margin: 8px 0 0 18px; }}
    table {{ width: 100%; border-collapse: collapse; }}
    td {{ border-top: 1px solid #eee; padding: 6px 8px; vertical-align: top; }}
    @media print {{
      .no-print {{ display: none; }}
      body {{ margin: 10mm; }}
    }}
  </style>
</head>
<body>
  <div class="no-print card" style="background:#fff7e6;border-color:#ffd591;">
    <b>PDF yapmak için:</b> Bu dosyayı indir → tarayıcıda aç → <b>Ctrl+P</b> → <b>Save as PDF</b>.
  </div>
  <h1>{esc(title)}</h1>
  <div class="muted">
    Generated: {esc(time.strftime('%Y-%m-%d %H:%M:%S'))}<br>
    Market: {esc(meta.get('market'))} | Ticker: {esc(meta.get('ticker'))} | Interval: {esc(meta.get('interval'))} | Period: {esc(meta.get('period'))}<br>
    Preset: {esc(meta.get('preset'))} | EMA: {esc(meta.get('ema_fast'))}/{esc(meta.get('ema_slow'))} | RSI: {esc(meta.get('rsi_period'))} | BB: {esc(meta.get('bb_period'))}/{esc(meta.get('bb_std'))} | ATR: {esc(meta.get('atr_period'))} | VolSMA: {esc(meta.get('vol_sma'))}
  </div>
  <div class="grid" style="margin-top:14px;">
    <div class="card">
      <h2>Checkpoints</h2>
      <ul>{cp_list}</ul>
    </div>
    <div class="card">
      <h2>Backtest</h2>
      <div>Total Return: {metrics.get('Total Return',0)*100:.1f}%</div>
      <div>Ann Return: {metrics.get('Annualized Return',0)*100:.1f}%</div>
      <div>Sharpe: {metrics.get('Sharpe',0):.2f}</div>
      <div>Max DD: {metrics.get('Max Drawdown',0)*100:.1f}%</div>
      <div>Trades: {metrics.get('Trades',0)}</div>
      <div>Win Rate: {metrics.get('Win Rate',0)*100:.1f}%</div>
      <div>Beta: {metrics.get('Beta',0):.2f}</div>
      <div>Alpha: {metrics.get('Alpha',0):.2f}</div>
      <div>Info Ratio: {metrics.get('Information Ratio',0):.2f}</div>
      <div>Ulcer Index: {metrics.get('Ulcer Index',0):.4f}</div>
      <div>Kelly Önerisi: {metrics.get('Kelly % (öneri)',0):.1f}%</div>
    </div>
  </div>
  {overbought_html}
  <div class="card" style="margin-top:16px;">
    <h2>Target Band</h2>
    <div>Base: {tp.get('base',0):.2f}</div>
    <div>Bull: {(bull[0] if bull else 0):.2f} → {(bull[1] if bull else 0):.2f} | R1: {(bull[2] if bull else 'N/A')}</div>
    <div>Bear: {(bear[0] if bear else 0):.2f} → {(bear[1] if bear else 0):.2f} | S1: {(bear[2] if bear else 'N/A')}</div>
    <div>RR: {('N/A' if rr_info.get('rr') is None else f"1:{rr_info.get('rr'):.2f}")}</div>
    <div class="muted"><br>Seviyeler ve Güçleri:<br>{levels_txt}</div>
  </div>
  {gemini_block}
  {sentiment_block}
  {pa_block}
  <div class="card" style="margin-top:16px;">
    <h2>Fundamental Screener Snapshot (Selected Ticker)</h2>
    <table>{fa_rows_html}</table>
  </div>
  <div style="margin-top:18px;">
    {''.join(fig_blocks)}
  </div>
</body>
</html>
"""
    return html.encode("utf-8")


def _plotly_fig_to_png_bytes(fig: go.Figure) -> Optional[bytes]:
    try:
        return fig.to_image(format="png", scale=2)
    except Exception:
        return None


def _pdf_write_lines(c, lines: List[str], x: float, y: float, lh: float, bottom: float):
    for line in lines:
        if y <= bottom:
            c.showPage()
            y = A4[1] - 2.0 * cm
        c.drawString(x, y, (line or "")[:220])
        y -= lh
    return y


def generate_pdf_report(
    *,
    title: str,
    subtitle: str,
    meta: dict,
    checkpoints: dict,
    ta_summary: dict,
    target_band: dict,
    rr_info: dict,
    backtest_metrics: dict,
    fa_row: Optional[Dict[str, Any]],
    levels: Optional[List[dict]],
    trades_df: Optional[pd.DataFrame],
    figs: Optional[Dict[str, go.Figure]],
    include_charts: bool = True,
    gemini_insight: Optional[str] = None,
    pa_pack: Optional[Dict[str, Any]] = None,
    sentiment_summary: Optional[str] = None,
    sentiment_items: Optional[List[dict]] = None,
    overbought_result: Optional[Dict[str, Any]] = None,
) -> Optional[bytes]:
    if not REPORTLAB_OK:
        return None

    buf = BytesIO()
    c = canvas.Canvas(buf, pagesize=A4)
    W, H = A4
    left = 1.6 * cm
    right = W - 1.6 * cm
    top = H - 1.6 * cm
    bottom = 1.6 * cm
    y = top

    c.setFont("Helvetica-Bold", 16)
    c.drawString(left, y, title[:90])
    y -= 18

    c.setFont("Helvetica", 10)
    c.drawString(left, y, subtitle[:140])
    y -= 14

    c.setFont("Helvetica", 9)
    y = _pdf_write_lines(
        c,
        [
            f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}",
            f"Market: {meta.get('market','')} | Ticker: {meta.get('ticker','')} | Interval: {meta.get('interval','')} | Period: {meta.get('period','')}",
            f"Preset: {meta.get('preset','')} | EMA: {meta.get('ema_fast','')}/{meta.get('ema_slow','')} | RSI: {meta.get('rsi_period','')} | BB: {meta.get('bb_period','')}/{meta.get('bb_std','')} | ATR: {meta.get('atr_period','')} | VolSMA: {meta.get('vol_sma','')}",
        ],
        left,
        y,
        12,
        bottom,
    )
    y -= 6

    c.setFont("Helvetica-Bold", 12)
    c.drawString(left, y, "Technical Summary")
    y -= 14

    c.setFont("Helvetica", 9)
    y = _pdf_write_lines(
        c,
        [
            f"Recommendation: {ta_summary.get('rec','')}",
            f"Last Close (bar): {ta_summary.get('close','N/A')} | Live/Last: {ta_summary.get('live','N/A')}",
            f"Score: {ta_summary.get('score','N/A')} | RSI: {ta_summary.get('rsi','N/A')} | EMA50: {ta_summary.get('ema50','N/A')} | EMA200: {ta_summary.get('ema200','N/A')} | ATR%: {ta_summary.get('atr_pct','N/A')}",
        ],
        left,
        y,
        12,
        bottom,
    )
    y -= 6

    c.setFont("Helvetica-Bold", 12)
    c.drawString(left, y, "Checkpoints (Last Bar)")
    y -= 14

    c.setFont("Helvetica", 9)
    cp_lines = [f"[{'OK' if v else 'NO'}] {k}" for k, v in checkpoints.items()]
    y = _pdf_write_lines(c, cp_lines, left, y, 11, bottom)
    y -= 6

    if overbought_result:
        c.setFont("Helvetica-Bold", 12)
        c.drawString(left, y, "Aşırı Alım / Spekülasyon")
        y -= 14
        c.setFont("Helvetica", 9)
        
        sf_val = overbought_result.get("short_percent_float", np.nan)
        sr_val = overbought_result.get("short_ratio", np.nan)
        sf_str = f"{sf_val * 100:.2f}%" if pd.notna(sf_val) else "N/A"
        sr_str = f"{sr_val:.2f}" if pd.notna(sr_val) else "N/A"
        
        ob_lines = [
            f"Karar: {overbought_result['verdict']}",
            f"Aşırı Alım Skoru: {overbought_result['overbought_score']}/100",
            f"Aşırı Satım Skoru: {overbought_result['oversold_score']}/100",
            f"Spekülasyon Skoru: {overbought_result['speculation_score']}/100",
            f"Kısa Poz. %: {sf_str} | Kapatma Gün: {sr_str}",
            "Detaylar:",
        ]
        for _, v in overbought_result.get("details", {}).items():
            ob_lines.append(f"  - {v}")
        y = _pdf_write_lines(c, ob_lines, left, y, 11, bottom)
        y -= 6

    c.setFont("Helvetica-Bold", 12)
    c.drawString(left, y, "Target Price Band (Scenario)")
    y -= 14
    c.setFont("Helvetica", 9)

    base = target_band.get("base")
    bull = target_band.get("bull")
    bear = target_band.get("bear")
    rr = rr_info.get("rr")
    stop = rr_info.get("stop")

    band_lines = [f"Base: {fmt_num(base)}"]
    if bull:
        band_lines.append(f"Bull: {fmt_num(bull[0])} -> {fmt_num(bull[1])} | Near Resistance: {fmt_num(bull[2])}")
    else:
        band_lines.append("Bull: N/A")

    if bear:
        band_lines.append(f"Bear: {fmt_num(bear[0])} -> {fmt_num(bear[1])} | Near Support: {fmt_num(bear[2])}")
    else:
        band_lines.append("Bear: N/A")

    band_lines.append(f"RR (Backtest first target vs stop): {'N/A' if rr is None else f'1:{rr:.2f}'} | Stop(ATR): {fmt_num(stop)}")
    y = _pdf_write_lines(c, band_lines, left, y, 12, bottom)
    y -= 6

    c.setFont("Helvetica-Bold", 12)
    c.drawString(left, y, "Levels (Güç, Uzunluk, Hacim)")
    y -= 14

    c.setFont("Helvetica", 9)
    if levels:
        lv_lines = []
        for i in range(0, min(len(levels), 20), 2):
            chunk = levels[i:i+2]
            line = " | ".join([f"{x['price']:.2f} (G:%{x.get('strength_pct',0):.0f}, {x.get('duration_bars',0)}B, V:%{x.get('vol_diff_pct',0):+.1f})" for x in chunk])
            lv_lines.append(line)
    else:
        lv_lines = ["N/A"]
    y = _pdf_write_lines(c, lv_lines, left, y, 11, bottom)
    y -= 6

    c.setFont("Helvetica-Bold", 12)
    c.drawString(left, y, "Backtest Summary (Long-only)")
    y -= 14

    c.setFont("Helvetica", 9)
    bm = backtest_metrics or {}
    y = _pdf_write_lines(
        c,
        [
            f"Total Return: {fmt_pct(bm.get('Total Return'))} | Ann Return: {fmt_pct(bm.get('Annualized Return'))} | Ann Vol: {fmt_pct(bm.get('Annualized Volatility'))}",
            f"Sharpe: {fmt_num(bm.get('Sharpe'), 2)} | Sortino: {fmt_num(bm.get('Sortino'), 2)} | Calmar: {fmt_num(bm.get('Calmar'), 2)}",
            f"Max DD: {fmt_pct(bm.get('Max Drawdown'))} | Trades: {bm.get('Trades','')} | Win Rate: {fmt_pct(bm.get('Win Rate'))} | Profit Factor: {fmt_num(bm.get('Profit Factor'), 2)}",
            f"Beta: {fmt_num(bm.get('Beta'),2)} | Alpha: {fmt_num(bm.get('Alpha'),2)} | Info Ratio: {fmt_num(bm.get('Information Ratio'),2)} | Ulcer Index: {fmt_num(bm.get('Ulcer Index'),4)} | Kelly: {fmt_num(bm.get('Kelly % (öneri)'),1)}%",
        ],
        left,
        y,
        12,
        bottom,
    )
    y -= 6

    if sentiment_summary:
        c.setFont("Helvetica-Bold", 12)
        c.drawString(left, y, "Haber Duygu Analizi (Google News)")
        y -= 14
        c.setFont("Helvetica", 9)
        y = _pdf_write_lines(c, sentiment_summary.splitlines(), left, y, 11, bottom)
        y -= 6
        
        if sentiment_items:
            c.setFont("Helvetica-Bold", 10)
            y = _pdf_write_lines(c, ["Kaynak Haberler:"], left, y, 11, bottom)
            c.setFont("Helvetica", 8)
            for item in sentiment_items:
                y = _pdf_write_lines(c, [f"- {item['title'][:110]}"], left, y, 10, bottom)
                c.setFillColorRGB(0, 0, 1)
                y = _pdf_write_lines(c, [f"  {item['link'][:115]}"], left, y, 10, bottom)
                c.setFillColorRGB(0, 0, 0)
            y -= 6

    if pa_pack:
        c.setFont("Helvetica-Bold", 12)
        c.drawString(left, y, "Price Action Pack (Last Bars)")
        y -= 14
        c.setFont("Helvetica", 8)
        pa_txt = json.dumps(pa_pack, ensure_ascii=False, indent=2).splitlines()
        y = _pdf_write_lines(c, pa_txt, left, y, 9, bottom)
        y -= 6

    if gemini_insight:
        c.setFont("Helvetica-Bold", 12)
        c.drawString(left, y, "Gemini Insight")
        y -= 14
        c.setFont("Helvetica", 9)
        gi_lines = (gemini_insight or "").splitlines()
        y = _pdf_write_lines(c, gi_lines, left, y, 11, bottom)
        y -= 6

    c.setFont("Helvetica-Bold", 12)
    c.drawString(left, y, "Fundamental Screener Snapshot (Selected Ticker)")
    y -= 14

    c.setFont("Helvetica", 9)
    if fa_row:
        keys = [
            "ticker", "longName", "FA_pass", "FA_score", "FA_ok_count", "FA_coverage",
            "sector", "industry", "trailingPE", "forwardPE", "pegRatio",
            "priceToSalesTrailing12Months", "priceToBook", "returnOnEquity",
            "operatingMargins", "profitMargins", "debtToEquity",
            "revenueGrowth", "earningsGrowth", "marketCap",
        ]
        lines = [f"{k}: {fa_row.get(k)}" for k in keys if k in fa_row]
        if not lines:
            lines = ["(No fields)"]
    else:
        lines = ["Screener satırı bulunamadı (screener çalıştırılmamış olabilir)."]

    y = _pdf_write_lines(c, lines, left, y, 11, bottom)
    y -= 6

    if trades_df is not None and not trades_df.empty:
        c.setFont("Helvetica-Bold", 12)
        c.drawString(left, y, "Trades (first 25 rows)")
        y -= 14
        c.setFont("Helvetica", 8)

        td = trades_df.copy().head(25)
        cols = [cc for cc in ["entry_date", "entry_price", "exit_date", "exit_price", "exit_reason", "pnl", "return_%", "holding_days"] if cc in td.columns]
        header = " | ".join(cols)
        y = _pdf_write_lines(c, [header], left, y, 10, bottom)

        for _, r in td.iterrows():
            row_txt = " | ".join([str(r.get(k, ""))[:18] for k in cols])
            y = _pdf_write_lines(c, [row_txt], left, y, 10, bottom)

        y -= 6

    chart_added = False
    if include_charts and figs:
        for name, fig in figs.items():
            img = _plotly_fig_to_png_bytes(fig)
            if not img:
                continue
            chart_added = True
            c.showPage()
            c.setFont("Helvetica-Bold", 14)
            c.drawString(left, top, f"Chart: {name}")
            img_reader = ImageReader(BytesIO(img))
            usable_w = (right - left)
            usable_h = (H - 3.2 * cm - 2.0 * cm)
            c.drawImage(img_reader, left, 2.0 * cm, width=usable_w, height=usable_h, preserveAspectRatio=True, anchor="c")

    if include_charts and figs and not chart_added:
        c.showPage()
        c.setFont("Helvetica-Bold", 14)
        c.drawString(left, top, "Charts could not be embedded.")
        c.setFont("Helvetica", 10)
        c.drawString(left, top - 18, "Reason: Plotly image export needs 'kaleido' in requirements.txt.")
        c.drawString(left, top - 34, "Fallback: Download HTML report and print to PDF (keeps charts).")

    c.save()
    buf.seek(0)
    return buf.read()


# =============================
# RR helper 
# =============================
def rr_from_atr_stop(latest_row: pd.Series, tp_dict: dict, cfg: dict):
    close = float(latest_row["Close"])
    atrv = float(latest_row.get("ATR", np.nan)) if pd.notna(latest_row.get("ATR", np.nan)) else np.nan
    
    if not np.isfinite(atrv) or atrv <= 0:
        return {"rr": None, "stop": None, "risk": None, "reward": None}

    if latest_row.get("KANGAROO_BULL", 0) == 1:
        stop = float(latest_row["Low"]) - (0.5 * atrv)
    else:
        stop = close - (float(cfg["atr_stop_mult"]) * atrv)
        
    risk = close - stop

    r1 = None
    if tp_dict and tp_dict.get("bull"):
        r1 = tp_dict["bull"][2] 
        
    if r1 is not None and np.isfinite(r1) and r1 > close:
        target = float(r1)
        target_type = "Resistance (R1)"
    else:
        tp_mult = cfg.get("take_profit_mult", 2.0)
        target = close + (tp_mult * cfg["atr_stop_mult"] * atrv)
        target_type = f"ATR-based Target ({tp_mult}x)"

    reward = target - close

    if risk <= 0 or reward <= 0:
        return {"rr": None, "stop": stop, "risk": risk, "reward": reward, "target_type": target_type}

    rr_val = float(reward / risk) if reward is not None else None
    
    return {
        "rr": rr_val, 
        "stop": float(stop), 
        "risk": float(risk), 
        "reward": reward, 
        "target_type": target_type
    }


def fmt_rr(rr):
    if rr is None or (isinstance(rr, float) and (not np.isfinite(rr))):
        return "N/A"
    return f"1:{rr:.2f}"


def pct_dist(level: float, base: float):
    if level is None or not np.isfinite(level) or base == 0:
        return None
    return (level / base - 1.0) * 100.0


# =============================
# Cached data loader
# =============================
@st.cache_data(ttl=PRICE_CACHE_TTL_SECONDS, show_spinner=False)
def load_data_cached(ticker: str, period: str, interval: str, end_date=None, force_latest: bool = False) -> pd.DataFrame:
    if end_date is not None:
        import datetime
        bitis_obj = end_date + datetime.timedelta(days=1)
        bitis_str = bitis_obj.strftime('%Y-%m-%d')
        
        if period == "45d":
            baslangic_obj = end_date - datetime.timedelta(days=45)
        elif period == "3mo":
            baslangic_obj = end_date - datetime.timedelta(days=90)
        elif period == "6mo":
            baslangic_obj = end_date - datetime.timedelta(days=180)
        elif period == "1y":
            baslangic_obj = end_date - datetime.timedelta(days=365)
        elif period == "2y":
            baslangic_obj = end_date - datetime.timedelta(days=730)
        elif period == "5y":
            baslangic_obj = end_date - datetime.timedelta(days=365 * 5)
        elif period == "10y":
            baslangic_obj = end_date - datetime.timedelta(days=365 * 10)
        else:
            baslangic_obj = end_date - datetime.timedelta(days=730)
            
        baslangic_str = baslangic_obj.strftime('%Y-%m-%d')
        
        df = yf.download(ticker, start=baslangic_str, end=bitis_str, interval=interval, auto_adjust=True, progress=False)
    else:
        df = yf.download(ticker, period=period, interval=interval, auto_adjust=True, progress=False)
        
    df = _flatten_yf(df)

    if force_latest and end_date is None and interval == "1d" and not df.empty:
        try:
            today_data = yf.download(ticker, period="1d", interval="1m", progress=False)
            today_data = _flatten_yf(today_data)
            
            if not today_data.empty:
                today_date = today_data.index[-1].date()
                last_df_date = df.index[-1].date()
                
                if today_date > last_df_date:
                    v = float(today_data["Volume"].sum())
                    if v > 0:
                        o = float(today_data["Open"].iloc[0])
                        h = float(today_data["High"].max())
                        l = float(today_data["Low"].min())
                        c = float(today_data["Close"].iloc[-1])
                        
                        new_idx = pd.to_datetime(str(today_date))
                        if df.index.tz is not None:
                            new_idx = new_idx.tz_localize(df.index.tz)
                            
                        new_row = pd.DataFrame({
                            "Open": [o],
                            "High": [h],
                            "Low": [l],
                            "Close": [c],
                            "Volume": [v]
                        }, index=[new_idx])
                        
                        df = pd.concat([df, new_row])
        except Exception:
            pass
            
    return df




# =============================
# BIST TL / USD Price Conversion Helpers
# =============================
@st.cache_data(ttl=PRICE_CACHE_TTL_SECONDS, show_spinner=False)
def load_usdtry_cached(period: str, interval: str, end_date=None) -> pd.DataFrame:
    """
    Yahoo Finance'ta TRY=X çoğunlukla USD/TRY kurudur.
    BIST TL fiyatlarını USD'ye çevirmek için OHLC fiyatları USDTRY'ye böleriz.
    """
    fx_interval = interval
    # Yahoo bazı sembollerde 4h interval'i desteklemeyebilir; 1h FX verisiyle ffill daha güvenli olur.
    if fx_interval == "4h":
        fx_interval = "1h"

    try:
        fx = load_data_cached("TRY=X", period, fx_interval, end_date=end_date, force_latest=False)
        fx = _flatten_yf(fx)
    except Exception:
        fx = pd.DataFrame()

    if fx is None or fx.empty:
        try:
            fx = load_data_cached("TRY=X", period, "1d", end_date=end_date, force_latest=False)
            fx = _flatten_yf(fx)
        except Exception:
            fx = pd.DataFrame()

    return fx if fx is not None else pd.DataFrame()


def _make_naive_datetime_index(idx) -> pd.DatetimeIndex:
    out_idx = pd.to_datetime(idx, errors="coerce")
    try:
        if getattr(out_idx, "tz", None) is not None:
            out_idx = out_idx.tz_localize(None)
    except Exception:
        try:
            out_idx = out_idx.tz_convert(None)
        except Exception:
            pass
    return pd.DatetimeIndex(out_idx)


def convert_bist_ohlcv_to_usd(df_tl: pd.DataFrame, period: str, interval: str, end_date=None) -> Tuple[pd.DataFrame, float, str]:
    """
    BIST OHLC verisini TL'den USD'ye çevirir.
    Hacim lot/adet olduğu için değişmez; Open/High/Low/Close/Adj Close USDTRY'ye bölünür.
    Dönen: (usd_df, son_kur, hata_mesajı)
    """
    if df_tl is None or df_tl.empty:
        return pd.DataFrame(), np.nan, "BIST fiyat verisi boş."

    fx = load_usdtry_cached(period, interval, end_date=end_date)
    if fx is None or fx.empty or "Close" not in fx.columns:
        return df_tl.copy(), np.nan, "USDTRY verisi alınamadı; TL veriler korunuyor."

    out = df_tl.copy()
    fx_close = fx["Close"].copy()

    try:
        out.index = _make_naive_datetime_index(out.index)
        fx_close.index = _make_naive_datetime_index(fx_close.index)
        out = out[out.index.notna()]
        fx_close = fx_close[fx_close.index.notna()]
    except Exception:
        return df_tl.copy(), np.nan, "Tarih hizalama sırasında USD dönüşümü yapılamadı; TL veriler korunuyor."

    fx_close = pd.to_numeric(fx_close, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    fx_close = fx_close[fx_close > 0]

    if fx_close.empty:
        return df_tl.copy(), np.nan, "USDTRY kapanış verisi geçersiz; TL veriler korunuyor."

    fx_aligned = fx_close.reindex(out.index).ffill().bfill()

    if fx_aligned.isna().all():
        # Günlük FX ile saatlik/hisse index'i birebir çakışmazsa tarih bazlı ikinci hizalama dene.
        try:
            fx_daily = fx_close.copy()
            fx_daily.index = pd.to_datetime(fx_daily.index).normalize()
            tmp_dates = pd.Series(pd.to_datetime(out.index).normalize(), index=out.index)
            fx_aligned = tmp_dates.map(fx_daily.groupby(fx_daily.index).last()).ffill().bfill()
        except Exception:
            pass

    fx_aligned = pd.to_numeric(fx_aligned, errors="coerce").replace([np.inf, -np.inf], np.nan)
    if fx_aligned.isna().all():
        return df_tl.copy(), np.nan, "USDTRY verisi fiyat tarihleriyle hizalanamadı; TL veriler korunuyor."

    fx_aligned = fx_aligned.ffill().bfill()
    last_fx = safe_float(fx_aligned.iloc[-1])

    price_cols = [c for c in ["Open", "High", "Low", "Close", "Adj Close"] if c in out.columns]
    for col in price_cols:
        out[col] = pd.to_numeric(out[col], errors="coerce") / fx_aligned.values

    out["USDTRY"] = fx_aligned.values

    return out, last_fx, ""


# =============================
# Dashboard Strong Buy Scan Helpers (1W / 2Y)
# =============================
def _dashboard_recommendation_from_latest(latest_row: pd.Series) -> str:
    try:
        score_val = safe_float(latest_row.get("SCORE", np.nan))
        entry_val = int(latest_row.get("ENTRY", 0)) if pd.notna(latest_row.get("ENTRY", np.nan)) else 0
        exit_val = int(latest_row.get("EXIT", 0)) if pd.notna(latest_row.get("EXIT", np.nan)) else 0

        # Dashboard ana karar metniyle birebir aynı öncelik:
        # ENTRY varsa AL, EXIT varsa SAT, aksi halde skor >= 80 ise AL (Güçlü Trend).
        if entry_val == 1:
            return "AL"
        elif exit_val == 1:
            return "SAT"
        else:
            return "AL (Güçlü Trend)" if score_val >= 80 else ("İZLE (Orta)" if score_val >= 60 else "UZAK DUR")
    except Exception as e:
        log_app_error("_dashboard_recommendation_from_latest", e, level="DEBUG")
        return "N/A"


def dashboard_strong_buy_scan_one(
    scan_symbol: str,
    selected_market: str,
    cfg_in: dict,
    bist_currency_mode: str = "TL",
) -> Dict[str, Any]:
    """
    Tarama sekmesinde kullanılmak üzere tek sembol için Dashboard mantığını 1wk / 2y üzerinde çalıştırır.
    Mevcut Dashboard skor motoru olan build_features + signal_with_checkpoints kullanılır.
    """
    symbol_norm = normalize_ticker(scan_symbol, selected_market)
    out: Dict[str, Any] = {
        "Sembol": symbol_norm,
        "Periyot": "2y",
        "Interval": "1wk",
        "Para Birimi": "USD" if selected_market == "BIST" and bist_currency_mode == "USD" else ("TL" if selected_market == "BIST" else "USD"),
    }

    try:
        raw = load_data_cached(symbol_norm, "2y", "1wk", end_date=None, force_latest=False)
        raw = _flatten_yf(raw)

        if raw is None or raw.empty:
            out["Hata"] = "Veri yok"
            return out

        if selected_market == "BIST" and bist_currency_mode == "USD":
            raw, usdtry_last_local, usd_err = convert_bist_ohlcv_to_usd(raw, "2y", "1wk", end_date=None)
            out["USDTRY"] = usdtry_last_local
            if usd_err:
                out["Hata"] = usd_err
                return out

        required_cols = {"Open", "High", "Low", "Close", "Volume"}
        if not required_cols.issubset(set(raw.columns)):
            out["Hata"] = "OHLCV eksik"
            return out

        if len(raw) < 60:
            out["Hata"] = "Yetersiz haftalık veri"
            return out

        feat = build_features(raw, cfg_in)
        feat, _ = signal_with_checkpoints(feat, cfg_in)

        if feat is None or feat.empty:
            out["Hata"] = "Feature üretilemedi"
            return out

        latest_local = feat.iloc[-1]
        rec = _dashboard_recommendation_from_latest(latest_local)

        close_val = safe_float(latest_local.get("Close", np.nan))
        score_val = safe_float(latest_local.get("SCORE", np.nan))
        rsi_val = safe_float(latest_local.get("RSI", np.nan))
        macd_hist_val = safe_float(latest_local.get("MACD_hist", np.nan))
        atr_pct_val = safe_float(latest_local.get("ATR_PCT", np.nan))
        volume_ratio_val = safe_float(latest_local.get("Volume_Ratio", np.nan))
        ema50_val = safe_float(latest_local.get("EMA50", np.nan))
        ema200_val = safe_float(latest_local.get("EMA200", np.nan))
        bb_mid_val = safe_float(latest_local.get("BB_mid", np.nan))
        obv_val = safe_float(latest_local.get("OBV", np.nan))
        obv_ema_val = safe_float(latest_local.get("OBV_EMA", np.nan))

        out.update({
            "Dashboard Sinyali": rec,
            "Skor": score_val,
            "Son Kapanış": close_val,
            "RSI": rsi_val,
            "MACD Hist": macd_hist_val,
            "ATR%": atr_pct_val,
            "Hacim Oranı": volume_ratio_val,
            "Trend OK": bool(pd.notna(ema50_val) and pd.notna(ema200_val) and pd.notna(close_val) and close_val > ema200_val and ema50_val > ema200_val),
            "BB OK": bool(pd.notna(bb_mid_val) and pd.notna(close_val) and close_val > bb_mid_val),
            "OBV OK": bool(pd.notna(obv_val) and pd.notna(obv_ema_val) and obv_val > obv_ema_val),
            "ENTRY": int(latest_local.get("ENTRY", 0)) if pd.notna(latest_local.get("ENTRY", np.nan)) else 0,
            "EXIT": int(latest_local.get("EXIT", 0)) if pd.notna(latest_local.get("EXIT", np.nan)) else 0,
            "Son Bar Tarihi": str(pd.to_datetime(feat.index[-1]).date()),
            "Hata": "",
        })
        return out

    except Exception as e:
        log_app_error("dashboard_strong_buy_scan_one", e, {"ticker": scan_symbol, "market": selected_market}, level="DEBUG")
        out["Hata"] = str(e)[:180]
        return out


def dashboard_strong_buy_scan_many(
    symbols: List[str],
    selected_market: str,
    cfg_in: dict,
    bist_currency_mode: str = "TL",
    max_workers: int = 6,
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    clean_symbols = []
    seen_symbols = set()

    for sym in symbols or []:
        sym_norm = normalize_ticker(str(sym).strip().upper(), selected_market)
        key = naked_ticker(sym_norm)
        if key and key not in seen_symbols:
            seen_symbols.add(key)
            clean_symbols.append(sym_norm)

    if not clean_symbols:
        return pd.DataFrame()

    worker_count = int(max(1, min(max_workers, len(clean_symbols))))

    try:
        with ThreadPoolExecutor(max_workers=worker_count) as ex:
            futures = {
                ex.submit(dashboard_strong_buy_scan_one, sym, selected_market, cfg_in, bist_currency_mode): sym
                for sym in clean_symbols
            }
            for fut in as_completed(futures):
                try:
                    rows.append(fut.result())
                except Exception as e:
                    sym = futures.get(fut, "")
                    log_app_error("dashboard_strong_buy_scan_many.future", e, {"ticker": sym}, level="DEBUG")
                    rows.append({"Sembol": sym, "Hata": str(e)[:180]})
    except Exception as e:
        log_app_error("dashboard_strong_buy_scan_many.threadpool", e, {"count": len(clean_symbols)}, level="DEBUG")
        for sym in clean_symbols:
            rows.append(dashboard_strong_buy_scan_one(sym, selected_market, cfg_in, bist_currency_mode))

    df_out = pd.DataFrame(rows)
    if df_out.empty:
        return df_out

    if "Skor" in df_out.columns:
        df_out["Skor"] = pd.to_numeric(df_out["Skor"], errors="coerce")
    if "ATR%" in df_out.columns:
        df_out["ATR%"] = pd.to_numeric(df_out["ATR%"], errors="coerce")
    if "Hacim Oranı" in df_out.columns:
        df_out["Hacim Oranı"] = pd.to_numeric(df_out["Hacim Oranı"], errors="coerce")

    return df_out



# =============================
# UI STATE
# =============================
st.title("📈 FA→TA Trading Uygulaması + 🤖 AI Analiz")
st.caption("Önce fundamental ile evreni daralt, sonra teknik analizle giriş/çıkış zamanla. Otomatik emir göndermez.")

if "app_errors" not in st.session_state:
    st.session_state.app_errors = []

if "screener_df" not in st.session_state:
    st.session_state.screener_df = pd.DataFrame()
if "selected_ticker" not in st.session_state:
    st.session_state.selected_ticker = None
if "ta_ran" not in st.session_state:
    st.session_state.ta_ran = False
if "gemini_text" not in st.session_state:
    st.session_state.gemini_text = ""
if "pa_pack" not in st.session_state:
    st.session_state.pa_pack = {}
if "sentiment_summary" not in st.session_state:
    st.session_state.sentiment_summary = ""
if "sentiment_items" not in st.session_state:
    st.session_state.sentiment_items = []

if "show_ema13_channel" not in st.session_state:
    st.session_state.show_ema13_channel = False

if "show_sr_lines_dashboard" not in st.session_state:
    st.session_state.show_sr_lines_dashboard = False

if "show_sr_lines_index_center" not in st.session_state:
    st.session_state.show_sr_lines_index_center = False

if "divergence_scan_results" not in st.session_state:
    st.session_state.divergence_scan_results = pd.DataFrame()

if "dashboard_strong_buy_scan_results" not in st.session_state:
    st.session_state.dashboard_strong_buy_scan_results = pd.DataFrame()


if "ai_messages" in st.session_state:
    del st.session_state.ai_messages


# =============================
# Sidebar
# =============================
with st.sidebar:
    st.header("Piyasa")
    market = st.selectbox(
        "Market",
        ["USA", "BIST"],
        index=0,
        help="Analiz edilecek borsayı seçin. USA için ABD hisseleri, BIST için Borsa İstanbul hisseleri.",
    )

    if "last_market" not in st.session_state:
        st.session_state.last_market = market
    elif st.session_state.last_market != market:
        st.session_state.screener_df = pd.DataFrame()
        st.session_state.selected_ticker = None
        st.session_state.last_market = market

    bist_price_currency = "TL"
    if market == "BIST":
        bist_price_currency = st.radio(
            "BIST fiyat para birimi",
            ["TL", "USD"],
            index=0,
            horizontal=True,
            help="TL seçiliyse BIST verileri normal TL fiyatlarla hesaplanır. USD seçiliyse OHLC fiyatları USDTRY kuruna bölünür ve Dashboard, indikatörler, osilatörler, backtest, Future Price ve diğer teknik analizler USD bazlı fiyat serisi üzerinden çalışır.",
        )

    usa_bucket = None
    if market == "USA":
        usa_bucket = st.selectbox(
            "USA Universe",
            ["S&P 500", "Nasdaq 100"],
            index=0,
            help="Hangi endeksteki hisseleri tarayacağınızı seçin. S&P 500 daha geniş, Nasdaq 100 teknoloji ağırlıklı.",
        )

    st.header("1) Fundamental Screener")
    use_fa = st.checkbox(
        "Fundamental filtreyi kullan",
        value=True,
        help="Temel analiz (FA) kurallarını aktifleştirir. Şirketlerin mali tablolarını değerlendirerek puanlar.",
    )
    fa_mode = st.selectbox(
        "Fundamental Mod",
        ["Quality", "Value", "Growth"],
        index=0,
        disabled=(not use_fa),
        help="Quality: Karlılık ve düşük borç arar. Value: Ucuz kalmış hisseleri bulur. Growth: Yüksek büyüme oranlarına odaklanır.",
    )

    st.caption("Eşikler (Genel) — BIST'te coverage düşük olabilir")
    roe = st.slider(
        "ROE min",
        0.0,
        0.40,
        0.15,
        0.01,
        disabled=(not use_fa),
        help="Özkaynak Karlılığı (Return on Equity). Şirketin öz sermayesini ne kadar verimli kullandığını gösterir. Yüksek ROE genellikle iyidir.",
    )
    op_margin = st.slider(
        "Operating Margin min",
        0.0,
        0.40,
        0.10,
        0.01,
        disabled=(not use_fa),
        help="Faaliyet Kar Marjı. Şirketin ana faaliyetlerinden elde ettiği kârlılık. Sektör ortalamasıyla karşılaştırın.",
    )
    profit_margin = st.slider(
        "Profit Margin min",
        0.0,
        0.40,
        0.08,
        0.01,
        disabled=(not use_fa),
        help="Net Kar Marjı. Tüm giderler ve vergiler düşüldükten sonra kalan net kârın satışlara oranı.",
    )
    dte = st.slider(
        "Debt/Equity max",
        0.0,
        3.0,
        1.0,
        0.05,
        disabled=(not use_fa),
        help="Borç/Özkaynak oranı. Şirketin ne kadar borçlu olduğunu gösterir. 1.0 altı genellikle güvenli kabul edilir.",
    )
    fpe = st.slider(
        "Forward P/E max",
        0.0,
        60.0,
        20.0,
        1.0,
        disabled=(not use_fa),
        help="İleri F/K (Fiyat/Kazanç) oranı. Gelecek yıl beklenen kazançlara göre hissenin ucuz/pahalı olduğunu gösterir.",
    )
    peg = st.slider(
        "PEG max",
        0.0,
        5.0,
        1.5,
        0.1,
        disabled=(not use_fa),
        help="F/K / Büyüme oranı. 1.0 civarı hissenin büyüme potansiyeline göre adil fiyatlandığını gösterir.",
    )
    ps = st.slider(
        "P/S max",
        0.0,
        30.0,
        6.0,
        0.5,
        disabled=(not use_fa),
        help="Fiyat/Satış oranı. Henüz kâr etmeyen ancak satışları büyük şirketler için kullanılır.",
    )
    pb = st.slider(
        "P/B max",
        0.0,
        30.0,
        6.0,
        0.5,
        disabled=(not use_fa),
        help="Piyasa Değeri / Defter Değeri. Şirketin net varlıklarına göre kaç katından işlem gördüğü.",
    )
    rev_g = st.slider(
        "Revenue Growth min",
        0.0,
        0.50,
        0.10,
        0.01,
        disabled=(not use_fa),
        help="Yıllık ciro büyümesi. Şirketin satışlarının ne kadar arttığını gösterir.",
    )
    earn_g = st.slider(
        "Earnings Growth min",
        0.0,
        0.50,
        0.10,
        0.01,
        disabled=(not use_fa),
        help="Yıllık kâr büyümesi. Net kârdaki artış oranı.",
    )

    min_score = st.slider(
        "Min Fundamental Score",
        0,
        100,
        60,
        1,
        disabled=(not use_fa),
        help="Ağırlıklı puanın alt limiti. Bu puanın üzerindeki hisseler 'PASS' olarak işaretlenir.",
    )
    min_ok = st.slider(
        "Min OK sayısı",
        1,
        5,
        3,
        1,
        disabled=(not use_fa),
        help="Başarılı kriter sayısı. En az bu kadar kriteri sağlamalı.",
    )
    min_coverage = st.slider(
        "Min Coverage (NaN olmayan)",
        1,
        5,
        3,
        1,
        disabled=(not use_fa),
        help="Verisi olan kriter sayısı. BIST'te veri eksikliği olabileceği için bu sayı düşük tutulabilir.",
    )

    thresholds = {
        "roe": roe,
        "op_margin": op_margin,
        "profit_margin": profit_margin,
        "dte": dte,
        "fpe": fpe,
        "peg": peg,
        "ps": ps,
        "pb": pb,
        "rev_g": rev_g,
        "earn_g": earn_g,
        "min_score": min_score,
        "min_ok": min_ok,
        "min_coverage": min_coverage,
    }

    if market == "USA":
        if usa_bucket == "S&P 500":
            universe = load_universe_file(pjoin("universes", "sp500.txt"))
        else:
            universe = load_universe_file(pjoin("universes", "nasdaq100.txt"))
        st.caption(f"Universe: {usa_bucket} (count: {len(universe)})")
    else:
        universe = load_universe_file(pjoin("universes", "bist100.txt"))
        st.caption(f"Universe: BIST100 (count: {len(universe)})")

    if not universe:
        st.error("Universe listesi boş!")
        st.stop()

    run_screener = st.button("🔎 Screener Çalıştır", type="secondary", disabled=(not use_fa))

    st.divider()
    st.header("2) Teknik Analiz + Backtest")
    preset_name = st.selectbox(
        "Teknik Mod",
        list(PRESETS.keys()),
        index=1,
        help="Önceden tanımlı risk profilleri. Defansif: düşük risk, Agresif: yüksek risk.",
    )

    st.subheader("Sembol (TA)")
    if st.session_state.selected_ticker:
        st.caption(f"Screener seçimi: **{st.session_state.selected_ticker}**")
        raw_ticker = st.text_input("Sembol", value=st.session_state.selected_ticker)
    else:
        raw_ticker = st.text_input(
            "Sembol (USA: AAPL, SPY) / BIST: THYAO",
            value="AAPL" if market == "USA" else "THYAO",
            help="Analiz etmek istediğiniz hisse senedinin sembolü. BIST için otomatik .IS eklenir.",
        )

    ticker = normalize_ticker(raw_ticker, market)

    st.subheader("Zaman Aralığı")
    interval = st.selectbox(
        "Interval",
        ["1d", "1wk", "4h", "1h"],
        index=0,
        help="Mum zaman dilimi. 1d günlük, 1wk haftalık, 4h 4 saatlik, 1h saatlik analiz için. Backtest için 1d önerilir.",
    )
    period = st.selectbox(
        "Periyot",
        ["45d", "3mo", "6mo", "1y", "2y"],
        index=3,
        help="Verinin ne kadar geriye gidileceği. Daha uzun periyot daha sağlıklı backtest sağlar.",
    )

    use_custom_end_date = st.checkbox("Geçmiş Bir Tarihe Göre Analiz Yap (Repaint Önleme)", value=False)
    if use_custom_end_date:
        bugun = datetime.date.today()
        gecen_cuma = bugun - datetime.timedelta(days=bugun.weekday() + 3)
        
        custom_end_date = st.date_input(
            "Bitiş Tarihi Seçin", 
            value=gecen_cuma,
            help="Seçtiğiniz tarihe kadar olan veriler çekilir. Haftalık kapanışlar için Cuma gününü seçin."
        )
    else:
        custom_end_date = None

    force_latest_candle = st.checkbox(
        "Eksik Güncel Mumu Zorla Ekle (Live Candle Hack)", 
        value=False, 
        disabled=(use_custom_end_date or interval != "1d"),
        help="Yahoo Finance günlük mumu henüz vermediyse gün içi dakikalık verilerden o mumu inşa eder. (Sadece günlük periyotta ve güncel analizde çalışır)"
    )

    use_live_last_override = st.checkbox(
        "Base ve Daily Close için Live/Last kullan + hesapları son bara uygula",
        value=False,
        disabled=use_custom_end_date,
        help="Açıldığında son barın Close/High/Low değerleri Live/Last fiyatına göre güncellenir. Böylece Base, Daily Close, mum formasyonları, RSI/MACD/ATR, Stochastic RSI, Bollinger, hacim oranı, 3 Ekranlı Sistem ve Future Price hesapları Live/Last baz alınarak yeniden hesaplanır."
    )

    st.divider()
    st.subheader("Teknik Parametreler")
    ema_fast = st.number_input(
        "EMA Fast",
        min_value=5,
        max_value=100,
        value=50,
        step=1,
        help="Kısa vadeli üstel hareketli ortalama. Fiyatın bu ortalamanın üstünde olması kısa vadeli yükseliş trendini gösterir.",
    )
    ema_slow = st.number_input(
        "EMA Slow",
        min_value=50,
        max_value=400,
        value=200,
        step=1,
        help="Uzun vadeli üstel hareketli ortalama. Fiyat bu ortalamanın üstündeyse ana trend yükseliş, altındaysa düşüş trendi.",
    )
    rsi_period = st.number_input(
        "RSI Period",
        min_value=5,
        max_value=30,
        value=14,
        step=1,
        help="RSI hesaplama periyodu. 14 gün standarttır. 70 üstü aşırı alım, 30 altı aşırı satım.",
    )
    bb_period = st.number_input(
        "Bollinger Period",
        min_value=10,
        max_value=50,
        value=20,
        step=1,
        help="Bollinger bandı ortalama periyodu. Fiyatın üst banda yaklaşması aşırı alım, alt banda yaklaşması aşırı satım.",
    )
    bb_std = st.number_input(
        "Bollinger Std",
        min_value=1.0,
        max_value=3.5,
        value=2.0,
        step=0.1,
        help="Bollinger bandı standart sapma katsayısı. 2 standart sapma %95 güven aralığı verir.",
    )
    atr_period = st.number_input(
        "ATR Period",
        min_value=5,
        max_value=30,
        value=14,
        step=1,
        help="Ortalama Gerçek Aralık (Average True Range) periyodu. Volatilitenin ölçüsü, stop seviyesi belirlemede kullanılır.",
    )
    vol_sma = st.number_input(
        "Volume SMA",
        min_value=5,
        max_value=60,
        value=20,
        step=1,
        help="Hacim basit hareketli ortalaması. Hacim bu ortalamanın üzerindeyse likidite yüksek, işlem anlamlıdır.",
    )

    st.subheader("Market Filtreleri")
    use_spy_filter = st.checkbox(
        "SPY > EMA200 filtresi (Sadece USA)",
        value=True,
        disabled=(market != "USA"),
        help="S&P 500 endeksi 200 günlük ortalamanın altındaysa (ayı piyasası) alım sinyallerini engeller.",
    )
    use_bist_filter = st.checkbox(
        "XU100 > EMA200 filtresi (Sadece BIST)",
        value=True,
        disabled=(market != "BIST"),
        help="BIST 100 endeksi 200 günlük ortalamanın altındaysa alım sinyallerini engeller.",
    )
    use_higher_tf_filter = st.checkbox(
        "Haftalık trend filtresi (Fiyat > EMA200)",
        value=True,
        help="Haftalık grafikte fiyatın 200 haftalık ortalamanın üzerinde olması gerekir. Ana trendin yükseliş olduğunu onaylar.",
    )

    st.subheader("Risk / Backtest Ayarları")
    initial_capital = st.number_input(
        "Başlangıç Sermayesi",
        min_value=100.0,
        value=10000.0,
        step=500.0,
        help="Backtest için simüle edilecek başlangıç parası.",
    )
    risk_per_trade = st.slider(
        "Trade başı risk (equity %)",
        min_value=0.002,
        max_value=0.05,
        value=0.01,
        step=0.001,
        help="Her işlemde kasanın yüzde kaçını riske edeceğiniz. Stop loss ile kaybedilecek maksimum miktar.",
    )
    commission_bps = st.number_input(
        "Komisyon (bps)",
        min_value=0.0,
        value=5.0,
        step=1.0,
        help="İşlem başına komisyon (baz puan). 1 bps = %0.01.",
    )
    slippage_bps = st.number_input(
        "Slippage (bps)",
        min_value=0.0,
        value=2.0,
        step=1.0,
        help="Kayma maliyeti. Sinyal fiyatından daha kötü fiyattan işlem gerçekleşme riski.",
    )
    risk_free_annual = st.number_input(
        "Risk-Free (yıllık)",
        min_value=0.0,
        value=0.0,
        step=0.01,
        help="Risksiz faiz oranı (örnek: 0.05 = %5). Sharpe ve Sortino hesaplamalarında kullanılır.",
    )

    st.divider()
    st.header("3) AI Ayarları (Gemini)")
    ai_on = st.checkbox(
        "Gemini AI aktif",
        value=True,
        help="Google Gemini AI ile grafik ve veri analizi yapılır.",
    )
    gemini_model = st.selectbox(
        "Gemini Model",
        options=[
            "gemini-3.5-flash",
            "gemini-flash-latest",
            "gemini-3.1-pro",
            "gemini-3-flash",
            "gemini-3.1-flash-lite",
            "gemini-2.5-pro",
            "gemini-2.5-flash",
            "gemini-2.5-flash-lite",
        ],
        index=0,
        help="Kullanılacak Gemini modeli. Stable modeller daha tutarlı; gemini-flash-latest ise Google yeni Flash sürümü yayınladığında otomatik güncellenebilir.",
    )
    gemini_temp = st.slider(
        "Temperature",
        0.0,
        1.0,
        0.2,
        0.05,
        help="Modelin yaratıcılığı. Düşük değerler daha tutarlı, yüksek değerler daha yaratıcı cevaplar üretir.",
    )
    gemini_max_tokens = st.slider(
        "Max Output Tokens",
        256,
        8192,
        2048,
        128,
        help="Modelin üreteceği maksimum token sayısı. Daha uzun cevaplar için artırın.",
    )

    st.divider()
    st.header("4) Haber Duygu Analizi (Google News + Gemini)")
    use_sentiment = st.checkbox(
        "Haber duygu analizini aktifleştir",
        value=True,
        help="Google News'ten haber başlıklarını çeker, Gemini ile duygu analizi yapar.",
    )

    run_btn = st.button("🚀 Teknik Analizi Çalıştır", type="primary")
    if run_btn:
        st.session_state.ta_ran = True

# -----------------------------
# Config
# -----------------------------
cfg = {
    "ema_fast": ema_fast,
    "ema_slow": ema_slow,
    "rsi_period": rsi_period,
    "bb_period": bb_period,
    "bb_std": bb_std,
    "atr_period": atr_period,
    "vol_sma": vol_sma,
    "initial_capital": initial_capital,
    "risk_per_trade": risk_per_trade,
    "commission_bps": commission_bps,
    "slippage_bps": slippage_bps,
}
cfg.update(PRESETS[preset_name])

update_app_config(
    market=market,
    ticker=ticker,
    interval=interval,
    period=period,
    preset_name=preset_name,
    use_live_last_override=use_live_last_override,
    cfg=cfg.copy(),
)


# -----------------------------
# Fundamental screener action
# -----------------------------
if run_screener and use_fa:
    with st.spinner(f"Fundamental veriler çekiliyor ({market})... (Bu işlem çoklu iş parçacığıyla hızlandırılmıştır)"):
        rows = []
        
        def fetch_one(tk):
            try:
                tk_norm = normalize_ticker(tk, market)
                f = fetch_fundamentals_generic(tk_norm, market=market)
                score, breakdown, passed = fundamental_score_row(f, fa_mode, thresholds)
                f["FA_score"] = score
                f["FA_pass"] = passed
                f["FA_ok_count"] = sum(1 for v in breakdown.values() if v.get("available") and v.get("ok"))
                f["FA_coverage"] = sum(1 for v in breakdown.values() if v.get("available"))
                return f
            except Exception as e:
                log_app_error("fundamental_screener.fetch_one", e, {"ticker": tk}, level="DEBUG")
                return {"ticker": tk, "error": f"{str(e)}"}

        with ThreadPoolExecutor(max_workers=10) as ex:
            futures = {ex.submit(fetch_one, tk): tk for tk in universe}
            for future in as_completed(futures):
                try:
                    res = future.result()
                    if "error" in res:
                        st.session_state.app_errors.append(f"FA Hatası ({res.get('ticker','')}): {res['error']}")
                    else:
                        rows.append(res)
                except Exception as e:
                    log_app_error("fundamental_screener.thread", e, {"ticker": futures.get(future, "")})
                    st.session_state.app_errors.append(f"Thread Hatası: {str(e)}")

        sdf = pd.DataFrame(rows)
        if not sdf.empty:
            sdf["FA_pass_int"] = sdf["FA_pass"].astype(int)
            sdf = sdf.sort_values(["FA_pass_int", "FA_score", "FA_coverage"], ascending=[False, False, False]).drop(columns=["FA_pass_int"])
        st.session_state.screener_df = sdf.copy()


# -----------------------------
# If TA not ran yet: show screener and stop
# -----------------------------
if not st.session_state.ta_ran:
    # Hata Gösterimi
    if "app_errors" in st.session_state and st.session_state.app_errors:
        for err in st.session_state.app_errors:
            st.error(f"⚠️ {err}")
        st.session_state.app_errors = []

    if use_fa and not st.session_state.screener_df.empty:
        st.subheader(f"🧾 Fundamental Screener Sonuçları ({market})")
        sdf = st.session_state.screener_df.copy()

        show_cols = [
            "ticker",
            "longName",
            "FA_pass",
            "FA_score",
            "FA_ok_count",
            "FA_coverage",
            "sector",
            "industry",
            "trailingPE",
            "forwardPE",
            "pegRatio",
            "priceToSalesTrailing12Months",
            "priceToBook",
            "returnOnEquity",
            "operatingMargins",
            "profitMargins",
            "debtToEquity",
            "revenueGrowth",
            "earningsGrowth",
            "marketCap",
        ]
        sdf_show = sdf[[c for c in show_cols if c in sdf.columns]].copy()
        st.dataframe(sdf_show, use_container_width=True, height=360)

        pass_list = sdf.loc[sdf["FA_pass"] == True, "ticker"].tolist()
        if len(pass_list) == 0:
            st.warning("Bu eşiklerle PASS çıkan hisse yok. Eşikleri gevşet / mode değiştir / coverage düşür.")
        else:
            st.success(f"PASS sayısı: {len(pass_list)}")
            picked = st.selectbox("PASS listesinden hisse seç (TA’ya gönder)", pass_list, index=0)
            if st.button("➡️ Seçimi Teknik Analize Aktar"):
                st.session_state.selected_ticker = picked
                st.rerun()

    st.info("Sol menüden ayarları yapıp **Teknik Analizi Çalıştır**’a basın.")
    st.stop()


# =============================
# Run TA pipeline
# =============================
market_filter_series = None
if market == "USA" and use_spy_filter:
    with st.spinner("SPY rejimi kontrol ediliyor..."):
        market_filter_series = get_spy_regime_series()
elif market == "BIST" and use_bist_filter:
    with st.spinner("XU100 rejimi kontrol ediliyor..."):
        market_filter_series = get_bist_regime_series()

higher_tf_filter_series = None
if use_higher_tf_filter:
    with st.spinner("Haftalık trend kontrol ediliyor..."):
        higher_tf_filter_series = get_higher_tf_trend_series(ticker, higher_tf_interval="1wk", ema_period=200)

sentiment_summary = ""
if use_sentiment and ai_on:
    company_name = ""
    if not st.session_state.screener_df.empty:
        row = find_screener_row(st.session_state.screener_df, ticker)
        if row and row.get("longName"):
            company_name = row["longName"]

    with st.spinner("Google News'ten haberler çekiliyor ve Gemini ile analiz ediliyor..."):
        sent = get_news_sentiment(ticker, company_name, gemini_model, gemini_temp, gemini_max_tokens)
        if sent.get("error") is None:
            sentiment_summary = sent["summary"]
            st.session_state.sentiment_summary = sentiment_summary
            st.session_state.sentiment_items = sent.get("news_items", [])
        else:
            sentiment_summary = f"Haber analizi başarısız: {sent['error']}"
            st.session_state.sentiment_summary = sentiment_summary
            st.session_state.sentiment_items = []
elif use_sentiment and not ai_on:
    st.warning("Haber duygu analizi için Gemini'nin açık olması gerekir.")


with st.spinner(f"Veri indiriliyor: {ticker}"):
    df_raw = load_data_cached(ticker, period, interval, end_date=custom_end_date, force_latest=force_latest_candle)

live = get_live_price(ticker)
live_price = live.get("last_price", np.nan)

if use_live_last_override:
    df_raw = apply_live_last_override_to_df(df_raw, live_price)

analysis_currency = "USD" if market == "USA" else "TL"
usdtry_last = np.nan
if market == "BIST" and bist_price_currency == "USD":
    with st.spinner("BIST fiyatları USD bazına çevriliyor (USDTRY)..."):
        df_raw, usdtry_last, usd_err = convert_bist_ohlcv_to_usd(df_raw, period, interval, end_date=custom_end_date)
    if usd_err:
        st.warning(usd_err)
    else:
        analysis_currency = "USD"
        if np.isfinite(live_price) and np.isfinite(usdtry_last) and usdtry_last > 0:
            live_price = live_price / usdtry_last
        st.caption(f"💵 BIST USD modu aktif — fiyat serisi USD bazlıdır. Kullanılan son USDTRY: {usdtry_last:.4f}")

if df_raw.empty:
    st.session_state.app_errors.append(f"Veri çekilemedi: {ticker} (Bölünme/delist veya API engeli olabilir)")
    st.error(f"Veri gelmedi: {ticker}")
    st.stop()

required_cols = {"Open", "High", "Low", "Close", "Volume"}
if not required_cols.issubset(set(df_raw.columns)):
    st.session_state.app_errors.append(f"Eksik veri sütunları var (Örn: Hacim verisi yok): {ticker}")
    st.error("Veri setinde gerekli OHLCV kolonları eksik.")
    st.stop()

if len(df_raw) < 260 and interval == "1d":
    st.warning("Günlükte 260 bar altı: metrikler daha oynak olabilir.")

df = build_features(df_raw, cfg)

benchmark_ticker = "SPY" if market == "USA" else "XU100.IS"
benchmark_df = load_data_cached(benchmark_ticker, period, interval, end_date=custom_end_date, force_latest=False)
benchmark_returns = benchmark_df["Close"].pct_change().dropna() if not benchmark_df.empty else None

df, checkpoints = signal_with_checkpoints(
    df,
    cfg,
    market_filter_series=market_filter_series,
    higher_tf_filter_series=higher_tf_filter_series,
)
latest = df.iloc[-1]

if int(latest["ENTRY"]) == 1:
    rec = "AL"
elif int(latest["EXIT"]) == 1:
    rec = "SAT"
else:
    rec = "AL (Güçlü Trend)" if latest["SCORE"] >= 80 else ("İZLE (Orta)" if latest["SCORE"] >= 60 else "UZAK DUR")

eq, tdf, metrics = backtest_long_only(df, cfg, risk_free_annual=risk_free_annual, benchmark_returns=benchmark_returns)
tp = target_price_band(df)
rr_info = rr_from_atr_stop(latest, tp, cfg)
overbought_result = detect_speculation(df)

short_info = get_short_info(ticker)
overbought_result["short_percent_float"] = short_info["short_percent_float"]
overbought_result["short_ratio"] = short_info["short_ratio"]

df["EMA13_High"] = ema(df["High"], 13)
df["EMA13_Low"] = ema(df["Low"], 13)
df["EMA13_Close"] = ema(df["Close"], 13)

# =============================
# Build figures
# =============================

if "show_chart_patterns" not in st.session_state:
    st.session_state.show_chart_patterns = True

fig_price = go.Figure()
fig_price.add_trace(go.Candlestick(x=df.index, open=df["Open"], high=df["High"], low=df["Low"], close=df["Close"], name="Price"))
fig_price.add_trace(go.Scatter(x=df.index, y=df["EMA50"], name="EMA Fast"))
fig_price.add_trace(go.Scatter(x=df.index, y=df["EMA200"], name="EMA Slow"))
fig_price.add_trace(go.Scatter(x=df.index, y=df["BB_upper"], name="BB Upper", line=dict(dash="dot")))
fig_price.add_trace(go.Scatter(x=df.index, y=df["BB_mid"], name="BB Mid", line=dict(dash="dot")))
fig_price.add_trace(go.Scatter(x=df.index, y=df["BB_lower"], name="BB Lower", line=dict(dash="dot")))

if st.session_state.show_ema13_channel:
    fig_price.add_trace(go.Scatter(x=df.index, y=df["EMA13_High"], name="13 EMA High", line=dict(color='rgba(255, 165, 0, 0.8)', width=1)))
    fig_price.add_trace(go.Scatter(x=df.index, y=df["EMA13_Low"], name="13 EMA Low", fill='tonexty', fillcolor='rgba(255, 165, 0, 0.2)', line=dict(color='rgba(255, 165, 0, 0.8)', width=1)))
    fig_price.add_trace(go.Scatter(x=df.index, y=df["EMA13_Close"], name="13 EMA Close", line=dict(color='darkorange', width=2)))

entries = df[df["ENTRY"] == 1]
exits = df[df["EXIT"] == 1]
fig_price.add_trace(go.Scatter(x=entries.index, y=entries["Close"], mode="markers", name="ENTRY", marker=dict(symbol="triangle-up", size=10)))
fig_price.add_trace(go.Scatter(x=exits.index, y=exits["Close"], mode="markers", name="EXIT", marker=dict(symbol="triangle-down", size=10)))

if st.session_state.show_chart_patterns:
    bull_patterns = {
        "KANGAROO_BULL": "🟩🦘 LONG KANGURU",
        "PATTERN_HAMMER": "🟩🔨 HAMMER",
        "PATTERN_INV_HAMMER": "🟩🔨 INV HAMMER",
        "PATTERN_ENGULFING_BULL": "🟢 ENGULFING",
        "PATTERN_HARAMI_BULL": "🟢🤰 HARAMI",
        "PATTERN_MARUBOZU_BULL": "🟩 MARUBOZU",
        "PATTERN_TWEEZER_BOTTOM": "🟢✌️ TWEEZER",
        "PATTERN_PIERCING": "🟢🗡️ PIERCING",
        "PATTERN_MORNING_STAR": "🟢🌅 M.STAR",
        "PATTERN_LL_DOJI": "🟢⚖️ LL DOJI",
    }

    bear_patterns = {
        "KANGAROO_BEAR": "🟥🦘 SHORT KANGURU",
        "PATTERN_HANGING_MAN": "🟥🪢 HANGING M.",
        "PATTERN_SHOOTING_STAR": "🟥🌠 S.STAR",
        "PATTERN_ENGULFING_BEAR": "🔴 ENGULFING",
        "PATTERN_HARAMI_BEAR": "🔴🤰 HARAMI",
        "PATTERN_MARUBOZU_BEAR": "🟥 MARUBOZU",
        "PATTERN_TWEEZER_TOP": "🔴✌️ TWEEZER",
        "PATTERN_DARK_CLOUD": "🔴🌩️ D.CLOUD",
        "PATTERN_EVENING_STAR": "🔴🌃 E.STAR"
    }

    bull_texts = pd.Series("", index=df.index)
    bear_texts = pd.Series("", index=df.index)

    for col, name in bull_patterns.items():
        if col in df.columns:
            mask = df[col] == 1
            bull_texts[mask] += name + "<br>"

    for col, name in bear_patterns.items():
        if col in df.columns:
            mask = df[col] == 1
            bear_texts[mask] += name + "<br>"

    bull_texts = bull_texts.str.rstrip("<br>")
    bear_texts = bear_texts.str.rstrip("<br>")

    bull_mask = bull_texts != ""
    if bull_mask.any():
        fig_price.add_trace(go.Scatter(
            x=df.index[bull_mask], y=df["Low"][bull_mask], mode="markers+text", name="Boğa Formasyonları",
            text=bull_texts[bull_mask], textposition="bottom center",
            textfont=dict(color="green", size=10, family="Arial Black"),
            marker=dict(symbol="triangle-up", size=10, color="green", line=dict(width=1, color="DarkSlateGrey"))
        ))

    bear_mask = bear_texts != ""
    if bear_mask.any():
        fig_price.add_trace(go.Scatter(
            x=df.index[bear_mask], y=df["High"][bear_mask], mode="markers+text", name="Ayı Formasyonları",
            text=bear_texts[bear_mask], textposition="top center",
            textfont=dict(color="red", size=10, family="Arial Black"),
            marker=dict(symbol="triangle-down", size=10, color="red", line=dict(width=1, color="DarkSlateGrey"))
        ))

if st.session_state.get("show_sr_lines_dashboard", False):
    fig_price = add_support_resistance_trend_overlays(fig_price, df)

fig_price.update_layout(
    height=600,
    xaxis_rangeslider_visible=False,
    title="Fiyat Grafiği + EMA + Bollinger + Sinyaller & Formasyonlar",
    yaxis_title="Fiyat",
    xaxis_title="Tarih",
)

fig_rsi = go.Figure()
fig_rsi.add_trace(go.Scatter(x=df.index, y=df["RSI"], name="RSI"))
fig_rsi.add_hline(y=70, line_dash="dash", line_color="red", annotation_text="Aşırı Alım")
fig_rsi.add_hline(y=30, line_dash="dash", line_color="green", annotation_text="Aşırı Satım")
fig_rsi.update_layout(height=260, title="RSI (Göreceli Güç Endeksi)", yaxis_title="RSI", xaxis_title="Tarih")

fig_macd = go.Figure()
fig_macd.add_trace(go.Scatter(x=df.index, y=df["MACD"], name="MACD"))
fig_macd.add_trace(go.Scatter(x=df.index, y=df["MACD_signal"], name="Signal"))
fig_macd.add_trace(go.Bar(x=df.index, y=df["MACD_hist"], name="Hist"))
fig_macd.update_layout(height=260, title="MACD (Moving Average Convergence Divergence)", yaxis_title="MACD", xaxis_title="Tarih")

fig_atr = go.Figure()
fig_atr.add_trace(go.Scatter(x=df.index, y=df["ATR_PCT"] * 100, name="ATR%"))
fig_atr.update_layout(height=260, title="ATR % (Ortalama Gerçek Aralık / Fiyat)", yaxis_title="%", xaxis_title="Tarih")

fig_stoch = go.Figure()
if "STOCH_RSI_K" in df.columns and "STOCH_RSI_D" in df.columns:
    fig_stoch.add_trace(go.Scatter(x=df.index, y=df["STOCH_RSI_K"], name="Stochastic RSI K"))
    fig_stoch.add_trace(go.Scatter(x=df.index, y=df["STOCH_RSI_D"], name="Stochastic RSI D"))
    fig_stoch.add_hline(y=80, line_dash="dash", line_color="red", annotation_text="Aşırı Alım")
    fig_stoch.add_hline(y=20, line_dash="dash", line_color="green", annotation_text="Aşırı Satım")
fig_stoch.update_layout(height=260, title="Stochastic RSI (K & D)", yaxis_title="Değer", xaxis_title="Tarih")

fig_bbwidth = go.Figure()
if "BB_WIDTH" in df.columns:
    fig_bbwidth.add_trace(go.Scatter(x=df.index, y=df["BB_WIDTH"] * 100, name="BB % Genişlik"))
    fig_bbwidth.add_hline(y=2, line_dash="dash", line_color="orange", annotation_text="Sıkışma Bölgesi")
fig_bbwidth.update_layout(height=260, title="Bollinger Bandı Genişliği %", yaxis_title="Genişlik %", xaxis_title="Tarih")

fig_volratio = go.Figure()
if "VOL_RATIO" in df.columns:
    fig_volratio.add_trace(go.Bar(x=df.index, y=df["VOL_RATIO"], name="Hacim Oranı"))
    fig_volratio.add_hline(y=1.5, line_dash="dash", line_color="red", annotation_text="Anormal Hacim")
fig_volratio.update_layout(height=260, title="Hacim Oranı (Son Hacim / SMA)", yaxis_title="Oran", xaxis_title="Tarih")

fig_eq = go.Figure()
fig_eq.add_trace(go.Scatter(x=eq.index, y=eq.values, name="Equity"))
fig_eq.update_layout(height=320, title="Backtest Sermaye Eğrisi", yaxis_title="Sermaye", xaxis_title="Tarih")

df["VOL_SMA_10"] = df["Volume"].rolling(10).mean()
if benchmark_df is not None and not benchmark_df.empty:
    bench_vol = benchmark_df["Volume"].reindex(df.index).fillna(0)
else:
    bench_vol = pd.Series(0, index=df.index)

fig_vol_market = make_subplots(specs=[[{"secondary_y": True}]])
fig_vol_market.add_trace(go.Bar(x=df.index, y=df["Volume"], name="Hisse Hacmi", marker_color='lightblue', opacity=0.7), secondary_y=False)
fig_vol_market.add_trace(go.Scatter(x=df.index, y=bench_vol, name=f"Endeks ({benchmark_ticker})", line=dict(color='orange', width=2)), secondary_y=True)
fig_vol_market.update_layout(height=260, title=f"Hisse vs Endeks Hacmi", yaxis_title="Hisse", yaxis2_title="Endeks", xaxis_title="Tarih", margin=dict(l=0, r=0, t=40, b=0))

fig_vol_2wk = go.Figure()
fig_vol_2wk.add_trace(go.Bar(x=df.index, y=df["Volume"], name="Hacim", marker_color='cadetblue', opacity=0.7))
fig_vol_2wk.add_trace(go.Scatter(x=df.index, y=df["VOL_SMA_10"], name="2 Haftalık Ort. (10 Bar)", line=dict(color='red', width=2)))
fig_vol_2wk.update_layout(height=260, title="Hisse Hacmi vs 2 Haftalık Ortalama", yaxis_title="Hacim", xaxis_title="Tarih", margin=dict(l=0, r=0, t=40, b=0))

fig_obv = go.Figure()
if "OBV" in df.columns and "OBV_EMA" in df.columns:
    fig_obv.add_trace(go.Scatter(x=df.index, y=df["OBV"], name="OBV", line=dict(color='blue')))
    fig_obv.add_trace(go.Scatter(x=df.index, y=df["OBV_EMA"], name="OBV EMA (21)", line=dict(color='orange', dash='dot')))
fig_obv.update_layout(height=260, title="On-Balance Volume (OBV)", yaxis_title="OBV", xaxis_title="Tarih", margin=dict(l=0, r=0, t=40, b=0))

fig_vpvr, vp_poc_price = build_volume_profile_figure(df, bins=28, lookback=220)

figs_for_report = {
    "Price + EMA + Bollinger + Signals": fig_price,
    "RSI": fig_rsi,
    "MACD": fig_macd,
    "ATR%": fig_atr,
    "Stochastic RSI": fig_stoch,
    "Bollinger Band Width": fig_bbwidth,
    "Volume Ratio": fig_volratio,
    "Volume vs Market": fig_vol_market,
    "Volume vs 2W Avg": fig_vol_2wk,
    "On-Balance Volume": fig_obv,
    "Volume Profile VPVR": fig_vpvr,
    "Equity Curve": fig_eq,
}


# =============================
# DIVERGENCE SCAN HELPERS
# =============================
def get_scan_reference_end_date(timeframe_name: str):
    today = datetime.date.today()

    if timeframe_name == "Aylık":
        first_day_of_current_month = today.replace(day=1)
        return first_day_of_current_month - datetime.timedelta(days=1)

    if timeframe_name == "Haftalık":
        days_since_friday = (today.weekday() - 4) % 7
        if days_since_friday == 0:
            days_since_friday = 7
        return today - datetime.timedelta(days=days_since_friday)

    if timeframe_name == "Günlük":
        return today - datetime.timedelta(days=1)

    return None

@st.cache_data(ttl=30 * 60, show_spinner=False)
def scan_divergences_for_symbol(ticker: str, timeframe_name: str, force_latest_daily: bool = False) -> pd.DataFrame:
    timeframe_map = {
        "Aylık": ("10y", "1mo"),
        "Haftalık": ("5y", "1wk"),
        "Günlük": ("2y", "1d"),
        "Saatlik": ("60d", "1h"),
    }

    period, interval = timeframe_map[timeframe_name]
    scan_end_date = get_scan_reference_end_date(timeframe_name)

    df_scan = load_data_cached(
        ticker,
        period,
        interval,
        end_date=scan_end_date,
        force_latest=False,
    )

    if df_scan is None or df_scan.empty or len(df_scan) < 35:
        return pd.DataFrame()

    required_cols = {"Open", "High", "Low", "Close", "Volume"}
    if not required_cols.issubset(set(df_scan.columns)):
        return pd.DataFrame()

    close = df_scan["Close"]
    high = df_scan["High"]
    low = df_scan["Low"]
    volume = df_scan["Volume"]

    _, _, macd_hist = macd(close)
    ema13 = ema(close, 13)
    adx_line, _, _ = adx_indicator(high, low, close)
    fi_line = ema(force_index(close, volume), 2)
    rsi13_line = rsi(close, 13)
    stoch_k, _ = stochastic(high, low, close, k_period=5, d_period=3)
    _, bull_power, bear_power = elder_ray(high, low, close, 13)

    results = []

    def add_result(indicator_name: str, bull_tuple: Tuple[bool, int], bear_tuple: Tuple[bool, int]):
        bull_found, bull_bars_ago = bull_tuple
        bear_found, bear_bars_ago = bear_tuple

        if bull_found:
            results.append({
                "Sembol": ticker,
                "Zaman Dilimi": timeframe_name,
                "Gösterge": indicator_name,
                "Uyumsuzluk": "Pozitif",
                "Kaç Bar Önce": int(bull_bars_ago),
                "Son Kapanış": round(float(close.iloc[-1]), 4),
            })

        if bear_found:
            results.append({
                "Sembol": ticker,
                "Zaman Dilimi": timeframe_name,
                "Gösterge": indicator_name,
                "Uyumsuzluk": "Negatif",
                "Kaç Bar Önce": int(bear_bars_ago),
                "Son Kapanış": round(float(close.iloc[-1]), 4),
            })

    add_result("MACD Histogram", check_bullish_divergence(close, macd_hist, volume=volume), check_bearish_divergence(close, macd_hist, volume=volume))
    add_result("EMA", check_bullish_divergence(close, ema13, volume=volume), check_bearish_divergence(close, ema13, volume=volume))
    add_result("ADX", check_bullish_divergence(close, adx_line, volume=volume), check_bearish_divergence(close, adx_line, volume=volume))
    add_result("Kuvvet Endeksi (FI)", check_bullish_divergence(close, fi_line, volume=volume), check_bearish_divergence(close, fi_line, volume=volume))
    add_result("RSI (13)", check_bullish_divergence(close, rsi13_line, volume=volume), check_bearish_divergence(close, rsi13_line, volume=volume))
    add_result("Stokastik (5)", check_bullish_divergence(close, stoch_k, volume=volume), check_bearish_divergence(close, stoch_k, volume=volume))

    elder_bull, elder_bull_ago = check_bullish_divergence(close, bear_power, volume=volume)
    elder_bear, elder_bear_ago = check_bearish_divergence(close, bull_power, volume=volume)

    if elder_bull:
        results.append({
            "Sembol": ticker,
            "Zaman Dilimi": timeframe_name,
            "Gösterge": "Elder-Ray",
            "Uyumsuzluk": "Pozitif",
            "Kaç Bar Önce": int(elder_bull_ago),
            "Son Kapanış": round(float(close.iloc[-1]), 4),
        })

    if elder_bear:
        results.append({
            "Sembol": ticker,
            "Zaman Dilimi": timeframe_name,
            "Gösterge": "Elder-Ray",
            "Uyumsuzluk": "Negatif",
            "Kaç Bar Önce": int(elder_bear_ago),
            "Son Kapanış": round(float(close.iloc[-1]), 4),
        })

    return pd.DataFrame(results)


# =============================
# FUTURE PRICE HELPERS
# =============================
FUTURE_PRICE_MODEL_NAMES = [
    "Ridge",
    "Linear Regression",
    "Random Forest",
    "Gradient Boosting",
]


def build_future_price_features(df: pd.DataFrame) -> pd.DataFrame:
    feat = pd.DataFrame(index=df.index)

    base_cols = [
        "Open", "High", "Low", "Close", "Volume", "EMA50", "EMA200", "RSI",
        "MACD", "MACD_signal", "MACD_hist", "BB_mid", "BB_upper", "BB_lower",
        "ATR", "ATR_PCT", "OBV", "OBV_EMA", "VOL_RATIO", "BB_WIDTH"
    ]
    for col in base_cols:
        if col in df.columns:
            feat[col] = pd.to_numeric(df[col], errors="coerce")

    feat["RET_1"] = pd.to_numeric(df["Close"].pct_change(1), errors="coerce")
    feat["RET_3"] = pd.to_numeric(df["Close"].pct_change(3), errors="coerce")
    feat["RET_5"] = pd.to_numeric(df["Close"].pct_change(5), errors="coerce")
    feat["VOL_CHG_1"] = pd.to_numeric(df["Volume"].pct_change(1), errors="coerce")
    feat["CLOSE_OPEN_PCT"] = pd.to_numeric((df["Close"] / df["Open"] - 1.0).replace([np.inf, -np.inf], np.nan), errors="coerce")
    feat["HIGH_LOW_PCT"] = pd.to_numeric((df["High"] / df["Low"] - 1.0).replace([np.inf, -np.inf], np.nan), errors="coerce")

    if "RSI" in df.columns:
        feat["RSI_CHG_1"] = pd.to_numeric(df["RSI"].diff(1), errors="coerce")
    if "MACD_hist" in df.columns:
        feat["MACD_HIST_CHG_1"] = pd.to_numeric(df["MACD_hist"].diff(1), errors="coerce")
    if "ATR" in df.columns:
        feat["ATR_CHG_1"] = pd.to_numeric(df["ATR"].pct_change(1), errors="coerce")

    lag_cols = ["Close", "Volume", "RSI", "MACD_hist", "ATR", "OBV"]
    for lag in [1, 2, 3, 5, 8]:
        for col in lag_cols:
            if col in df.columns:
                feat[f"{col}_lag_{lag}"] = pd.to_numeric(df[col].shift(lag), errors="coerce")

    feat = feat.replace([np.inf, -np.inf], np.nan)
    return feat


def _future_make_model(model_name: str):
    if model_name == "Ridge":
        return Ridge(alpha=1.0)
    if model_name == "Linear Regression":
        return LinearRegression()
    if model_name == "Random Forest":
        return RandomForestRegressor(
            n_estimators=140,
            max_depth=6,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1,
        )
    if model_name == "Gradient Boosting":
        return GradientBoostingRegressor(
            n_estimators=180,
            learning_rate=0.05,
            max_depth=3,
            random_state=42,
        )
    raise ValueError(f"Bilinmeyen model: {model_name}")



def _future_fit_predict(model_name: str, X_train_df: pd.DataFrame, y_train: pd.Series, X_pred_df: pd.DataFrame):
    model = _future_make_model(model_name)

    train_medians = X_train_df.median(axis=0).fillna(0.0)
    X_train_imp = X_train_df.fillna(train_medians)
    X_pred_imp = X_pred_df.fillna(train_medians)

    if model_name in ["Ridge", "Linear Regression"]:
        mu = X_train_imp.mean(axis=0)
        sigma = X_train_imp.std(axis=0).replace(0, 1.0).fillna(1.0)
        X_train_use = (X_train_imp - mu) / sigma
        X_pred_use = (X_pred_imp - mu) / sigma
        model.fit(X_train_use.values, y_train.values)
        preds = model.predict(X_pred_use.values)
        return preds, model, {"mu": mu, "sigma": sigma, "train_medians": train_medians}

    model.fit(X_train_imp.values, y_train.values)
    preds = model.predict(X_pred_imp.values)
    return preds, model, {"train_medians": train_medians}



def _future_metrics(y_true: np.ndarray, y_pred: np.ndarray, current_close_arr: np.ndarray) -> Dict[str, float]:
    mae = float(np.mean(np.abs(y_pred - y_true))) if len(y_true) > 0 else np.nan
    rmse = float(np.sqrt(np.mean((y_pred - y_true) ** 2))) if len(y_true) > 0 else np.nan

    safe_y = np.where(y_true == 0, np.nan, y_true)
    mape = float(np.nanmean(np.abs((y_pred - y_true) / safe_y)) * 100) if len(y_true) > 0 else np.nan

    actual_dir = np.sign(y_true - current_close_arr)
    pred_dir = np.sign(y_pred - current_close_arr)
    direction_acc = float(np.mean(actual_dir == pred_dir) * 100) if len(actual_dir) > 0 else np.nan

    return {
        "mae": mae,
        "rmse": rmse,
        "mape": mape,
        "direction_acc": direction_acc,
    }


def _future_feature_importance(model_name: str, fitted_model, feature_cols: List[str], aux: Dict[str, Any]) -> pd.DataFrame:
    vals = None

    if hasattr(fitted_model, "feature_importances_"):
        vals = np.asarray(fitted_model.feature_importances_, dtype=float)
    elif hasattr(fitted_model, "coef_"):
        vals = np.asarray(np.abs(fitted_model.coef_), dtype=float)
        sigma = aux.get("sigma")
        if sigma is not None:
            sigma_vals = np.asarray(sigma.values, dtype=float)
            sigma_vals = np.where(sigma_vals == 0, 1.0, sigma_vals)
            vals = vals / sigma_vals

    if vals is None:
        return pd.DataFrame(columns=["Feature", "Importance"])

    out = pd.DataFrame({"Feature": feature_cols, "Importance": vals})
    out["Importance"] = pd.to_numeric(out["Importance"], errors="coerce").abs()
    out = out.replace([np.inf, -np.inf], np.nan).dropna().sort_values("Importance", ascending=False)
    return out.head(15).reset_index(drop=True)



def _future_prepare_dataset(df: pd.DataFrame, horizon_bars: int):
    features = build_future_price_features(df)
    close = pd.to_numeric(df["Close"], errors="coerce")
    target_ret = ((close.shift(-horizon_bars) / close) - 1.0).rename("TARGET_RET")
    base_close = close.rename("BASE_CLOSE")

    raw_dataset = pd.concat([features, target_ret, base_close], axis=1).replace([np.inf, -np.inf], np.nan)
    dataset = raw_dataset.dropna()

    # Eğitim seti, hedefi bilinen satırlardan oluşur. Son horizon satır target üretmeyeceği için zaten dışarıda kalır.
    feature_cols = [c for c in dataset.columns if c not in ["TARGET_RET", "BASE_CLOSE"]]

    latest_features_all = features.reindex(columns=feature_cols).replace([np.inf, -np.inf], np.nan).dropna()
    latest_base_close = float(close.iloc[-1]) if not df.empty and pd.notna(close.iloc[-1]) else np.nan

    # Son tahmin için hedefi bilinmeyen en güncel feature satırı kullanılabilir; eğitim datasetine karışmaz.
    latest_features = latest_features_all.tail(1)
    latest_feature_is_out_of_sample = False
    try:
        latest_feature_is_out_of_sample = bool(len(latest_features) > 0 and (len(dataset) == 0 or latest_features.index[-1] not in dataset.index))
    except Exception:
        latest_feature_is_out_of_sample = False

    return dataset, feature_cols, latest_features, latest_base_close, latest_feature_is_out_of_sample


def _future_direction_probabilities(current_price: float, predicted_price: float, residuals: np.ndarray):
    if residuals is None or len(residuals) == 0:
        return np.nan, np.nan, np.nan

    simulated_prices = predicted_price + residuals
    flat_band = max(abs(current_price) * 0.0025, 1e-9)

    up_prob = float(np.mean(simulated_prices > (current_price + flat_band)) * 100)
    down_prob = float(np.mean(simulated_prices < (current_price - flat_band)) * 100)
    flat_prob = max(0.0, 100.0 - up_prob - down_prob)
    return up_prob, down_prob, flat_prob


def _future_confidence_score(mape: float, direction_acc: float, band_width_pct: float) -> float:
    score = 100.0
    if pd.notna(mape):
        score -= mape * 8.0
    else:
        score -= 35.0

    if pd.notna(direction_acc):
        score += (direction_acc - 50.0) * 0.6

    if pd.notna(band_width_pct):
        score -= band_width_pct * 1.5

    return float(max(0.0, min(100.0, score)))



def future_price_single_model_eval(df: pd.DataFrame, horizon_bars: int, model_name: str, use_walkforward: bool = False) -> Dict[str, Any]:
    dataset, feature_cols, latest_features, latest_base_close, latest_feature_is_out_of_sample = _future_prepare_dataset(df, horizon_bars)
    if dataset.empty:
        return {"error": "Model veri seti oluşturulamadı."}
    if latest_features is None or latest_features.empty:
        return {"error": "Tahmin için kullanılabilir son feature satırı bulunamadı."}

    min_rows_needed = max(100, horizon_bars * 10)
    if len(dataset) < min_rows_needed:
        return {"error": f"Yeterli eğitim verisi yok. En az yaklaşık {min_rows_needed} satır gerekli, mevcut: {len(dataset)}."}

    test_size = min(max(20, len(dataset) // 5), 60)
    split_idx = len(dataset) - test_size
    if split_idx < 50:
        return {"error": "Eğitim/test ayrımı için yeterli veri yok."}

    y_test_ret = dataset.iloc[split_idx:]["TARGET_RET"].astype(float).values
    test_index = dataset.index[split_idx:]
    current_close_test = dataset.iloc[split_idx:]["BASE_CLOSE"].astype(float).values
    actual_future_prices = current_close_test * (1.0 + y_test_ret)

    if use_walkforward:
        n_splits = min(5, max(3, len(dataset) // max(25, test_size // 2)))
        if split_idx < n_splits + 20:
            n_splits = 3
        tscv = TimeSeriesSplit(n_splits=n_splits)
        wf_preds_ret = np.full(len(dataset), np.nan, dtype=float)
        for train_idx, test_idx in tscv.split(dataset):
            X_hist = dataset.iloc[train_idx][feature_cols].astype(float)
            y_hist = dataset.iloc[train_idx]["TARGET_RET"].astype(float)
            X_test_fold = dataset.iloc[test_idx][feature_cols].astype(float)
            pred_fold, _, _ = _future_fit_predict(model_name, X_hist, y_hist, X_test_fold)
            wf_preds_ret[test_idx] = np.asarray(pred_fold, dtype=float)
        y_pred_ret_test = wf_preds_ret[split_idx:]
        valid_mask = np.isfinite(y_pred_ret_test)
        if valid_mask.sum() < max(10, test_size // 2):
            wf_preds = []
            for ds_idx in range(split_idx, len(dataset)):
                X_hist = dataset.iloc[:ds_idx][feature_cols].astype(float)
                y_hist = dataset.iloc[:ds_idx]["TARGET_RET"].astype(float)
                X_one = dataset.iloc[[ds_idx]][feature_cols].astype(float)
                pred_one, _, _ = _future_fit_predict(model_name, X_hist, y_hist, X_one)
                wf_preds.append(float(pred_one[0]))
            y_pred_ret_test = np.asarray(wf_preds, dtype=float)
            valid_mask = np.isfinite(y_pred_ret_test)
        current_close_test = current_close_test[valid_mask]
        y_test_ret = y_test_ret[valid_mask]
        actual_future_prices = actual_future_prices[valid_mask]
        y_pred_ret_test = y_pred_ret_test[valid_mask]
    else:
        X_train = dataset.iloc[:split_idx][feature_cols].astype(float)
        y_train = dataset.iloc[:split_idx]["TARGET_RET"].astype(float)
        X_test = dataset.iloc[split_idx:][feature_cols].astype(float)
        y_pred_ret_test, _, _ = _future_fit_predict(model_name, X_train, y_train, X_test)
        y_pred_ret_test = np.asarray(y_pred_ret_test, dtype=float)

    y_pred_test = current_close_test * (1.0 + y_pred_ret_test)
    metrics = _future_metrics(actual_future_prices, y_pred_test, current_close_test)

    residuals = actual_future_prices - y_pred_test
    resid_std = float(np.nanstd(residuals)) if len(residuals) > 1 else np.nan

    X_full = dataset[feature_cols].astype(float)
    y_full = dataset["TARGET_RET"].astype(float)
    pred_latest_ret, fitted_model, aux = _future_fit_predict(model_name, X_full, y_full, latest_features.tail(1).astype(float))
    predicted_return = float(pred_latest_ret[0])
    predicted_price = float(latest_base_close * (1.0 + predicted_return))
    delta_pct = float(predicted_return * 100.0)

    lower_price = predicted_price - 1.28 * resid_std if np.isfinite(resid_std) else np.nan
    upper_price = predicted_price + 1.28 * resid_std if np.isfinite(resid_std) else np.nan
    band_width_pct = float(((upper_price - lower_price) / latest_base_close) * 100.0) if np.isfinite(lower_price) and np.isfinite(upper_price) and latest_base_close != 0 else np.nan

    up_prob, down_prob, flat_prob = _future_direction_probabilities(latest_base_close, predicted_price, residuals)
    confidence_score = _future_confidence_score(metrics.get("mape", np.nan), metrics.get("direction_acc", np.nan), band_width_pct)

    importance_df = _future_feature_importance(model_name, fitted_model, feature_cols, aux)

    return {
        "model": model_name,
        "current_price": float(latest_base_close),
        "predicted_price": predicted_price,
        "predicted_return_pct": float(predicted_return * 100.0),
        "delta_pct": delta_pct,
        "mae": metrics.get("mae", np.nan),
        "rmse": metrics.get("rmse", np.nan),
        "mape": metrics.get("mape", np.nan),
        "direction_acc": metrics.get("direction_acc", np.nan),
        "resid_std": resid_std,
        "lower_price": lower_price,
        "upper_price": upper_price,
        "predicted_low": lower_price,
        "predicted_high": upper_price,
        "band_width_pct": band_width_pct,
        "up_prob": up_prob,
        "down_prob": down_prob,
        "flat_prob": flat_prob,
        "confidence_score": confidence_score,
        "test_index": test_index,
        "y_true": actual_future_prices,
        "y_pred": y_pred_test,
        "current_close_test": current_close_test,
        "feature_importance": importance_df,
        "latest_base_close": latest_base_close,
        "last_feature_index": str(latest_features.index[-1]) if len(latest_features.index) > 0 else "N/A",
        "latest_feature_is_out_of_sample": bool(latest_feature_is_out_of_sample),
        "data_leakage_guard": "TRAIN_TARGET_KNOWN_ONLY_AND_LATEST_FEATURE_OUT_OF_SAMPLE",
        "train_rows": int(split_idx),
        "test_rows": int(len(y_test_ret)),
    }




def future_price_horizon_benchmark(df: pd.DataFrame, model_name: str, requested_horizon: int) -> pd.DataFrame:
    candidate_horizons = sorted({1, 3, 5, 10, 20, int(requested_horizon)})
    rows = []
    for hz in candidate_horizons:
        if hz < 1:
            continue
        quick = future_price_single_model_eval(df, hz, model_name, use_walkforward=True)
        if quick.get("error"):
            continue
        rows.append({
            "Ufuk (bar)": hz,
            "MAPE %": quick.get("mape", np.nan),
            "RMSE": quick.get("rmse", np.nan),
            "Yön %": quick.get("direction_acc", np.nan),
            "Getiri %": quick.get("predicted_return_pct", np.nan),
            "Tahmin": quick.get("predicted_price", np.nan),
        })
    return pd.DataFrame(rows)




def future_price_ml_forecast(df: pd.DataFrame, horizon_bars: int) -> Dict[str, Any]:
    if df is None or df.empty:
        return {"error": "Tahmin için veri yok."}
    if horizon_bars < 1:
        return {"error": "Tahmin ufku en az 1 bar olmalıdır."}

    model_results = {}
    compare_rows = []

    for model_name in FUTURE_PRICE_MODEL_NAMES:
        model_result = future_price_single_model_eval(df, horizon_bars, model_name, use_walkforward=True)
        if model_result.get("error"):
            return model_result
        model_results[model_name] = model_result
        compare_rows.append({
            "Model": model_name,
            "MAPE %": model_result.get("mape", np.nan),
            "MAE": model_result.get("mae", np.nan),
            "RMSE": model_result.get("rmse", np.nan),
            "Yön %": model_result.get("direction_acc", np.nan),
            "Güven %": model_result.get("confidence_score", np.nan),
            "Tahmin": model_result.get("predicted_price", np.nan),
            "Getiri %": model_result.get("predicted_return_pct", np.nan),
            "Değişim %": model_result.get("delta_pct", np.nan),
        })

    compare_df = pd.DataFrame(compare_rows)
    compare_df["_rank_mape"] = compare_df["MAPE %"].fillna(np.inf)
    compare_df["_rank_rmse"] = compare_df["RMSE"].fillna(np.inf)
    compare_df = compare_df.sort_values(["_rank_mape", "_rank_rmse", "Yön %", "Güven %"], ascending=[True, True, False, False]).reset_index(drop=True)
    best_model_name = str(compare_df.iloc[0]["Model"])
    compare_df["En İyi"] = np.where(compare_df["Model"] == best_model_name, "⭐ En İyi", "")
    compare_df = compare_df.drop(columns=["_rank_mape", "_rank_rmse"])

    best_model_result = model_results[best_model_name]
    horizon_quality_df = future_price_horizon_benchmark(df, best_model_name, horizon_bars)

    trend_regime = "Yükseliş" if ("EMA50" in df.columns and "EMA200" in df.columns and df["EMA50"].iloc[-1] > df["EMA200"].iloc[-1]) else "Nötr/Aşağı"
    current_price = float(df["Close"].iloc[-1])

    atr_pct_now = safe_float(df["ATR_PCT"].iloc[-1]) if "ATR_PCT" in df.columns and len(df) > 0 else np.nan
    vol_regime = "Yüksek" if np.isfinite(atr_pct_now) and atr_pct_now > 0.04 else ("Normal" if np.isfinite(atr_pct_now) else "N/A")

    return {
        "best_model_name": best_model_name,
        "best_model_result": best_model_result,
        "models": model_results,
        "compare_df": compare_df,
        "horizon_quality_df": horizon_quality_df,
        "horizon_bars": int(horizon_bars),
        "current_price": current_price,
        "trend_regime": trend_regime,
        "vol_regime": vol_regime,
    }




def _fp_safe_value(model_row: Dict[str, Any], fp_result: Dict[str, Any], key: str, default=np.nan):
    if isinstance(model_row, dict):
        if key in model_row and pd.notna(model_row.get(key)):
            return model_row.get(key)
        alias_map = {
            "current_price": ["latest_base_close"],
            "predicted_low": ["lower_price"],
            "predicted_high": ["upper_price"],
        }
        for alt_key in alias_map.get(key, []):
            if alt_key in model_row and pd.notna(model_row.get(alt_key)):
                return model_row.get(alt_key)
    if isinstance(fp_result, dict):
        if key in fp_result and pd.notna(fp_result.get(key)):
            return fp_result.get(key)
        alias_map_result = {
            "current_price": ["current_price"],
        }
        for alt_key in alias_map_result.get(key, []):
            if alt_key in fp_result and pd.notna(fp_result.get(alt_key)):
                return fp_result.get(alt_key)
    return default


def _fp_fmt_num(value, nd=2, suffix=""):
    try:
        if value is None or (isinstance(value, float) and not np.isfinite(value)):
            return "N/A"
        return f"{float(value):.{nd}f}{suffix}"
    except Exception:
        return "N/A"

def classify_text_polarity(text_value: str) -> str:
    txt = str(text_value or "").lower()
    pos_hits = sum(1 for term in POSITIVE_SENTIMENT_TERMS if term in txt)
    neg_hits = sum(1 for term in NEGATIVE_SENTIMENT_TERMS if term in txt)
    if pos_hits > neg_hits and pos_hits > 0:
        return "Pozitif"
    if neg_hits > pos_hits and neg_hits > 0:
        return "Negatif"
    return "Nötr"

def get_company_name_for_social(selected_ticker: str, selected_market: str, sdf: pd.DataFrame) -> str:
    try:
        if sdf is not None and not sdf.empty:
            row = find_screener_row(sdf, selected_ticker)
            if row and row.get("longName"):
                return str(row.get("longName")).strip()
    except Exception:
        pass
    try:
        f = fetch_fundamentals_generic(selected_ticker, market=selected_market)
        long_name = str(f.get("longName") or "").strip()
        if long_name:
            return long_name
    except Exception:
        pass
    return naked_ticker(selected_ticker)

def _x_bearer_token(override: str = "") -> str:
    if override and str(override).strip():
        return str(override).strip()
    try:
        return str(st.secrets.get("X_BEARER_TOKEN", "")).strip()
    except Exception:
        return ""

def _youtube_api_key(override: str = "") -> str:
    if override and str(override).strip():
        return str(override).strip()
    try:
        return str(st.secrets.get("YOUTUBE_API_KEY", "")).strip()
    except Exception:
        return ""

@st.cache_data(ttl=15 * 60, show_spinner=False)
def fetch_x_trends_bundle(
    ticker_keyword: str,
    company_keyword: str,
    bearer_token_override: str = "",
    max_results: int = 50,
) -> Dict[str, Any]:
    bearer = _x_bearer_token(bearer_token_override)
    if not bearer:
        return {"error": "X API için X_BEARER_TOKEN gerekli. secrets veya giriş kutusundan token verin."}

    ticker_kw = str(ticker_keyword or "").strip()
    company_kw = str(company_keyword or "").strip()

    query_parts = []
    if ticker_kw:
        query_parts.append(f'"{ticker_kw}"')
    if company_kw and company_kw.lower() != ticker_kw.lower():
        query_parts.append(f'"{company_kw}"')

    if not query_parts:
        return {"error": "X araması için anahtar kelime bulunamadı."}

    query = "(" + " OR ".join(query_parts) + ") -is:retweet lang:tr"
    headers = {"Authorization": f"Bearer {bearer}"}
    params = {
        "query": query,
        "max_results": min(max(int(max_results), 10), 100),
        "tweet.fields": "created_at,public_metrics,lang",
        "expansions": "author_id",
        "user.fields": "name,username",
    }

    try:
        resp = requests.get(
            "https://api.x.com/2/tweets/search/recent",
            headers=headers,
            params=params,
            timeout=30,
        )
        resp.raise_for_status()
        payload = resp.json()
    except Exception as e:
        return {"error": f"X recent search başarısız: {e}"}

    user_map = {}
    for user in payload.get("includes", {}).get("users", []) or []:
        user_map[str(user.get("id"))] = {
            "username": user.get("username", ""),
            "name": user.get("name", ""),
        }

    rows = []
    for item in payload.get("data", []) or []:
        metrics = item.get("public_metrics", {}) or {}
        text_value = str(item.get("text") or "")
        rows.append({
            "ID": str(item.get("id", "")),
            "Date": pd.to_datetime(item.get("created_at")),
            "Text": text_value,
            "Author": user_map.get(str(item.get("author_id")), {}).get("username", ""),
            "Author Name": user_map.get(str(item.get("author_id")), {}).get("name", ""),
            "Like Count": safe_float(metrics.get("like_count")),
            "Retweet Count": safe_float(metrics.get("retweet_count")),
            "Reply Count": safe_float(metrics.get("reply_count")),
            "Quote Count": safe_float(metrics.get("quote_count")),
            "Impression Count": safe_float(metrics.get("impression_count")),
            "Polarity": classify_text_polarity(text_value),
        })

    posts_df = pd.DataFrame(rows)
    if not posts_df.empty:
        posts_df = posts_df.sort_values("Date").reset_index(drop=True)
        posts_df["Engagement"] = posts_df[["Like Count", "Retweet Count", "Reply Count", "Quote Count"]].fillna(0).sum(axis=1)
        posts_df["Day"] = posts_df["Date"].dt.floor("D")
    else:
        posts_df = pd.DataFrame(columns=["ID", "Date", "Text", "Author", "Author Name", "Like Count", "Retweet Count", "Reply Count", "Quote Count", "Impression Count", "Polarity", "Engagement", "Day"])

    return {
        "error": None,
        "query": query,
        "posts_df": posts_df,
        "meta": payload.get("meta", {}),
    }

def build_x_indicators(posts_df: pd.DataFrame) -> Dict[str, Any]:
    if posts_df is None or posts_df.empty:
        return {"error": "X araması veri döndürmedi."}

    volume_df = posts_df.groupby("Day").size().reset_index(name="Post Count")
    polarity_df = posts_df.groupby("Polarity").size().reset_index(name="Count")

    pos_count = int((posts_df["Polarity"] == "Pozitif").sum())
    neg_count = int((posts_df["Polarity"] == "Negatif").sum())
    neu_count = int((posts_df["Polarity"] == "Nötr").sum())
    total_count = max(len(posts_df), 1)

    positive_score = round(pos_count / total_count * 100.0, 1)
    negative_score = round(neg_count / total_count * 100.0, 1)

    daily_counts = volume_df["Post Count"].astype(float)
    volume_momentum = float(daily_counts.iloc[-1] - daily_counts.iloc[0]) if len(daily_counts) >= 2 else np.nan

    avg_engagement = float(posts_df["Engagement"].fillna(0).mean()) if "Engagement" in posts_df else np.nan
    pos_engagement = float(posts_df.loc[posts_df["Polarity"] == "Pozitif", "Engagement"].fillna(0).mean()) if pos_count > 0 else 0.0
    neg_engagement = float(posts_df.loc[posts_df["Polarity"] == "Negatif", "Engagement"].fillna(0).mean()) if neg_count > 0 else 0.0

    if positive_score >= 55 and positive_score > negative_score:
        verdict = "POZİTİF X görünümü"
    elif negative_score >= 55 and negative_score > positive_score:
        verdict = "NEGATİF X görünümü"
    else:
        verdict = "NÖTR / karışık X görünümü"

    return {
        "error": None,
        "volume_df": volume_df,
        "polarity_df": polarity_df,
        "positive_score": positive_score,
        "negative_score": negative_score,
        "positive_count": pos_count,
        "negative_count": neg_count,
        "neutral_count": neu_count,
        "volume_momentum": volume_momentum,
        "avg_engagement": avg_engagement,
        "positive_engagement": pos_engagement,
        "negative_engagement": neg_engagement,
        "verdict": verdict,
    }

@st.cache_data(ttl=30 * 60, show_spinner=False)
def _fetch_youtube_search_items(api_key: str, query: str, published_after_iso: Optional[str], max_results: int) -> List[dict]:
    params = {
        "part": "snippet",
        "type": "video",
        "q": query,
        "maxResults": min(max(int(max_results), 5), 50),
        "order": "date",
        "key": api_key,
    }
    if published_after_iso:
        params["publishedAfter"] = published_after_iso

    resp = requests.get(
        "https://www.googleapis.com/youtube/v3/search",
        params=params,
        timeout=30,
    )
    resp.raise_for_status()
    payload = resp.json()
    return payload.get("items", []) or []

@st.cache_data(ttl=30 * 60, show_spinner=False)
def _fetch_youtube_video_stats(api_key: str, video_ids: Tuple[str, ...]) -> Dict[str, dict]:
    if not video_ids:
        return {}
    params = {
        "part": "snippet,statistics",
        "id": ",".join(video_ids),
        "maxResults": len(video_ids),
        "key": api_key,
    }
    resp = requests.get(
        "https://www.googleapis.com/youtube/v3/videos",
        params=params,
        timeout=30,
    )
    resp.raise_for_status()
    payload = resp.json()

    out = {}
    for item in payload.get("items", []) or []:
        out[str(item.get("id"))] = item
    return out

@st.cache_data(ttl=30 * 60, show_spinner=False)
def fetch_youtube_trends_bundle(
    ticker_keyword: str,
    company_keyword: str,
    api_key_override: str = "",
    lookback_label: str = "Son 30 Gün",
    max_results: int = 25,
) -> Dict[str, Any]:
    api_key = _youtube_api_key(api_key_override)
    if not api_key:
        return {"error": "YouTube için YOUTUBE_API_KEY gerekli. secrets veya giriş kutusundan API key verin."}

    lookback_map = {
        "Son 1 Gün": 1,
        "Son 7 Gün": 7,
        "Son 30 Gün": 30,
        "Son 90 Gün": 90,
        "Son 12 Ay": 365,
    }
    lookback_days = int(lookback_map.get(lookback_label, 30))
    published_after_iso = (datetime.datetime.utcnow() - datetime.timedelta(days=lookback_days)).replace(microsecond=0).isoformat("T") + "Z"

    ticker_kw = str(ticker_keyword or "").strip()
    company_kw = str(company_keyword or "").strip()

    queries = []
    if ticker_kw:
        queries.append(ticker_kw)
    if company_kw and company_kw.lower() != ticker_kw.lower():
        queries.append(company_kw)

    if not queries:
        return {"error": "YouTube araması için anahtar kelime bulunamadı."}

    raw_items = []
    for q in queries:
        try:
            raw_items.extend(_fetch_youtube_search_items(api_key, q, published_after_iso, max_results))
        except Exception as e:
            return {"error": f"YouTube search başarısız: {e}"}

    seen_ids = set()
    ordered_video_ids = []
    for item in raw_items:
        vid = str(item.get("id", {}).get("videoId") or "").strip()
        if vid and vid not in seen_ids:
            seen_ids.add(vid)
            ordered_video_ids.append(vid)

    try:
        stats_map = _fetch_youtube_video_stats(api_key, tuple(ordered_video_ids[:50]))
    except Exception as e:
        return {"error": f"YouTube videos.list başarısız: {e}"}

    rows = []
    for vid in ordered_video_ids:
        item = stats_map.get(vid)
        if not item:
            continue
        snippet = item.get("snippet", {}) or {}
        stats = item.get("statistics", {}) or {}
        title = str(snippet.get("title") or "")
        description = str(snippet.get("description") or "")
        text_value = (title + " " + description).strip()
        rows.append({
            "Video ID": vid,
            "Published At": pd.to_datetime(snippet.get("publishedAt")),
            "Title": title,
            "Channel": snippet.get("channelTitle", ""),
            "Description": description,
            "View Count": safe_float(stats.get("viewCount")),
            "Like Count": safe_float(stats.get("likeCount")),
            "Comment Count": safe_float(stats.get("commentCount")),
            "Polarity": classify_text_polarity(text_value),
            "Keyword Match": "Ticker" if ticker_kw and ticker_kw.lower() in text_value.lower() else ("Company" if company_kw and company_kw.lower() in text_value.lower() else "Other"),
            "Video URL": f"https://www.youtube.com/watch?v={vid}",
        })

    videos_df = pd.DataFrame(rows)
    if not videos_df.empty:
        videos_df = videos_df.sort_values("Published At").reset_index(drop=True)
        videos_df["Day"] = videos_df["Published At"].dt.floor("D")
        videos_df["Engagement"] = videos_df[["Like Count", "Comment Count"]].fillna(0).sum(axis=1)
    else:
        videos_df = pd.DataFrame(columns=["Video ID", "Published At", "Title", "Channel", "Description", "View Count", "Like Count", "Comment Count", "Polarity", "Keyword Match", "Video URL", "Day", "Engagement"])

    return {
        "error": None,
        "videos_df": videos_df,
        "published_after_iso": published_after_iso,
    }

def build_youtube_indicators(videos_df: pd.DataFrame) -> Dict[str, Any]:
    if videos_df is None or videos_df.empty:
        return {"error": "YouTube araması veri döndürmedi."}

    volume_df = videos_df.groupby("Day").size().reset_index(name="Video Count")
    polarity_df = videos_df.groupby("Polarity").size().reset_index(name="Count")

    pos_count = int((videos_df["Polarity"] == "Pozitif").sum())
    neg_count = int((videos_df["Polarity"] == "Negatif").sum())
    neu_count = int((videos_df["Polarity"] == "Nötr").sum())
    total_count = max(len(videos_df), 1)

    positive_score = round(pos_count / total_count * 100.0, 1)
    negative_score = round(neg_count / total_count * 100.0, 1)

    avg_views = float(videos_df["View Count"].fillna(0).mean())
    avg_engagement = float(videos_df["Engagement"].fillna(0).mean())
    pos_views = float(videos_df.loc[videos_df["Polarity"] == "Pozitif", "View Count"].fillna(0).mean()) if pos_count > 0 else 0.0
    neg_views = float(videos_df.loc[videos_df["Polarity"] == "Negatif", "View Count"].fillna(0).mean()) if neg_count > 0 else 0.0

    if positive_score >= 55 and positive_score > negative_score:
        verdict = "POZİTİF YouTube görünümü"
    elif negative_score >= 55 and negative_score > positive_score:
        verdict = "NEGATİF YouTube görünümü"
    else:
        verdict = "NÖTR / karışık YouTube görünümü"

    return {
        "error": None,
        "volume_df": volume_df,
        "polarity_df": polarity_df,
        "positive_score": positive_score,
        "negative_score": negative_score,
        "positive_count": pos_count,
        "negative_count": neg_count,
        "neutral_count": neu_count,
        "avg_views": avg_views,
        "avg_engagement": avg_engagement,
        "positive_views": pos_views,
        "negative_views": neg_views,
        "verdict": verdict,
    }



# =============================
# ECONOMIC CALENDAR HELPERS
# =============================
ECON_CALENDAR_EVENT_MAP = {
    "interest rate": {
        "impact": "Faiz kararı; tahvil faizi, bankacılık hisseleri ve genel endeks üzerinde güçlü etki yaratabilir. Sürpriz artışlar büyüme hisselerini baskılayabilir.",
        "sectors": "Bankalar, finans, gayrimenkul, yüksek büyüme/teknoloji, tüketim.",
    },
    "inflation": {
        "impact": "Enflasyon verisi faiz beklentilerini değiştirir. Yüksek enflasyon tahvil faizlerini ve kur oynaklığını artırabilir.",
        "sectors": "Bankalar, perakende, tüketim, ulaştırma, ithalatçı şirketler, teknoloji.",
    },
    "cpi": {
        "impact": "TÜFE beklenenden yüksek gelirse sıkı para politikası beklentisi artabilir. Bu genelde değerlemeleri baskılar, bankalarda ise karışık etki yaratır.",
        "sectors": "Bankalar, perakende, tüketim, büyüme hisseleri, gayrimenkul.",
    },
    "ppi": {
        "impact": "Üretici fiyatları maliyet baskısını gösterir. Marj baskısı özellikle sanayi ve tüketim şirketleri için önemlidir.",
        "sectors": "Sanayi, üretim, perakende, dayanıklı tüketim, otomotiv.",
    },
    "gdp": {
        "impact": "Büyüme verisi ekonomik aktivitenin hızını gösterir. Güçlü veri döngüsel hisseleri destekleyebilir.",
        "sectors": "Sanayi, bankalar, ulaştırma, enerji, inşaat, emtia bağlantılı hisseler.",
    },
    "non farm": {
        "impact": "Tarım dışı istihdam ekonomik ısınma ve faiz beklentileri için kritik veridir. Çok güçlü veri bazen hisse için iyi, bazen de 'faiz daha yüksek kalır' endişesi nedeniyle karışık olabilir.",
        "sectors": "Bankalar, endeks genel, tüketim, sanayi, dolar hassas şirketler.",
    },
    "unemployment": {
        "impact": "İşsizlik oranı işgücü piyasasının gücünü gösterir. Düşük işsizlik tüketim için olumlu, ancak ücret baskısı ve faiz beklentileri için olumsuz olabilir.",
        "sectors": "Tüketim, perakende, bankalar, sanayi.",
    },
    "payroll": {
        "impact": "İstihdam artışı iç talep ve faiz beklentileri için önemlidir. Özellikle endeks yönü üzerinde etkili olabilir.",
        "sectors": "Bankalar, tüketim, sanayi, endeks geneli.",
    },
    "pmi": {
        "impact": "PMI ve ISM tipi veriler ekonomik aktivitenin öncü göstergeleridir. 50 üzeri genelde büyüme sinyalidir.",
        "sectors": "Sanayi, lojistik, emtia, otomotiv, ihracatçı şirketler.",
    },
    "ism": {
        "impact": "ISM verileri üretim ve hizmet aktivitesinin öncü sinyalidir. Beklenti üzeri veri risk iştahını artırabilir.",
        "sectors": "Sanayi, ulaştırma, hizmet, bankalar, endeks geneli.",
    },
    "retail sales": {
        "impact": "Perakende satışlar iç talebi gösterir. Güçlü veri tüketim hisseleri için pozitif olabilir.",
        "sectors": "Perakende, e-ticaret, tüketim, bankalar, ödeme sistemleri.",
    },
    "consumer confidence": {
        "impact": "Tüketici güveni harcama iştahını etkiler. Zayıf veri perakende ve dayanıklı tüketimi baskılayabilir.",
        "sectors": "Perakende, otomotiv, beyaz eşya, bankalar, turizm.",
    },
    "industrial production": {
        "impact": "Sanayi üretimi ekonomik aktivitenin sert verisidir. Beklenenden güçlü gelmesi döngüsel hisseleri destekleyebilir.",
        "sectors": "Sanayi, otomotiv, çelik, çimento, enerji, taşımacılık.",
    },
    "housing": {
        "impact": "Konut başlangıçları / satışları faiz duyarlılığı yüksek göstergelerdir. Zayıf veri gayrimenkul ve inşaatı baskılayabilir.",
        "sectors": "Gayrimenkul, inşaat, çimento, banka, yapı malzemeleri.",
    },
    "trade balance": {
        "impact": "Dış ticaret dengesi kur ve ihracat görünümünü etkileyebilir. Özellikle açık veren ekonomilerde önemli olabilir.",
        "sectors": "İhracatçı sanayi, lojistik, havacılık, ithalatçı perakende.",
    },
    "oil": {
        "impact": "Petrol ve stok verileri enerji maliyetlerini ve emtia fiyatlarını etkiler. Enerji maliyetine hassas sektörlerde oynaklık yaratabilir.",
        "sectors": "Enerji, petrokimya, ulaştırma, havacılık, lojistik, sanayi.",
    },
}

def get_econ_event_education(event_name: str, category: str = "", country: str = "") -> str:
    event_low = str(event_name or "").lower()
    cat_low = str(category or "").lower()

    selected = None
    for key, val in ECON_CALENDAR_EVENT_MAP.items():
        if key in event_low or key in cat_low:
            selected = val
            break

    if selected is None:
        selected = {
            "impact": "Bu veri, ilgili ülkenin büyüme, enflasyon, faiz veya tüketim görünümü üzerinden piyasayı etkileyebilir. Beklenti sapması ne kadar büyükse piyasa tepkisi de o kadar sert olabilir.",
            "sectors": "Bankalar, endeks geneli ve veriyle ilişkili sektörler.",
        }

    return f"Piyasa Etkisi: {selected['impact']} | Etkilenebilecek Sektörler: {selected['sectors']}"

def make_hover_question_html(event_name: str, category: str = "", country: str = "") -> str:
    tooltip = get_econ_event_education(event_name, category, country)
    tooltip = tooltip.replace("&", "&amp;").replace('"', "&quot;").replace("<", "&lt;").replace(">", "&gt;")
    return f'<span title="{tooltip}" style="cursor:help; font-weight:700; color:#d97706;"> ? </span>'

def _normalize_te_calendar_df(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()

    out = df.copy()
    if "Date" in out.columns:
        out["Date"] = pd.to_datetime(out["Date"], errors="coerce", utc=True).dt.tz_convert(None)
    else:
        out["Date"] = pd.NaT

    preferred_cols = [
        "Date", "Country", "Category", "Event", "Actual", "Forecast", "Previous",
        "Importance", "Unit", "Currency", "Reference", "Ticker", "Symbol", "URL"
    ]
    for col in preferred_cols:
        if col not in out.columns:
            out[col] = np.nan

    out["ImportanceLabel"] = out["Importance"].map({1: "Düşük", 2: "Orta", 3: "Yüksek"}).fillna("N/A")
    out["Education"] = out.apply(
        lambda r: get_econ_event_education(r.get("Event", ""), r.get("Category", ""), r.get("Country", "")),
        axis=1,
    )
    out = out.sort_values(["Date", "Country", "Importance"], ascending=[True, True, False]).reset_index(drop=True)
    return out[preferred_cols + ["ImportanceLabel", "Education"]]

@st.cache_data(ttl=15 * 60, show_spinner=False)

def fetch_economic_calendar(country_names: Tuple[str, ...], importance: str = "3", api_key_override: str = "", days_back: int = 1, days_forward: int = 14) -> Dict[str, Any]:
    api_key = str(api_key_override or "").strip()
    if not api_key:
        try:
            api_key = str(st.secrets.get("FMP_API_KEY", "")).strip()
        except Exception:
            api_key = ""

    if not api_key:
        return {
            "error": "FMP economic calendar için API key gerekli. Secrets içine FMP_API_KEY eklemeli veya kutuya API key girmelisin.",
            "df": pd.DataFrame(),
            "source": "none",
        }

    clean_countries = [str(c).strip().lower() for c in country_names if str(c).strip()]
    if not clean_countries:
        return {"error": "En az bir ülke seçmelisin.", "df": pd.DataFrame(), "source": "none"}

    now_dt = datetime.datetime.utcnow()
    start_dt = (now_dt - datetime.timedelta(days=int(days_back))).date()
    end_dt = (now_dt + datetime.timedelta(days=int(days_forward))).date()

    country_name_map = {
        "united states": "US",
        "euro area": "EU",
        "united kingdom": "GB",
        "turkey": "TR",
        "china": "CN",
        "japan": "JP",
        "germany": "DE",
        "france": "FR",
        "canada": "CA",
        "australia": "AU",
        "india": "IN",
        "brazil": "BR",
    }

    importance_allowed = {int(x) for x in str(importance).split(",") if str(x).strip().isdigit()}

    try:
        url = "https://financialmodelingprep.com/stable/economic-calendar"
        params = {
            "from": start_dt.isoformat(),
            "to": end_dt.isoformat(),
            "apikey": api_key,
        }
        resp = requests.get(url, params=params, timeout=35)
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        return {"error": f"FMP economic calendar çağrısı başarısız: {e}", "df": pd.DataFrame(), "source": "fmp"}

    df = pd.DataFrame(data)
    if df.empty:
        return {"error": "FMP economic calendar seçilen aralıkta veri döndürmedi.", "df": pd.DataFrame(), "source": "fmp"}

    rename_map = {
        "date": "Date",
        "country": "Country",
        "event": "Event",
        "impact": "Importance",
        "actual": "Actual",
        "estimate": "Forecast",
        "previous": "Previous",
        "changePercentage": "ChangePercentage",
        "currency": "Currency",
    }
    for old, new in rename_map.items():
        if old in df.columns and new not in df.columns:
            df[new] = df[old]

    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce", utc=True).dt.tz_convert(None)
    else:
        df["Date"] = pd.NaT

    if "Country" in df.columns:
        df["Country"] = df["Country"].astype(str)
        lower_country = df["Country"].str.lower()
        country_codes = {country_name_map.get(c, c).lower() for c in clean_countries}
        df = df[lower_country.isin(set(clean_countries) | country_codes)].copy()

    if "Importance" not in df.columns:
        df["Importance"] = np.nan

    def _impact_to_importance(val):
        sval = str(val).strip().lower()
        if sval in {"high", "3"}:
            return 3
        if sval in {"medium", "moderate", "2"}:
            return 2
        if sval in {"low", "1"}:
            return 1
        return np.nan

    df["Importance"] = df["Importance"].apply(_impact_to_importance)
    if importance_allowed:
        df = df[df["Importance"].isin(importance_allowed)].copy()

    if "Category" not in df.columns:
        df["Category"] = "Economic Data"

    preferred_cols = ["Date", "Country", "Category", "Event", "Actual", "Forecast", "Previous", "Importance", "Unit", "Currency", "Reference", "Ticker", "Symbol", "URL"]
    for col in preferred_cols:
        if col not in df.columns:
            df[col] = np.nan

    df["ImportanceLabel"] = df["Importance"].map({1: "Düşük", 2: "Orta", 3: "Yüksek"}).fillna("N/A")
    df["Education"] = df.apply(lambda r: get_econ_event_education(r.get("Event", ""), r.get("Category", ""), r.get("Country", "")), axis=1)
    df = df.sort_values(["Date", "Country", "Importance"], ascending=[True, True, False]).reset_index(drop=True)
    return {"error": None, "df": df[preferred_cols + ["ImportanceLabel", "Education"]], "source": "fmp"}

@st.cache_data(ttl=30 * 60, show_spinner=False)

def fetch_bist100_nhnl_indicator() -> Dict[str, Any]:
    bist100 = load_universe_file(pjoin("universes", "bist100.txt"))
    if not bist100:
        return {"error": "BIST100 universe listesi bulunamadı.", "df": pd.DataFrame()}

    tickers = [normalize_ticker(t, "BIST") for t in bist100]
    try:
        raw = yf.download(tickers, period="2y", interval="1d", auto_adjust=True, group_by="ticker", threads=True, progress=False)
    except Exception as e:
        return {"error": f"NH-NL verisi indirilemedi: {e}", "df": pd.DataFrame()}

    if raw is None or raw.empty:
        return {"error": "NH-NL için veri gelmedi.", "df": pd.DataFrame()}

    nh_flags = []
    nl_flags = []

    for tk in tickers:
        try:
            sub = raw[tk].copy() if isinstance(raw.columns, pd.MultiIndex) else raw.copy()
            sub = _flatten_yf(sub)
            if sub.empty or "High" not in sub.columns or "Low" not in sub.columns:
                continue

            rolling_high_52w = sub["High"].rolling(252, min_periods=100).max()
            rolling_low_52w = sub["Low"].rolling(252, min_periods=100).min()

            nh_flag = (sub["High"] >= rolling_high_52w).astype(int).rename(f"{tk}_NH")
            nl_flag = (sub["Low"] <= rolling_low_52w).astype(int).rename(f"{tk}_NL")

            nh_flags.append(nh_flag)
            nl_flags.append(nl_flag)
        except Exception:
            continue

    if not nh_flags or not nl_flags:
        return {"error": "NH-NL hesaplamak için yeterli hisse verisi oluşmadı.", "df": pd.DataFrame()}

    nh_df = pd.concat(nh_flags, axis=1).fillna(0)
    nl_df = pd.concat(nl_flags, axis=1).fillna(0)

    out = pd.DataFrame(index=nh_df.index.union(nl_df.index).sort_values())
    out["New_Highs"] = nh_df.sum(axis=1).reindex(out.index).fillna(0)
    out["New_Lows"] = nl_df.sum(axis=1).reindex(out.index).fillna(0)
    out["NH_NL"] = out["New_Highs"] - out["New_Lows"]
    out["NH_NL_EMA10"] = ema(out["NH_NL"], 10)
    universe_count = max(len(tickers), 1)
    out["NH_NL_%"] = (out["NH_NL"] / universe_count) * 100.0
    out["Zero_Line"] = 0.0
    out = out.sort_index()
    return {"error": None, "df": out}

@st.cache_data(ttl=30 * 60, show_spinner=False)
def fetch_vix_series() -> Dict[str, Any]:
    try:
        vix_df = yf.download("^VIX", period="1y", interval="1d", auto_adjust=True, progress=False)
        vix_df = _flatten_yf(vix_df)
    except Exception as e:
        return {"error": f"VIX verisi alınamadı: {e}", "df": pd.DataFrame()}

    if vix_df.empty:
        return {"error": "VIX verisi boş döndü.", "df": pd.DataFrame()}

    vix_df["EMA13"] = ema(vix_df["Close"], 13)
    return {"error": None, "df": vix_df}

@st.cache_data(ttl=30 * 60, show_spinner=False)
def fetch_bist100_force_index_panel() -> Dict[str, Any]:
    try:
        xu100 = yf.download("XU100.IS", period="2y", interval="1d", auto_adjust=True, progress=False)
        xu100 = _flatten_yf(xu100)
    except Exception as e:
        return {"error": f"BIST100 panel verisi alınamadı: {e}", "df": pd.DataFrame()}

    if xu100.empty:
        return {"error": "BIST100 panel verisi boş döndü.", "df": pd.DataFrame()}

    xu100["EMA13"] = ema(xu100["Close"], 13)
    fi = force_index(xu100["Close"], xu100["Volume"])
    xu100["ForceIndex"] = fi
    xu100["ForceIndexEMA13"] = ema(fi, 13)
    return {"error": None, "df": xu100}


@st.cache_data(ttl=6 * 3600, show_spinner=False)
def fetch_next_corporate_dates(symbol: str, selected_market: str = "USA") -> Dict[str, Any]:
    ticker_norm = normalize_ticker(symbol, selected_market)
    out = {
        "ticker": ticker_norm,
        "next_earnings_date": None,
        "next_dividend_date": None,
        "earnings_source": "",
        "dividend_source": "",
        "last_dividend_date": None,
    }

    def _to_dt(val):
        try:
            if val is None:
                return None
            if isinstance(val, pd.Timestamp):
                ts = val
            elif isinstance(val, (list, tuple)):
                if len(val) == 0:
                    return None
                for item in val:
                    dt = _to_dt(item)
                    if dt is not None:
                        return dt
                return None
            elif isinstance(val, dict):
                for item in val.values():
                    dt = _to_dt(item)
                    if dt is not None:
                        return dt
                return None
            else:
                if isinstance(val, (int, float, np.integer, np.floating)) and not pd.isna(val):
                    if float(val) > 1e12:
                        ts = pd.to_datetime(val, unit="ms", errors="coerce", utc=True)
                    else:
                        ts = pd.to_datetime(val, unit="s", errors="coerce", utc=True)
                else:
                    ts = pd.to_datetime(val, errors="coerce", utc=True)

            if ts is None or pd.isna(ts):
                return None
            if getattr(ts, "tzinfo", None) is not None:
                ts = ts.tz_convert(None)
            return ts
        except Exception:
            return None

    def _pick_future(candidates):
        now_dt = pd.Timestamp(datetime.datetime.utcnow())
        parsed = []
        for c in candidates:
            dt = _to_dt(c)
            if dt is not None:
                parsed.append(dt)
        parsed = sorted(set(parsed))
        future = [d for d in parsed if d >= (now_dt - pd.Timedelta(days=1))]
        return future[0] if future else (parsed[-1] if parsed else None)

    try:
        t = yf.Ticker(ticker_norm)
    except Exception:
        return out

    # 1) Calendar route
    try:
        cal = t.calendar
    except Exception:
        cal = None

    cal_dict = {}
    try:
        if isinstance(cal, pd.DataFrame):
            if not cal.empty:
                if cal.shape[1] == 1:
                    cal_dict = {str(idx): cal.iloc[i, 0] for i, idx in enumerate(cal.index)}
                else:
                    cal_dict = cal.to_dict()
        elif isinstance(cal, pd.Series):
            cal_dict = cal.to_dict()
        elif isinstance(cal, dict):
            cal_dict = cal
    except Exception:
        cal_dict = {}

    if cal_dict:
        earnings_candidates = []
        dividend_candidates = []
        for k, v in cal_dict.items():
            key_low = str(k).lower()
            if "earning" in key_low:
                earnings_candidates.append(v)
            if "dividend" in key_low:
                dividend_candidates.append(v)

        picked_earn = _pick_future(earnings_candidates)
        picked_div = _pick_future(dividend_candidates)

        if picked_earn is not None:
            out["next_earnings_date"] = picked_earn
            out["earnings_source"] = "calendar"
        if picked_div is not None:
            out["next_dividend_date"] = picked_div
            out["dividend_source"] = "calendar"

    # 2) Earnings dates route
    if out["next_earnings_date"] is None:
        try:
            edf = t.get_earnings_dates(limit=12)
            if isinstance(edf, pd.DataFrame) and not edf.empty:
                candidates = []
                if isinstance(edf.index, pd.DatetimeIndex):
                    candidates.extend(list(edf.index))
                for col in edf.columns:
                    if "date" in str(col).lower():
                        candidates.extend(edf[col].tolist())
                picked = _pick_future(candidates)
                if picked is not None:
                    out["next_earnings_date"] = picked
                    out["earnings_source"] = "earnings_dates"
        except Exception:
            pass

    # 3) Info / fast info route
    try:
        info = t.info or {}
    except Exception:
        info = {}

    if out["next_earnings_date"] is None:
        earnings_info_candidates = [
            info.get("earningsTimestamp"),
            info.get("earningsTimestampStart"),
            info.get("earningsTimestampEnd"),
            info.get("nextFiscalYearEnd"),
        ]
        picked = _pick_future(earnings_info_candidates)
        if picked is not None:
            out["next_earnings_date"] = picked
            out["earnings_source"] = "info"

    if out["next_dividend_date"] is None:
        dividend_info_candidates = [
            info.get("dividendDate"),
            info.get("exDividendDate"),
        ]
        picked = _pick_future(dividend_info_candidates)
        if picked is not None:
            out["next_dividend_date"] = picked
            out["dividend_source"] = "info"

    # 4) Dividend history fallback (last known date only)
    try:
        divs = getattr(t, "dividends", pd.Series(dtype=float))
        if isinstance(divs, pd.Series) and not divs.empty:
            idx = divs.dropna().index
            if len(idx) > 0:
                last_dt = _to_dt(idx[-1])
                out["last_dividend_date"] = last_dt
    except Exception:
        pass

    return out


# =============================
# CHART PATTERN SCAN HELPERS
# =============================
CHART_PATTERN_GROUPS = [
    "OBO / TOBO",
    "İkili Tepe / İkili Dip",
    "Yükselen / Alçalan Takoz",
    "Üçlü Tepe / Üçlü Dip",
    "Yuvarlak Dip (Çanak)",
    "Bayrak",
    "Flama",
    "Fincan Kulp",
    "Yükselen / Alçalan Üçgen",
    "Dikdörtgen",
]

def _pattern_match(group: str, subtype: str, start_pos: int, end_pos: int, points: List[dict], note: str = "") -> Dict[str, Any]:
    return {
        "group": group,
        "subtype": subtype,
        "start_pos": int(start_pos),
        "end_pos": int(end_pos),
        "points": points,
        "note": note,
    }

def _safe_rel_diff(a: float, b: float) -> float:
    denom = max((abs(float(a)) + abs(float(b))) / 2.0, 1e-9)
    return abs(float(a) - float(b)) / denom

def _get_pivots_with_pos(df: pd.DataFrame, left: int = 3, right: int = 3):
    highs, lows = [], []
    if df is None or df.empty or len(df) < left + right + 5:
        return highs, lows

    H = df["High"].astype(float).values
    L = df["Low"].astype(float).values
    C = df["Close"].astype(float).values
    idx = df.index

    for i in range(left, len(df) - right):
        hwin = H[i-left:i+right+1]
        lwin = L[i-left:i+right+1]
        if H[i] >= np.max(hwin):
            highs.append({"pos": i, "time": idx[i], "price": float(H[i]), "close": float(C[i])})
        if L[i] <= np.min(lwin):
            lows.append({"pos": i, "time": idx[i], "price": float(L[i]), "close": float(C[i])})
    return highs, lows

def _find_between(points: List[dict], left_pos: int, right_pos: int):
    return [p for p in points if left_pos < p["pos"] < right_pos]

def _fit_line(y_vals: np.ndarray):
    x = np.arange(len(y_vals), dtype=float)
    if len(y_vals) < 2:
        return 0.0, float(y_vals[-1]) if len(y_vals) else 0.0
    slope, intercept = np.polyfit(x, y_vals.astype(float), 1)
    return float(slope), float(intercept)

def _detect_head_shoulders(df: pd.DataFrame):
    highs, lows = _get_pivots_with_pos(df)
    if len(highs) < 3 or len(lows) < 2:
        return None

    recent_highs = highs[-10:]
    for i in range(len(recent_highs) - 3, -1, -1):
        h1, h2, h3 = recent_highs[i:i+3]
        if not (h1["pos"] < h2["pos"] < h3["pos"]):
            continue
        l1_list = _find_between(lows, h1["pos"], h2["pos"])
        l2_list = _find_between(lows, h2["pos"], h3["pos"])
        if not l1_list or not l2_list:
            continue
        l1 = min(l1_list, key=lambda x: x["price"])
        l2 = min(l2_list, key=lambda x: x["price"])

        shoulders_close = _safe_rel_diff(h1["price"], h3["price"]) <= 0.08
        head_higher = h2["price"] > max(h1["price"], h3["price"]) * 1.03
        neckline_ok = _safe_rel_diff(l1["price"], l2["price"]) <= 0.08
        spacing_ok = 4 <= (h2["pos"] - h1["pos"]) <= 60 and 4 <= (h3["pos"] - h2["pos"]) <= 60

        if shoulders_close and head_higher and neckline_ok and spacing_ok:
            return _pattern_match(
                "OBO / TOBO",
                "Omuz Baş Omuz (OBO)",
                h1["pos"],
                h3["pos"],
                [h1, l1, h2, l2, h3],
                "Baş, iki omuzdan belirgin yüksek; omuzlar birbirine yakın.",
            )

    if len(lows) < 3 or len(highs) < 2:
        return None

    recent_lows = lows[-10:]
    for i in range(len(recent_lows) - 3, -1, -1):
        l1, l2, l3 = recent_lows[i:i+3]
        if not (l1["pos"] < l2["pos"] < l3["pos"]):
            continue
        h1_list = _find_between(highs, l1["pos"], l2["pos"])
        h2_list = _find_between(highs, l2["pos"], l3["pos"])
        if not h1_list or not h2_list:
            continue
        h1 = max(h1_list, key=lambda x: x["price"])
        h2 = max(h2_list, key=lambda x: x["price"])

        shoulders_close = _safe_rel_diff(l1["price"], l3["price"]) <= 0.08
        head_lower = l2["price"] < min(l1["price"], l3["price"]) * 0.97
        neckline_ok = _safe_rel_diff(h1["price"], h2["price"]) <= 0.08
        spacing_ok = 4 <= (l2["pos"] - l1["pos"]) <= 60 and 4 <= (l3["pos"] - l2["pos"]) <= 60

        if shoulders_close and head_lower and neckline_ok and spacing_ok:
            return _pattern_match(
                "OBO / TOBO",
                "Ters Omuz Baş Omuz (TOBO)",
                l1["pos"],
                l3["pos"],
                [l1, h1, l2, h2, l3],
                "Baş, iki omuzdan belirgin düşük; omuzlar birbirine yakın.",
            )
    return None

def _detect_double_top_bottom(df: pd.DataFrame):
    highs, lows = _get_pivots_with_pos(df)
    if len(highs) >= 2:
        recent_highs = highs[-8:]
        for i in range(len(recent_highs) - 2, -1, -1):
            h1, h2 = recent_highs[i:i+2]
            mids = _find_between(lows, h1["pos"], h2["pos"])
            if not mids:
                continue
            mid = min(mids, key=lambda x: x["price"])
            if 4 <= (h2["pos"] - h1["pos"]) <= 80 and _safe_rel_diff(h1["price"], h2["price"]) <= 0.04:
                trough_depth = (min(h1["price"], h2["price"]) - mid["price"]) / max(min(h1["price"], h2["price"]), 1e-9)
                if trough_depth >= 0.03:
                    return _pattern_match(
                        "İkili Tepe / İkili Dip",
                        "İkili Tepe",
                        h1["pos"],
                        h2["pos"],
                        [h1, mid, h2],
                        "İki tepe benzer seviyede, aradaki dip anlamlı.",
                    )
    if len(lows) >= 2:
        recent_lows = lows[-8:]
        for i in range(len(recent_lows) - 2, -1, -1):
            l1, l2 = recent_lows[i:i+2]
            mids = _find_between(highs, l1["pos"], l2["pos"])
            if not mids:
                continue
            mid = max(mids, key=lambda x: x["price"])
            if 4 <= (l2["pos"] - l1["pos"]) <= 80 and _safe_rel_diff(l1["price"], l2["price"]) <= 0.04:
                peak_height = (mid["price"] - max(l1["price"], l2["price"])) / max(mid["price"], 1e-9)
                if peak_height >= 0.03:
                    return _pattern_match(
                        "İkili Tepe / İkili Dip",
                        "İkili Dip",
                        l1["pos"],
                        l2["pos"],
                        [l1, mid, l2],
                        "İki dip benzer seviyede, aradaki tepe anlamlı.",
                    )
    return None

def _detect_triple_top_bottom(df: pd.DataFrame):
    highs, lows = _get_pivots_with_pos(df)
    if len(highs) >= 3:
        recent_highs = highs[-10:]
        for i in range(len(recent_highs) - 3, -1, -1):
            trio = recent_highs[i:i+3]
            prices = [x["price"] for x in trio]
            if trio[0]["pos"] < trio[1]["pos"] < trio[2]["pos"] and max(prices) > 0 and (max(prices)-min(prices))/max(prices) <= 0.05:
                return _pattern_match(
                    "Üçlü Tepe / Üçlü Dip",
                    "Üçlü Tepe",
                    trio[0]["pos"],
                    trio[2]["pos"],
                    trio,
                    "Üç tepe yakın seviyede.",
                )
    if len(lows) >= 3:
        recent_lows = lows[-10:]
        for i in range(len(recent_lows) - 3, -1, -1):
            trio = recent_lows[i:i+3]
            prices = [x["price"] for x in trio]
            if trio[0]["pos"] < trio[1]["pos"] < trio[2]["pos"] and max(prices) > 0 and (max(prices)-min(prices))/max(prices) <= 0.05:
                return _pattern_match(
                    "Üçlü Tepe / Üçlü Dip",
                    "Üçlü Dip",
                    trio[0]["pos"],
                    trio[2]["pos"],
                    trio,
                    "Üç dip yakın seviyede.",
                )
    return None

def _detect_wedge(df: pd.DataFrame):
    lookback = min(40, len(df))
    if lookback < 18:
        return None
    use = df.tail(lookback).copy()
    slope_high, intercept_high = _fit_line(use["High"].values)
    slope_low, intercept_low = _fit_line(use["Low"].values)

    start_width = (intercept_high - intercept_low)
    end_width = ((slope_high * (lookback - 1) + intercept_high) - (slope_low * (lookback - 1) + intercept_low))
    narrowing = end_width < start_width * 0.9 if start_width > 0 else False

    if narrowing and slope_high > 0 and slope_low > 0 and slope_low > slope_high * 1.15:
        return _pattern_match(
            "Yükselen / Alçalan Takoz",
            "Yükselen Takoz",
            len(df) - lookback,
            len(df) - 1,
            [{"pos": len(df) - lookback, "time": use.index[0], "price": float(use["Low"].iloc[0])},
             {"pos": len(df) - 1, "time": use.index[-1], "price": float(use["High"].iloc[-1])}],
            "Her iki trend çizgisi de yukarı eğimli, bant daralıyor.",
        )

    if narrowing and slope_high < 0 and slope_low < 0 and abs(slope_high) > abs(slope_low) * 1.15:
        return _pattern_match(
            "Yükselen / Alçalan Takoz",
            "Alçalan Takoz",
            len(df) - lookback,
            len(df) - 1,
            [{"pos": len(df) - lookback, "time": use.index[0], "price": float(use["High"].iloc[0])},
             {"pos": len(df) - 1, "time": use.index[-1], "price": float(use["Low"].iloc[-1])}],
            "Her iki trend çizgisi de aşağı eğimli, bant daralıyor.",
        )
    return None

def _detect_rounding_bottom(df: pd.DataFrame):
    lookback = min(80, len(df))
    if lookback < 30:
        return None
    use = df.tail(lookback).copy()
    close = use["Close"].astype(float).values
    min_idx = int(np.argmin(close))
    if not (lookback * 0.2 <= min_idx <= lookback * 0.8):
        return None

    start_p = float(close[0]); end_p = float(close[-1]); low_p = float(close[min_idx])
    if low_p <= 0:
        return None
    depth_ok = start_p > low_p * 1.08 and end_p > low_p * 1.08
    end_recovery_ok = end_p >= start_p * 0.9

    x = np.arange(lookback, dtype=float)
    a, b, c = np.polyfit(x, close, 2)
    curve_ok = a > 0

    if depth_ok and end_recovery_ok and curve_ok:
        return _pattern_match(
            "Yuvarlak Dip (Çanak)",
            "Yuvarlak Dip (Çanak)",
            len(df) - lookback,
            len(df) - 1,
            [
                {"pos": len(df) - lookback, "time": use.index[0], "price": start_p},
                {"pos": len(df) - lookback + min_idx, "time": use.index[min_idx], "price": low_p},
                {"pos": len(df) - 1, "time": use.index[-1], "price": end_p},
            ],
            "Dip orta bölümde, fiyat soldan sağa kademeli toparlanıyor.",
        )
    return None

def _detect_flag(df: pd.DataFrame):
    if len(df) < 30:
        return None
    close = df["Close"].astype(float)
    for pole in [8, 10, 12]:
        for cons in [6, 8, 10]:
            if len(df) < pole + cons + 2:
                continue
            seg_pole = close.iloc[-(pole+cons):-cons]
            seg_cons = close.iloc[-cons:]
            pole_ret = (seg_pole.iloc[-1] / max(seg_pole.iloc[0], 1e-9)) - 1.0
            cons_ret = (seg_cons.iloc[-1] / max(seg_cons.iloc[0], 1e-9)) - 1.0
            cons_width = (df["High"].iloc[-cons:].max() - df["Low"].iloc[-cons:].min()) / max(seg_cons.mean(), 1e-9)
            if pole_ret > 0.08 and cons_ret < 0 and abs(cons_ret) < abs(pole_ret) * 0.5 and cons_width < 0.12:
                return _pattern_match(
                    "Bayrak",
                    "Boğa Bayrağı",
                    len(df) - (pole + cons),
                    len(df) - 1,
                    [
                        {"pos": len(df) - (pole + cons), "time": df.index[-(pole+cons)], "price": float(seg_pole.iloc[0])},
                        {"pos": len(df) - cons - 1, "time": df.index[-cons-1], "price": float(seg_pole.iloc[-1])},
                        {"pos": len(df) - 1, "time": df.index[-1], "price": float(seg_cons.iloc[-1])},
                    ],
                    "Güçlü yükseliş sonrası küçük aşağı eğimli/dar konsolidasyon.",
                )
            if pole_ret < -0.08 and cons_ret > 0 and abs(cons_ret) < abs(pole_ret) * 0.5 and cons_width < 0.12:
                return _pattern_match(
                    "Bayrak",
                    "Ayı Bayrağı",
                    len(df) - (pole + cons),
                    len(df) - 1,
                    [
                        {"pos": len(df) - (pole + cons), "time": df.index[-(pole+cons)], "price": float(seg_pole.iloc[0])},
                        {"pos": len(df) - cons - 1, "time": df.index[-cons-1], "price": float(seg_pole.iloc[-1])},
                        {"pos": len(df) - 1, "time": df.index[-1], "price": float(seg_cons.iloc[-1])},
                    ],
                    "Güçlü düşüş sonrası küçük yukarı eğimli/dar konsolidasyon.",
                )
    return None

def _detect_pennant(df: pd.DataFrame):
    if len(df) < 28:
        return None
    pole = 10
    cons = 10
    close = df["Close"].astype(float)
    seg_pole = close.iloc[-(pole+cons):-cons]
    seg_cons = df.iloc[-cons:].copy()
    pole_ret = (seg_pole.iloc[-1] / max(seg_pole.iloc[0], 1e-9)) - 1.0
    high_slope, _ = _fit_line(seg_cons["High"].values)
    low_slope, _ = _fit_line(seg_cons["Low"].values)
    width_start = float(seg_cons["High"].iloc[0] - seg_cons["Low"].iloc[0])
    width_end = float(seg_cons["High"].iloc[-1] - seg_cons["Low"].iloc[-1])
    converging = width_end < width_start * 0.85 if width_start > 0 else False

    if pole_ret > 0.08 and high_slope < 0 and low_slope > 0 and converging:
        return _pattern_match(
            "Flama",
            "Boğa Flaması",
            len(df) - (pole + cons),
            len(df) - 1,
            [
                {"pos": len(df) - (pole + cons), "time": df.index[-(pole+cons)], "price": float(seg_pole.iloc[0])},
                {"pos": len(df) - cons - 1, "time": df.index[-cons-1], "price": float(seg_pole.iloc[-1])},
                {"pos": len(df) - 1, "time": df.index[-1], "price": float(seg_cons["Close"].iloc[-1])},
            ],
            "Güçlü hareket sonrası daralan kısa üçgen.",
        )
    if pole_ret < -0.08 and high_slope < 0 and low_slope > 0 and converging:
        return _pattern_match(
            "Flama",
            "Ayı Flaması",
            len(df) - (pole + cons),
            len(df) - 1,
            [
                {"pos": len(df) - (pole + cons), "time": df.index[-(pole+cons)], "price": float(seg_pole.iloc[0])},
                {"pos": len(df) - cons - 1, "time": df.index[-cons-1], "price": float(seg_pole.iloc[-1])},
                {"pos": len(df) - 1, "time": df.index[-1], "price": float(seg_cons["Close"].iloc[-1])},
            ],
            "Güçlü düşüş sonrası daralan kısa üçgen.",
        )
    return None

def _detect_cup_handle(df: pd.DataFrame):
    if len(df) < 60:
        return None
    lookback = min(90, len(df))
    use = df.tail(lookback).copy()
    close = use["Close"].astype(float).values
    min_idx = int(np.argmin(close))
    if not (lookback * 0.2 <= min_idx <= lookback * 0.75):
        return None
    left_rim = float(np.max(close[:max(min_idx,1)]))
    right_rim = float(np.max(close[min_idx:])) 
    low_p = float(np.min(close))
    rims_close = _safe_rel_diff(left_rim, right_rim) <= 0.08
    depth_ok = min(left_rim, right_rim) > low_p * 1.12
    if not (rims_close and depth_ok):
        return None

    right_rim_pos = int(np.argmax(close[min_idx:]) + min_idx)
    handle_part = close[right_rim_pos:]
    if len(handle_part) < 4:
        return None
    handle_drop = (np.max(handle_part) - handle_part[-1]) / max(np.max(handle_part), 1e-9)
    if handle_drop <= 0.12:
        return _pattern_match(
            "Fincan Kulp",
            "Fincan Kulp",
            len(df) - lookback,
            len(df) - 1,
            [
                {"pos": len(df) - lookback, "time": use.index[0], "price": float(close[0])},
                {"pos": len(df) - lookback + min_idx, "time": use.index[min_idx], "price": low_p},
                {"pos": len(df) - lookback + right_rim_pos, "time": use.index[right_rim_pos], "price": float(close[right_rim_pos])},
                {"pos": len(df) - 1, "time": use.index[-1], "price": float(close[-1])},
            ],
            "Yuvarlak dip sonrası kısa ve sığ kulp geri çekilmesi.",
        )
    return None

def _detect_triangle(df: pd.DataFrame):
    lookback = min(40, len(df))
    if lookback < 18:
        return None
    use = df.tail(lookback).copy()
    highs, lows = _get_pivots_with_pos(use, left=2, right=2)
    recent_highs = highs[-4:]
    recent_lows = lows[-4:]

    if len(recent_highs) >= 2 and len(recent_lows) >= 2:
        high_prices = [x["price"] for x in recent_highs]
        low_prices = [x["price"] for x in recent_lows]
        highs_flat = (max(high_prices) - min(high_prices)) / max(max(high_prices), 1e-9) <= 0.03
        lows_rising = recent_lows[-1]["price"] > recent_lows[0]["price"] * 1.02
        if highs_flat and lows_rising:
            pts = recent_highs[:2] + recent_lows[:2]
            return _pattern_match(
                "Yükselen / Alçalan Üçgen",
                "Yükselen Üçgen",
                len(df) - lookback,
                len(df) - 1,
                pts,
                "Direnç yatay, dipler yükseliyor.",
            )

        lows_flat = (max(low_prices) - min(low_prices)) / max(max(low_prices), 1e-9) <= 0.03
        highs_falling = recent_highs[-1]["price"] < recent_highs[0]["price"] * 0.98
        if lows_flat and highs_falling:
            pts = recent_highs[:2] + recent_lows[:2]
            return _pattern_match(
                "Yükselen / Alçalan Üçgen",
                "Alçalan Üçgen",
                len(df) - lookback,
                len(df) - 1,
                pts,
                "Destek yatay, tepeler alçalıyor.",
            )
    return None

def _detect_rectangle(df: pd.DataFrame):
    lookback = min(35, len(df))
    if lookback < 15:
        return None
    use = df.tail(lookback).copy()
    high_band = float(use["High"].quantile(0.9))
    low_band = float(use["Low"].quantile(0.1))
    width = (high_band - low_band) / max(use["Close"].mean(), 1e-9)
    upper_touches = int((use["High"] >= high_band * 0.995).sum())
    lower_touches = int((use["Low"] <= low_band * 1.005).sum())
    if width <= 0.15 and upper_touches >= 2 and lower_touches >= 2:
        return _pattern_match(
            "Dikdörtgen",
            "Dikdörtgen",
            len(df) - lookback,
            len(df) - 1,
            [
                {"pos": len(df) - lookback, "time": use.index[0], "price": low_band},
                {"pos": len(df) - 1, "time": use.index[-1], "price": high_band},
            ],
            "Yatay destek/direnç aralığında sıkışma.",
        )
    return None

def detect_chart_patterns(df: pd.DataFrame) -> Dict[str, Optional[Dict[str, Any]]]:
    if df is None or df.empty or len(df) < 25:
        return {name: None for name in CHART_PATTERN_GROUPS}

    use = df[["Open", "High", "Low", "Close", "Volume"]].dropna().copy()
    if len(use) < 25:
        return {name: None for name in CHART_PATTERN_GROUPS}

    detectors = [
        _detect_head_shoulders,
        _detect_double_top_bottom,
        _detect_wedge,
        _detect_triple_top_bottom,
        _detect_rounding_bottom,
        _detect_flag,
        _detect_pennant,
        _detect_cup_handle,
        _detect_triangle,
        _detect_rectangle,
    ]

    out = {name: None for name in CHART_PATTERN_GROUPS}
    for fn in detectors:
        try:
            match = fn(use)
            if match and out.get(match["group"]) is None:
                out[match["group"]] = match
        except Exception:
            continue
    return out

@st.cache_data(ttl=30 * 60, show_spinner=False)
def scan_chart_patterns_for_symbol(selected_ticker: str, timeframe: str) -> Dict[str, Any]:
    tf = str(timeframe)
    period_map = {"1d": "2y", "1wk": "5y"}
    period_used = period_map.get(tf, "2y")
    try:
        sdf = load_data_cached(selected_ticker, period_used, tf, end_date=None, force_latest=False)
    except Exception as e:
        return {"error": f"Veri alınamadı: {e}", "matches": {}, "df": pd.DataFrame(), "timeframe": tf}

    sdf = _flatten_yf(sdf)
    if sdf is None or sdf.empty:
        return {"error": "Veri gelmedi.", "matches": {}, "df": pd.DataFrame(), "timeframe": tf}

    matches = detect_chart_patterns(sdf)
    return {"error": None, "matches": matches, "df": sdf, "timeframe": tf}

def build_chart_pattern_figure(df: pd.DataFrame, match: Dict[str, Any], ticker_label: str, timeframe_label: str) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=df.index,
        open=df["Open"],
        high=df["High"],
        low=df["Low"],
        close=df["Close"],
        name=ticker_label,
    ))

    start_pos = max(int(match["start_pos"]), 0)
    end_pos = min(int(match["end_pos"]), len(df) - 1)
    start_time = df.index[start_pos]
    end_time = df.index[end_pos]

    fig.add_vrect(
        x0=start_time,
        x1=end_time,
        fillcolor="rgba(30, 144, 255, 0.15)",
        line_width=1,
        line_color="dodgerblue",
        annotation_text=match["subtype"],
        annotation_position="top left",
    )

    point_x, point_y, point_text = [], [], []
    for p in match.get("points", []):
        pos = int(p.get("pos", 0))
        if 0 <= pos < len(df):
            point_x.append(df.index[pos])
            point_y.append(float(p.get("price", df["Close"].iloc[pos])))
            point_text.append(f'{match["subtype"]}<br>{df.index[pos]}<br>{float(p.get("price", 0)):.2f}')

    if point_x:
        fig.add_trace(go.Scatter(
            x=point_x,
            y=point_y,
            mode="markers+text",
            text=["●"] * len(point_x),
            textposition="top center",
            marker=dict(size=11, color="dodgerblue"),
            hovertext=point_text,
            hoverinfo="text",
            name="Formasyon Noktaları",
        ))

    low_band = float(df["Low"].iloc[start_pos:end_pos+1].min())
    high_band = float(df["High"].iloc[start_pos:end_pos+1].max())
    fig.add_hrect(
        y0=low_band,
        y1=high_band,
        fillcolor="rgba(30, 144, 255, 0.05)",
        line_width=0,
    )

    fig.update_layout(
        height=520,
        title=f'{ticker_label} — {match["subtype"]} ({timeframe_label})',
        xaxis_rangeslider_visible=False,
        yaxis_title="Fiyat",
        xaxis_title="Tarih",
    )
    return fig


# =============================
# FINANCIAL SNAPSHOT HELPERS
# =============================
FINANCIAL_METRIC_INFO = {
    "roe": "Özsermaye Karlılığı (ROE): Şirketin özkaynağını ne kadar verimli kullandığını gösterir. Genel olarak daha yüksek ROE daha güçlü kârlılık anlamına gelir.",
    "revenue_growth": "Net Satış Büyümesi: Satışların bir önceki döneme göre büyüme hızını gösterir. Artış, talep ve operasyonel genişleme açısından olumlu yorumlanabilir.",
    "ebitda": "FAVÖK (EBITDA): Faiz, vergi, amortisman ve itfa öncesi kâr. Şirketin ana faaliyetlerinden yarattığı operasyonel kârlılığı ölçmekte kullanılır.",
    "ebitda_margin": "FAVÖK Marjı: EBITDA / Satışlar. Operasyonel verimlilik ve fiyatlama gücü hakkında fikir verir. Daha yüksek marj genelde daha güçlüdür.",
    "debt_to_equity": "Borç / Özsermaye: Toplam borcun özkaynağa oranıdır. Çok yüksek oranlar bilanço riskini artırabilir; genelde daha düşük olması tercih edilir.",
    "current_ratio": "Cari Oran: Dönen varlıklar / kısa vadeli yükümlülükler. Kısa vadeli ödeme gücünü gösterir. 1'in üzeri çoğu zaman daha sağlıklı kabul edilir.",
    "net_profit_margin": "Net Kar Marjı: Net kâr / satışlar. Şirketin satışından ne kadar net kâr bıraktığını gösterir. Daha yüksek marj genelde daha iyidir.",
    "free_cash_flow": "Serbest Nakit Akışı (FCF): Faaliyetlerden gelen nakitten yatırımlar düşüldükten sonra geriye kalan nakit. Sürdürülebilirlik için kritik göstergedir.",
    "pe": "F/K (P/E): Piyasa değeri / net kâr. Aynı sektör içindeki şirketlerle kıyaslanır. Genel yaklaşımda daha düşük çarpan daha ucuz değerleme anlamına gelebilir.",
    "pb": "PD/DD (P/B): Piyasa değeri / özkaynak. Özellikle banka ve varlık yoğun şirketlerde önemli bir değerleme çarpanıdır.",
    "net_debt_ebitda": "Net Borç / FAVÖK: Net borcun operasyonel kâra göre büyüklüğünü gösterir. Daha düşük oran genelde daha sağlıklı borçluluk anlamına gelir.",
    "shares_outstanding": "Dolaşımdaki Hisse Sayısı: Şirketin yatırımcılar tarafından taşınan toplam hisse miktarını gösterir. Artış genelde sulanma (dilution), düşüş ise geri alım etkisi anlamına gelebilir; bu nedenle çoğu durumda daha düşük veya yatay seyir daha olumlu yorumlanır.",
    "eps": "Hisse Başına Kar (EPS): Net kârın hisse başına düşen kısmını gösterir. Genel olarak daha yüksek EPS daha güçlü kârlılık anlamına gelir; ancak hisse sayısındaki artış/azalış da bu metriği etkileyebilir.",
    "altman_z": "Altman Z-Skoru: İşletmenin bilanço gücü ve iflas riskini özetleyen bileşik bir skordur. Genel yaklaşımda 3 üstü daha güçlü, 1.8 altı daha riskli kabul edilir.",
    "piotroski_f": "Piotroski F-Skoru: Karlılık, bilanço gücü ve operasyonel verimlilik üzerinden 0-9 arası kalite skorudur. Genel yaklaşımda 7-9 güçlü, 0-3 zayıf kabul edilir.",
}

def _html_escape(value: Any) -> str:
    s = "" if value is None else str(value)
    return (
        s.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )

def make_fin_metric_help_html(metric_key: str) -> str:
    tip = _html_escape(FINANCIAL_METRIC_INFO.get(metric_key, "Finansal oran açıklaması bulunamadı."))
    return f'<span title="{tip}" style="cursor:help; font-weight:700; color:#d97706;"> ? </span>'

def _pick_series_from_rows(stmt_df: pd.DataFrame, row_names: List[str]) -> pd.Series:
    if stmt_df is None or stmt_df.empty:
        return pd.Series(dtype=float)
    for name in row_names:
        if name in stmt_df.columns:
            return stmt_df[name]
    return pd.Series(dtype=float)

def _safe_div_series(a: pd.Series, b: pd.Series) -> pd.Series:
    if a is None or len(a) == 0:
        return pd.Series(dtype=float)
    if b is None or len(b) == 0:
        return pd.Series(index=a.index, dtype=float)
    aa, bb = a.align(b, join="outer")
    out = aa.astype(float) / bb.astype(float).replace(0, np.nan)
    return out.replace([np.inf, -np.inf], np.nan)

def _money_fmt(val: float) -> str:
    try:
        if val is None or not np.isfinite(val):
            return "N/A"
        abs_v = abs(float(val))
        if abs_v >= 1_000_000_000_000:
            return f"{val/1_000_000_000_000:.2f}T"
        if abs_v >= 1_000_000_000:
            return f"{val/1_000_000_000:.2f}B"
        if abs_v >= 1_000_000:
            return f"{val/1_000_000:.2f}M"
        if abs_v >= 1_000:
            return f"{val/1_000:.2f}K"
        return f"{val:.2f}"
    except Exception:
        return "N/A"

def _fmt_fin_value(metric_key: str, val: float) -> str:
    if val is None or not np.isfinite(val):
        return "N/A"
    if metric_key in {"roe", "revenue_growth", "ebitda_margin", "net_profit_margin"}:
        return f"{val * 100:.2f}%"
    if metric_key in {"ebitda", "free_cash_flow", "shares_outstanding"}:
        return _money_fmt(float(val))
    if metric_key in {"debt_to_equity", "current_ratio", "pe", "pb", "net_debt_ebitda", "altman_z"}:
        return f"{float(val):.2f}"
    if metric_key in {"piotroski_f"}:
        return f"{int(round(float(val)))}"
    return f"{float(val):.2f}"

def _compute_delta_pct(current: float, previous: float) -> float:
    if current is None or previous is None:
        return np.nan
    if not np.isfinite(current) or not np.isfinite(previous):
        return np.nan
    if previous == 0:
        return np.nan
    return ((current - previous) / abs(previous)) * 100.0

def _statement_to_period_df(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    out = df.T.copy()
    try:
        out.index = pd.to_datetime(out.index)
    except Exception:
        pass
    out = out.sort_index(ascending=False)
    return out


@st.cache_data(ttl=30 * 60, show_spinner=False)
def fetch_financial_snapshot_analysis(symbol: str, selected_market: str = "USA", statement_mode: str = "quarterly", quarterly_compare_mode: str = "prev_quarter") -> Dict[str, Any]:
    ticker_norm = normalize_ticker(symbol, selected_market)
    t = yf.Ticker(ticker_norm)

    def _get_attr(obj, names: List[str]):
        for n in names:
            try:
                v = getattr(obj, n)
                if v is not None and not (hasattr(v, "empty") and v.empty):
                    return v
            except Exception:
                continue
        return pd.DataFrame()

    if statement_mode == "quarterly":
        income_raw = _get_attr(t, ["quarterly_income_stmt", "quarterly_financials"])
        balance_raw = _get_attr(t, ["quarterly_balance_sheet"])
        cash_raw = _get_attr(t, ["quarterly_cashflow"])
    else:
        income_raw = _get_attr(t, ["income_stmt", "financials"])
        balance_raw = _get_attr(t, ["balance_sheet"])
        cash_raw = _get_attr(t, ["cashflow"])

    income_df = _statement_to_period_df(income_raw)
    balance_df = _statement_to_period_df(balance_raw)
    cash_df = _statement_to_period_df(cash_raw)

    base_index = income_df.index
    if len(base_index) == 0:
        base_index = balance_df.index
    if len(base_index) == 0:
        base_index = cash_df.index
    if len(base_index) == 0:
        return {"error": "Seçilen hisse için bilanço / gelir tablosu verisi bulunamadı.", "table_html": "", "summary": {}}

    periods = pd.Index(sorted(pd.to_datetime(base_index).unique(), reverse=True))[:4]

    def _find_reference_value(series: pd.Series, curr_dt: pd.Timestamp) -> float:
        try:
            s = pd.Series(series.copy())
            s.index = pd.to_datetime(s.index)
            s = s.sort_index()
            if s.empty:
                return np.nan

            if statement_mode == "quarterly":
                if quarterly_compare_mode == "prev_period_legacy":
                    prev_candidates = [idx for idx in s.index if pd.to_datetime(idx) < pd.to_datetime(curr_dt)]
                    if not prev_candidates:
                        return np.nan
                    ref_idx = max(prev_candidates)
                    return safe_float(s.loc[ref_idx])

                if quarterly_compare_mode == "same_quarter_prev_year":
                    target_period = pd.Period(curr_dt, freq="Q") - 4
                else:
                    target_period = pd.Period(curr_dt, freq="Q") - 1

                matches = [idx for idx in s.index if pd.Period(idx, freq="Q") == target_period]
                if not matches:
                    return np.nan
                ref_idx = max(matches)
                return safe_float(s.loc[ref_idx])

            target_period = pd.Period(curr_dt, freq="Y") - 1
            matches = [idx for idx in s.index if pd.Period(idx, freq="Y") == target_period]
            if not matches:
                return np.nan
            ref_idx = max(matches)
            return safe_float(s.loc[ref_idx])
        except Exception:
            return np.nan

    revenue = _pick_series_from_rows(income_df, ["Total Revenue", "Operating Revenue", "Revenue"]).reindex(periods)
    net_income = _pick_series_from_rows(income_df, ["Net Income", "Net Income Common Stockholders", "Net Income Including Noncontrolling Interests"]).reindex(periods)
    ebitda = _pick_series_from_rows(income_df, ["EBITDA", "Ebitda"]).reindex(periods)

    shares_series = _pick_series_from_rows(
        income_df,
        [
            "Diluted Average Shares",
            "Diluted Weighted Average Shares",
            "Basic Average Shares",
            "Weighted Average Shares",
            "Weighted Average Shares Diluted",
            "Weighted Average Shares Basic",
        ],
    ).reindex(periods)

    if shares_series.empty or shares_series.isna().all():
        shares_series = _pick_series_from_rows(
            balance_df,
            [
                "Ordinary Shares Number",
                "Share Issued",
                "Common Stock Shares Outstanding",
                "Common Stock",
            ],
        ).reindex(periods)

    equity = _pick_series_from_rows(balance_df, ["Stockholders Equity", "Common Stock Equity", "Total Equity Gross Minority Interest", "Total Equity"]).reindex(periods)
    total_debt = _pick_series_from_rows(balance_df, ["Total Debt", "Total Borrowings"]).reindex(periods)
    if total_debt.empty or total_debt.isna().all():
        ltd = _pick_series_from_rows(balance_df, ["Long Term Debt", "Long Term Debt And Capital Lease Obligation"]).reindex(periods)
        std = _pick_series_from_rows(balance_df, ["Current Debt", "Current Debt And Capital Lease Obligation", "Short Long Term Debt"]).reindex(periods)
        total_debt = ltd.fillna(0) + std.fillna(0)

    current_assets = _pick_series_from_rows(balance_df, ["Current Assets", "Total Current Assets"]).reindex(periods)
    current_liabilities = _pick_series_from_rows(balance_df, ["Current Liabilities", "Total Current Liabilities"]).reindex(periods)
    cash = _pick_series_from_rows(balance_df, ["Cash And Cash Equivalents", "Cash Cash Equivalents And Short Term Investments", "Cash And Short Term Investments"]).reindex(periods)

    op_cf = _pick_series_from_rows(cash_df, ["Operating Cash Flow", "Cash Flow From Continuing Operating Activities", "Total Cash From Operating Activities"]).reindex(periods)
    capex = _pick_series_from_rows(cash_df, ["Capital Expenditure", "Capital Expenditures"]).reindex(periods)
    free_cash_flow = _pick_series_from_rows(cash_df, ["Free Cash Flow"]).reindex(periods)
    if free_cash_flow.empty or free_cash_flow.isna().all():
        free_cash_flow = op_cf.fillna(0) + capex.fillna(0)

    revenue_growth = revenue.astype(float).pct_change(periods=-1)
    ebitda_margin = _safe_div_series(ebitda, revenue)
    current_ratio = _safe_div_series(current_assets, current_liabilities)
    net_profit_margin = _safe_div_series(net_income, revenue)

    if statement_mode == "quarterly":
        roe_income = net_income.astype(float) * 4.0
        ni_ttm = net_income.sort_index().rolling(4).sum().sort_index(ascending=False).reindex(periods)
        ebitda_ref = ebitda.sort_index().rolling(4).sum().sort_index(ascending=False).reindex(periods)
        fcf_ref = free_cash_flow.sort_index().rolling(4).sum().sort_index(ascending=False).reindex(periods)
    else:
        roe_income = net_income.astype(float)
        ni_ttm = net_income.astype(float)
        ebitda_ref = ebitda.astype(float)
        fcf_ref = free_cash_flow.astype(float)

    roe = _safe_div_series(roe_income, equity)
    debt_to_equity = _safe_div_series(total_debt, equity)
    net_debt = total_debt.astype(float).fillna(0) - cash.astype(float).fillna(0)

    try:
        info = t.info or {}
    except Exception:
        info = {}

    current_shares_out = safe_float(info.get("sharesOutstanding"))
    current_price = safe_float(info.get("currentPrice") or info.get("regularMarketPrice"))
    current_market_cap = safe_float(info.get("marketCap"))
    if (not np.isfinite(current_shares_out)) and np.isfinite(current_market_cap) and np.isfinite(current_price) and current_price > 0:
        current_shares_out = current_market_cap / current_price

    shares_series = shares_series.astype(float)
    if shares_series.empty or shares_series.isna().all():
        shares_series = pd.Series(index=periods, data=current_shares_out, dtype=float)
    else:
        shares_series = shares_series.reindex(periods)
        shares_series = shares_series.ffill().bfill()
        if np.isfinite(current_shares_out):
            shares_series = shares_series.fillna(float(current_shares_out))

    eps_series = _pick_series_from_rows(
        income_df,
        ["Diluted EPS", "Basic EPS", "Reported EPS", "EPS"]
    ).reindex(periods)
    if eps_series.empty or eps_series.isna().all():
        eps_series = _safe_div_series(net_income.astype(float), shares_series.astype(float))
    else:
        eps_series = eps_series.astype(float).reindex(periods)
        eps_series = eps_series.fillna(_safe_div_series(net_income.astype(float), shares_series.astype(float)))

    hist_prices = load_data_cached(ticker_norm, "5y", "1d", end_date=None, force_latest=False)
    close_series = hist_prices["Close"].sort_index() if hist_prices is not None and not hist_prices.empty and "Close" in hist_prices.columns else pd.Series(dtype=float)

    market_caps = pd.Series(index=periods, dtype=float)
    for dt in periods:
        px = close_series.asof(pd.to_datetime(dt)) if not close_series.empty else np.nan
        period_shares = shares_series.loc[dt] if dt in shares_series.index else np.nan
        if np.isfinite(period_shares) and np.isfinite(px):
            market_caps.loc[dt] = float(px) * float(period_shares)
        elif np.isfinite(current_market_cap):
            market_caps.loc[dt] = float(current_market_cap)
        else:
            market_caps.loc[dt] = np.nan

    pe = _safe_div_series(market_caps, ni_ttm)
    pb = _safe_div_series(market_caps, equity)
    net_debt_ebitda = _safe_div_series(net_debt, ebitda_ref)

    total_assets = _pick_series_from_rows(balance_df, ["Total Assets"]).reindex(periods).astype(float)
    total_liabilities = _pick_series_from_rows(balance_df, ["Total Liabilities Net Minority Interest", "Total Liabilities", "Total Non Current Liabilities Net Minority Interest"]).reindex(periods).astype(float)
    retained_earnings = _pick_series_from_rows(balance_df, ["Retained Earnings", "RetainedEarnings"]).reindex(periods).astype(float)
    ebit_series = _pick_series_from_rows(income_df, ["EBIT", "Operating Income", "Operating Income As Reported"]).reindex(periods).astype(float)
    gross_profit = _pick_series_from_rows(income_df, ["Gross Profit"]).reindex(periods).astype(float)
    long_term_debt = _pick_series_from_rows(balance_df, ["Long Term Debt", "Long Term Debt And Capital Lease Obligation"]).reindex(periods).astype(float)

    if statement_mode == "quarterly":
        revenue_ref = revenue.sort_index().rolling(4).sum().sort_index(ascending=False).reindex(periods).astype(float)
        cfo_ref = op_cf.sort_index().rolling(4).sum().sort_index(ascending=False).reindex(periods).astype(float)
        gross_profit_ref = gross_profit.sort_index().rolling(4).sum().sort_index(ascending=False).reindex(periods).astype(float)
        ebit_ref_z = ebit_series.sort_index().rolling(4).sum().sort_index(ascending=False).reindex(periods).astype(float)
    else:
        revenue_ref = revenue.astype(float)
        cfo_ref = op_cf.astype(float)
        gross_profit_ref = gross_profit.astype(float)
        ebit_ref_z = ebit_series.astype(float)

    working_capital = current_assets.astype(float) - current_liabilities.astype(float)
    altman_z = (
        1.2 * _safe_div_series(working_capital, total_assets)
        + 1.4 * _safe_div_series(retained_earnings, total_assets)
        + 3.3 * _safe_div_series(ebit_ref_z, total_assets)
        + 0.6 * _safe_div_series(market_caps, total_liabilities)
        + 1.0 * _safe_div_series(revenue_ref, total_assets)
    )

    roa_series = _safe_div_series(ni_ttm.astype(float), total_assets)
    gross_margin = _safe_div_series(gross_profit_ref.astype(float), revenue_ref.astype(float))
    asset_turnover = _safe_div_series(revenue_ref.astype(float), total_assets)
    lt_debt_ratio = _safe_div_series(long_term_debt.astype(float), total_assets)

    piotroski_f = (
        (roa_series > 0).astype(int)
        + (cfo_ref > 0).astype(int)
        + (roa_series > roa_series.shift(-1)).astype(int)
        + (cfo_ref > ni_ttm.astype(float)).astype(int)
        + (lt_debt_ratio < lt_debt_ratio.shift(-1)).astype(int)
        + (current_ratio > current_ratio.shift(-1)).astype(int)
        + (shares_series.astype(float) <= shares_series.astype(float).shift(-1)).astype(int)
        + (gross_margin > gross_margin.shift(-1)).astype(int)
        + (asset_turnover > asset_turnover.shift(-1)).astype(int)
    ).astype(float)

    metric_rows = [
        {"label": "Özsermaye Karlılığı (ROE)", "key": "roe", "series": roe, "higher_better": True},
        {"label": "Net Satış Büyümesi", "key": "revenue_growth", "series": revenue_growth, "higher_better": True},
        {"label": "FAVÖK (EBITDA)", "key": "ebitda", "series": ebitda, "higher_better": True},
        {"label": "FAVÖK Marjı", "key": "ebitda_margin", "series": ebitda_margin, "higher_better": True},
        {"label": "Borç / Özsermaye", "key": "debt_to_equity", "series": debt_to_equity, "higher_better": False},
        {"label": "Cari Oran", "key": "current_ratio", "series": current_ratio, "higher_better": True},
        {"label": "Net Kar Marjı", "key": "net_profit_margin", "series": net_profit_margin, "higher_better": True},
        {"label": "Serbest Nakit Akışı", "key": "free_cash_flow", "series": free_cash_flow, "higher_better": True},
        {"label": "F/K (P/E)", "key": "pe", "series": pe, "higher_better": False},
        {"label": "PD/DD (P/B)", "key": "pb", "series": pb, "higher_better": False},
        {"label": "Net Borç / FAVÖK", "key": "net_debt_ebitda", "series": net_debt_ebitda, "higher_better": False},
        {"label": "Dolaşımdaki Hisse Sayısı", "key": "shares_outstanding", "series": shares_series, "higher_better": False},
        {"label": "Hisse Başına Kar (EPS)", "key": "eps", "series": eps_series, "higher_better": True},
        {"label": "Altman Z-Skoru", "key": "altman_z", "series": altman_z, "higher_better": True},
        {"label": "Piotroski F-Skoru", "key": "piotroski_f", "series": piotroski_f, "higher_better": True},
    ]

    period_labels = [pd.to_datetime(x).strftime("%Y-%m-%d") for x in periods]

    html_parts = []
    html_parts.append(
        '''
        <style>
        .fin-table-wrap {overflow-x:auto; margin-top:8px;}
        .fin-table {width:100%; border-collapse:collapse; font-size:13px;}
        .fin-table th, .fin-table td {border:1px solid #e5e7eb; padding:8px 10px; vertical-align:top;}
        .fin-table th {background:#f8fafc; font-weight:700; text-align:center;}
        .fin-label {font-weight:700;}
        .delta-green {color:#16a34a; font-weight:700; font-size:12px;}
        .delta-red {color:#dc2626; font-weight:700; font-size:12px;}
        .delta-gray {color:#6b7280; font-weight:700; font-size:12px;}
        .val-main {display:block; font-weight:600;}
        </style>
        <div class="fin-table-wrap"><table class="fin-table">
        '''
    )
    html_parts.append("<thead><tr><th>Metrix</th>" + "".join([f"<th>{_html_escape(lbl)}</th>" for lbl in period_labels]) + "</tr></thead><tbody>")

    for row in metric_rows:
        series = row["series"].reindex(periods)
        label_html = f'{_html_escape(row["label"])} {make_fin_metric_help_html(row["key"])}'
        row_html = [f"<tr><td class='fin-label'>{label_html}</td>"]
        vals = list(series.values)
        for i, val in enumerate(vals):
            value_str = _fmt_fin_value(row["key"], float(val)) if pd.notna(val) else "N/A"
            delta_html = "<span class='delta-gray'>—</span>"
            if pd.notna(val):
                try:
                    curr_dt = pd.to_datetime(periods[i])
                    ref_val = _find_reference_value(row["series"], curr_dt)
                except Exception:
                    ref_val = np.nan

                if pd.notna(ref_val):
                    delta_pct = _compute_delta_pct(float(val), float(ref_val))
                    if np.isfinite(delta_pct):
                        better = float(val) > float(ref_val) if row["higher_better"] else float(val) < float(ref_val)
                        css = "delta-green" if better else "delta-red"
                        delta_html = f"<span class='{css}'>{delta_pct:+.1f}%</span>"
            row_html.append(f"<td><span class='val-main'>{_html_escape(value_str)}</span>{delta_html}</td>")
        row_html.append("</tr>")
        html_parts.append("".join(row_html))

    html_parts.append("</tbody></table></div>")
    table_html = "".join(html_parts)

    latest_idx = periods[0]
    latest_revenue_growth = revenue_growth.loc[latest_idx] if latest_idx in revenue_growth.index else np.nan
    latest_net_margin = net_profit_margin.loc[latest_idx] if latest_idx in net_profit_margin.index else np.nan
    latest_roe = roe.loc[latest_idx] if latest_idx in roe.index else np.nan
    latest_de_ratio = debt_to_equity.loc[latest_idx] if latest_idx in debt_to_equity.index else np.nan
    latest_nd_ebitda = net_debt_ebitda.loc[latest_idx] if latest_idx in net_debt_ebitda.index else np.nan
    latest_equity = equity.loc[latest_idx] if latest_idx in equity.index else np.nan
    latest_earnings = ni_ttm.loc[latest_idx] if latest_idx in ni_ttm.index else np.nan
    latest_ebitda = ebitda_ref.loc[latest_idx] if latest_idx in ebitda_ref.index else np.nan
    latest_net_debt = net_debt.loc[latest_idx] if latest_idx in net_debt.index else np.nan
    latest_fcf = fcf_ref.loc[latest_idx] if latest_idx in fcf_ref.index else np.nan
    latest_shares = shares_series.loc[latest_idx] if latest_idx in shares_series.index else current_shares_out
    latest_altman_z = altman_z.loc[latest_idx] if latest_idx in altman_z.index else np.nan
    latest_piotroski_f = piotroski_f.loc[latest_idx] if latest_idx in piotroski_f.index else np.nan

    def _clip(val, low, high):
        try:
            return max(low, min(high, float(val)))
        except Exception:
            return low

    target_pe = _clip(8 + max(safe_float(latest_revenue_growth), -0.2) * 25 + max(safe_float(latest_net_margin), 0) * 30 + max(safe_float(latest_roe), 0) * 10 - max(safe_float(latest_de_ratio), 0) * 1.5, 5, 25)
    target_pb = _clip(0.8 + max(safe_float(latest_roe), 0) * 8 + max(safe_float(latest_net_margin), 0) * 4 - max(safe_float(latest_de_ratio), 0) * 0.3, 0.5, 5)
    target_ev_ebitda = _clip(5 + max(safe_float(latest_revenue_growth), -0.2) * 15 + max(safe_float(latest_net_margin), 0) * 12 - max(safe_float(latest_nd_ebitda), 0) * 0.5, 4, 16)

    fair_components = []
    fair_labels = []
    fair_weights = []

    if np.isfinite(latest_earnings) and latest_earnings > 0:
        fair_components.append(float(latest_earnings) * float(target_pe))
        fair_labels.append("F/K tabanlı")
        fair_weights.append(0.40)
    if np.isfinite(latest_equity) and latest_equity > 0:
        fair_components.append(float(latest_equity) * float(target_pb))
        fair_labels.append("PD/DD tabanlı")
        fair_weights.append(0.30)
    if np.isfinite(latest_ebitda) and latest_ebitda > 0:
        fair_components.append((float(latest_ebitda) * float(target_ev_ebitda)) - float(latest_net_debt if np.isfinite(latest_net_debt) else 0))
        fair_labels.append("EV/FAVÖK tabanlı")
        fair_weights.append(0.30)

    fair_market_cap = np.nan
    fair_breakdown = []
    if fair_components:
        total_w = sum(fair_weights)
        fair_market_cap = sum(v * w for v, w in zip(fair_components, fair_weights)) / total_w if total_w > 0 else np.nan
        fair_breakdown = [{"method": lab, "value": val} for lab, val in zip(fair_labels, fair_components)]

    # Simple DCF (education-grade) based on latest FCF, 5Y fade, terminal growth
    dcf_fair_market_cap = np.nan
    dcf_breakdown = {}
    if np.isfinite(latest_fcf) and latest_fcf > 0:
        dcf_growth = _clip(max(safe_float(latest_revenue_growth), 0.0) * 0.60 + 0.05, 0.02, 0.15)
        dcf_discount = 0.12
        dcf_terminal_growth = 0.03

        pv_stage = 0.0
        fcf_proj = float(latest_fcf)
        for year in range(1, 6):
            if dcf_growth > dcf_terminal_growth:
                step = (dcf_growth - dcf_terminal_growth) / 4.0
                growth_y = max(dcf_terminal_growth, dcf_growth - step * (year - 1))
            else:
                growth_y = dcf_growth
            fcf_proj = fcf_proj * (1.0 + growth_y)
            pv_stage += fcf_proj / ((1.0 + dcf_discount) ** year)

        terminal_fcf = fcf_proj * (1.0 + dcf_terminal_growth)
        if dcf_discount > dcf_terminal_growth:
            terminal_value = terminal_fcf / (dcf_discount - dcf_terminal_growth)
            pv_terminal = terminal_value / ((1.0 + dcf_discount) ** 5)
            dcf_equity_value = pv_stage + pv_terminal - float(latest_net_debt if np.isfinite(latest_net_debt) else 0)
            if np.isfinite(dcf_equity_value) and dcf_equity_value > 0:
                dcf_fair_market_cap = dcf_equity_value
                dcf_breakdown = {
                    "latest_fcf": float(latest_fcf),
                    "discount_rate": dcf_discount,
                    "growth_rate": dcf_growth,
                    "terminal_growth": dcf_terminal_growth,
                }

    latest_market_cap = market_caps.loc[latest_idx] if latest_idx in market_caps.index else np.nan
    upside_pct = _compute_delta_pct(float(fair_market_cap), float(latest_market_cap)) if np.isfinite(fair_market_cap) and np.isfinite(latest_market_cap) else np.nan
    fair_share_price = (float(fair_market_cap) / float(latest_shares)) if np.isfinite(fair_market_cap) and np.isfinite(latest_shares) and latest_shares > 0 else np.nan
    dcf_fair_share_price = (float(dcf_fair_market_cap) / float(latest_shares)) if np.isfinite(dcf_fair_market_cap) and np.isfinite(latest_shares) and latest_shares > 0 else np.nan

    def _altman_label(z: float) -> str:
        if not np.isfinite(z):
            return "N/A"
        if z >= 3.0:
            return "Güçlü"
        if z >= 1.8:
            return "Gri Bölge"
        return "Riskli"

    def _piotroski_label(f: float) -> str:
        if not np.isfinite(f):
            return "N/A"
        if f >= 7:
            return "Güçlü"
        if f >= 4:
            return "Orta"
        return "Zayıf"

    summary = {
        "ticker": ticker_norm,
        "statement_mode": statement_mode,
        "latest_period": pd.to_datetime(latest_idx).strftime("%Y-%m-%d") if latest_idx is not None else "",
        "current_market_cap": latest_market_cap,
        "fair_market_cap": fair_market_cap,
        "fair_share_price": fair_share_price,
        "dcf_fair_market_cap": dcf_fair_market_cap,
        "dcf_fair_share_price": dcf_fair_share_price,
        "upside_pct": upside_pct,
        "target_pe": target_pe,
        "target_pb": target_pb,
        "target_ev_ebitda": target_ev_ebitda,
        "latest_shares": latest_shares,
        "fair_breakdown": fair_breakdown,
        "dcf_breakdown": dcf_breakdown,
        "altman_z": latest_altman_z,
        "altman_state": _altman_label(latest_altman_z),
        "piotroski_f": latest_piotroski_f,
        "piotroski_state": _piotroski_label(latest_piotroski_f),
    }
    return {"error": None, "table_html": table_html, "summary": summary}


# =============================
# INDICATOR STATISTICS HELPERS
# =============================
def _cross_up(a: pd.Series, b: pd.Series) -> pd.Series:
    return (a > b) & (a.shift(1) <= b.shift(1))


def _cross_down(a: pd.Series, b: pd.Series) -> pd.Series:
    return (a < b) & (a.shift(1) >= b.shift(1))


def _rising(s: pd.Series) -> pd.Series:
    return s > s.shift(1)


def _falling(s: pd.Series) -> pd.Series:
    return s < s.shift(1)


def _rolling_divergence_flags(
    close: pd.Series,
    indicator: pd.Series,
    kind: str = "bull",
    lookback: int = 30,
    recent_bars: int = 2,
    volume: Optional[pd.Series] = None,
) -> pd.Series:
    """
    İndikatör İstatistik sekmesi için uyumsuzlukları tararken olayı tespit edildiği bara değil,
    gerçek pivot barına yazar. Volume varsa pivot temsilcisini seçerken kullanır.
    """
    flags = pd.Series(False, index=close.index)
    if close is None or indicator is None or len(close) < lookback + 3:
        return flags

    for i in range(lookback, len(close)):
        c_slice = close.iloc[: i + 1]
        ind_slice = indicator.iloc[: i + 1]
        vol_slice = volume.iloc[: i + 1] if volume is not None else None
        try:
            if kind == "bull":
                ok, bars_ago = check_bullish_divergence(c_slice, ind_slice, lookback=lookback, volume=vol_slice)
            else:
                ok, bars_ago = check_bearish_divergence(c_slice, ind_slice, lookback=lookback, volume=vol_slice)

            if ok and bars_ago <= recent_bars:
                pivot_pos = i - int(bars_ago)
                if 0 <= pivot_pos < len(flags):
                    flags.iloc[pivot_pos] = True
        except Exception:
            pass

    return flags


def _prepare_indicator_stats_frames(symbol: str, selected_market: str, cfg: dict) -> Dict[str, Any]:
    ticker_norm = normalize_ticker(symbol, selected_market)
    try:
        daily_raw = yf.download(ticker_norm, period="7y", interval="1d", auto_adjust=True, progress=False)
        weekly_raw = yf.download(ticker_norm, period="10y", interval="1wk", auto_adjust=True, progress=False)
    except Exception as e:
        return {"error": f"Veri indirilemedi: {e}", "daily": pd.DataFrame(), "weekly": pd.DataFrame(), "ticker": ticker_norm}

    daily = _flatten_yf(daily_raw)
    weekly = _flatten_yf(weekly_raw)

    if daily.empty and weekly.empty:
        return {"error": "İstatistik analizi için veri gelmedi.", "daily": pd.DataFrame(), "weekly": pd.DataFrame(), "ticker": ticker_norm}

    if not daily.empty:
        daily = build_features(daily, cfg)
        daily["EMA13_High"] = ema(daily["High"], 13)
        daily["EMA13_Low"] = ema(daily["Low"], 13)
        daily["EMA13_Close"] = ema(daily["Close"], 13)
        daily["EMA11"] = ema(daily["Close"], 11)
        daily["EMA22"] = ema(daily["Close"], 22)
        daily["FI"] = force_index(daily["Close"], daily["Volume"])
        daily["FI_EMA13"] = ema(daily["FI"], 13)
        daily["FI_EMA2"] = ema(daily["FI"], 2)
        daily["RSI13"] = rsi(daily["Close"], 13)
        daily["STOCH5"], daily["STOCH5_D"] = stochastic(daily["High"], daily["Low"], daily["Close"], k_period=5, d_period=3)
        daily["ER_EMA13"], daily["BULL_POWER"], daily["BEAR_POWER"] = elder_ray(daily["High"], daily["Low"], daily["Close"], 13)
        daily["ADX14"], daily["PDI14"], daily["MDI14"] = adx_indicator(daily["High"], daily["Low"], daily["Close"], 14)

    if not weekly.empty:
        weekly = build_features(weekly, cfg)
        weekly["EMA13"] = ema(weekly["Close"], 13)
        weekly["EMA26"] = ema(weekly["Close"], 26)
        weekly["ADX14"], weekly["PDI14"], weekly["MDI14"] = adx_indicator(weekly["High"], weekly["Low"], weekly["Close"], 14)

    return {"error": None, "daily": daily, "weekly": weekly, "ticker": ticker_norm}


def get_indicator_stats_catalog() -> Dict[str, Dict[str, Any]]:
    return {
        # Price action
        "KANGAROO_BULL": {"label": "Fiyat Aksiyonu • Kangaroo Bull", "timeframe": "1d", "direction": "bull"},
        "KANGAROO_BEAR": {"label": "Fiyat Aksiyonu • Kangaroo Bear", "timeframe": "1d", "direction": "bear"},
        "PATTERN_HAMMER": {"label": "Fiyat Aksiyonu • Hammer", "timeframe": "1d", "direction": "bull"},
        "PATTERN_HANGING_MAN": {"label": "Fiyat Aksiyonu • Hanging Man", "timeframe": "1d", "direction": "bear"},
        "PATTERN_SHOOTING_STAR": {"label": "Fiyat Aksiyonu • Shooting Star", "timeframe": "1d", "direction": "bear"},
        "PATTERN_INV_HAMMER": {"label": "Fiyat Aksiyonu • Inverted Hammer", "timeframe": "1d", "direction": "bull"},
        "PATTERN_ENGULFING_BULL": {"label": "Fiyat Aksiyonu • Bullish Engulfing", "timeframe": "1d", "direction": "bull"},
        "PATTERN_ENGULFING_BEAR": {"label": "Fiyat Aksiyonu • Bearish Engulfing", "timeframe": "1d", "direction": "bear"},
        "PATTERN_HARAMI_BULL": {"label": "Fiyat Aksiyonu • Bullish Harami", "timeframe": "1d", "direction": "bull"},
        "PATTERN_HARAMI_BEAR": {"label": "Fiyat Aksiyonu • Bearish Harami", "timeframe": "1d", "direction": "bear"},
        "PATTERN_MARUBOZU_BULL": {"label": "Fiyat Aksiyonu • Bullish Marubozu", "timeframe": "1d", "direction": "bull"},
        "PATTERN_MARUBOZU_BEAR": {"label": "Fiyat Aksiyonu • Bearish Marubozu", "timeframe": "1d", "direction": "bear"},
        "PATTERN_TWEEZER_BOTTOM": {"label": "Fiyat Aksiyonu • Tweezer Bottom", "timeframe": "1d", "direction": "bull"},
        "PATTERN_TWEEZER_TOP": {"label": "Fiyat Aksiyonu • Tweezer Top", "timeframe": "1d", "direction": "bear"},
        "PATTERN_PIERCING": {"label": "Fiyat Aksiyonu • Piercing Pattern", "timeframe": "1d", "direction": "bull"},
        "PATTERN_DARK_CLOUD": {"label": "Fiyat Aksiyonu • Dark Cloud", "timeframe": "1d", "direction": "bear"},
        "PATTERN_MORNING_STAR": {"label": "Fiyat Aksiyonu • Morning Star", "timeframe": "1d", "direction": "bull"},
        "PATTERN_EVENING_STAR": {"label": "Fiyat Aksiyonu • Evening Star", "timeframe": "1d", "direction": "bear"},
        "PATTERN_LL_DOJI": {"label": "Fiyat Aksiyonu • Long-Legged Doji", "timeframe": "1d", "direction": "neutral"},
        "PATTERN_DOJI": {"label": "Fiyat Aksiyonu • Doji", "timeframe": "1d", "direction": "neutral"},
        # EMA13 touches
        "EMA13_LOW_TOUCH": {"label": "13 EMA • EMA13 Low Teması", "timeframe": "1d", "direction": "bull"},
        "EMA13_CLOSE_TOUCH": {"label": "13 EMA • EMA13 Close Teması", "timeframe": "1d", "direction": "neutral"},
        "EMA13_HIGH_TOUCH": {"label": "13 EMA • EMA13 High Teması", "timeframe": "1d", "direction": "bear"},
        # Weekly screen 1
        "W_MACD_SLOPE_UP": {"label": "1. Ekran Haftalık • MACD Histogram Eğimi Yukarı", "timeframe": "1wk", "direction": "bull"},
        "W_MACD_SLOPE_DOWN": {"label": "1. Ekran Haftalık • MACD Histogram Eğimi Aşağı", "timeframe": "1wk", "direction": "bear"},
        "W_MACD_DIV_BULL": {"label": "1. Ekran Haftalık • MACD Pozitif Uyumsuzluk", "timeframe": "1wk", "direction": "bull"},
        "W_MACD_DIV_BEAR": {"label": "1. Ekran Haftalık • MACD Negatif Uyumsuzluk", "timeframe": "1wk", "direction": "bear"},
        "W_EMA1326_AL": {"label": "1. Ekran Haftalık • EMA(13-26) AL", "timeframe": "1wk", "direction": "bull"},
        "W_EMA1326_SAT": {"label": "1. Ekran Haftalık • EMA(13-26) SAT", "timeframe": "1wk", "direction": "bear"},
        "W_ADX_AL": {"label": "1. Ekran Haftalık • ADX(14) AL", "timeframe": "1wk", "direction": "bull"},
        "W_ADX_SAT": {"label": "1. Ekran Haftalık • ADX(14) SAT", "timeframe": "1wk", "direction": "bear"},
        # Daily screen 2
        "D_EMA1122_AL": {"label": "2. Ekran Günlük • EMA(11-22) AL", "timeframe": "1d", "direction": "bull"},
        "D_EMA1122_SAT": {"label": "2. Ekran Günlük • EMA(11-22) SAT", "timeframe": "1d", "direction": "bear"},
        "D_FI_AL": {"label": "2. Ekran Günlük • Kuvvet Endeksi (FI) AL", "timeframe": "1d", "direction": "bull"},
        "D_FI_SAT": {"label": "2. Ekran Günlük • Kuvvet Endeksi (FI) SAT", "timeframe": "1d", "direction": "bear"},
        "D_RSI13_AL": {"label": "2. Ekran Günlük • RSI(13) Aşırı Satım", "timeframe": "1d", "direction": "bull"},
        "D_RSI13_SAT": {"label": "2. Ekran Günlük • RSI(13) Aşırı Alım", "timeframe": "1d", "direction": "bear"},
        "D_RSI_DIV_BULL": {"label": "2. Ekran Günlük • RSI Pozitif Uyumsuzluk", "timeframe": "1d", "direction": "bull"},
        "D_RSI_DIV_BEAR": {"label": "2. Ekran Günlük • RSI Negatif Uyumsuzluk", "timeframe": "1d", "direction": "bear"},
        "D_STOCH5_AL": {"label": "2. Ekran Günlük • Stokastik(5) Aşırı Satım", "timeframe": "1d", "direction": "bull"},
        "D_STOCH5_SAT": {"label": "2. Ekran Günlük • Stokastik(5) Aşırı Alım", "timeframe": "1d", "direction": "bear"},
        "D_STOCH_DIV_BULL": {"label": "2. Ekran Günlük • Stokastik Pozitif Uyumsuzluk", "timeframe": "1d", "direction": "bull"},
        "D_STOCH_DIV_BEAR": {"label": "2. Ekran Günlük • Stokastik Negatif Uyumsuzluk", "timeframe": "1d", "direction": "bear"},
        "D_ELDERRAY_AL": {"label": "2. Ekran Günlük • Elder-Ray AL", "timeframe": "1d", "direction": "bull"},
        "D_ELDERRAY_SAT": {"label": "2. Ekran Günlük • Elder-Ray SAT", "timeframe": "1d", "direction": "bear"},
        "D_ELDERRAY_DIV_BULL": {"label": "2. Ekran Günlük • Elder-Ray Pozitif Uyumsuzluk", "timeframe": "1d", "direction": "bull"},
        "D_ELDERRAY_DIV_BEAR": {"label": "2. Ekran Günlük • Elder-Ray Negatif Uyumsuzluk", "timeframe": "1d", "direction": "bear"},
        "D_ADX_AL": {"label": "2. Ekran Günlük • ADX(14) AL", "timeframe": "1d", "direction": "bull"},
        "D_ADX_SAT": {"label": "2. Ekran Günlük • ADX(14) SAT", "timeframe": "1d", "direction": "bear"},
    }


def build_indicator_signal_series(event_key: str, daily: pd.DataFrame, weekly: pd.DataFrame) -> Tuple[pd.Series, pd.DataFrame, str, str]:
    catalog = get_indicator_stats_catalog()
    meta = catalog[event_key]
    timeframe = meta["timeframe"]
    direction = meta["direction"]
    df = daily if timeframe == "1d" else weekly
    if df is None or df.empty:
        return pd.Series(dtype=bool), pd.DataFrame(), timeframe, direction

    sig = pd.Series(False, index=df.index)

    if event_key in df.columns:
        sig = df[event_key].fillna(0).astype(int) == 1
    elif event_key == "EMA13_LOW_TOUCH":
        sig = (df["Low"] <= df["EMA13_Low"]) & (df["Close"] >= df["EMA13_Low"])
    elif event_key == "EMA13_CLOSE_TOUCH":
        sig = (df["Low"] <= df["EMA13_Close"]) & (df["High"] >= df["EMA13_Close"])
    elif event_key == "EMA13_HIGH_TOUCH":
        sig = (df["High"] >= df["EMA13_High"]) & (df["Close"] <= df["EMA13_High"])
    elif event_key == "W_MACD_SLOPE_UP":
        hist = df["MACD_hist"]
        sig = (hist.diff() > 0) & (hist.diff().shift(1) <= 0)
    elif event_key == "W_MACD_SLOPE_DOWN":
        hist = df["MACD_hist"]
        sig = (hist.diff() < 0) & (hist.diff().shift(1) >= 0)
    elif event_key == "W_MACD_DIV_BULL":
        sig = _rolling_divergence_flags(df["Close"], df["MACD_hist"], kind="bull", lookback=30, recent_bars=2, volume=df["Volume"])
    elif event_key == "W_MACD_DIV_BEAR":
        sig = _rolling_divergence_flags(df["Close"], df["MACD_hist"], kind="bear", lookback=30, recent_bars=2, volume=df["Volume"])
    elif event_key == "W_EMA1326_AL":
        state = (df["EMA13"] > df["EMA26"]) & (df["Close"] > df["EMA13"])
        sig = state & (~state.shift(1).fillna(False))
    elif event_key == "W_EMA1326_SAT":
        state = (df["EMA13"] < df["EMA26"]) & (df["Close"] < df["EMA13"])
        sig = state & (~state.shift(1).fillna(False))
    elif event_key == "W_ADX_AL":
        state = (df["ADX14"] >= 25) & (df["PDI14"] > df["MDI14"])
        sig = state & (~state.shift(1).fillna(False))
    elif event_key == "W_ADX_SAT":
        state = (df["ADX14"] >= 25) & (df["MDI14"] > df["PDI14"])
        sig = state & (~state.shift(1).fillna(False))
    elif event_key == "D_EMA1122_AL":
        state = (df["EMA11"] > df["EMA22"]) & (df["Close"] > df["EMA11"])
        sig = state & (~state.shift(1).fillna(False))
    elif event_key == "D_EMA1122_SAT":
        state = (df["EMA11"] < df["EMA22"]) & (df["Close"] < df["EMA11"])
        sig = state & (~state.shift(1).fillna(False))
    elif event_key == "D_FI_AL":
        state = (df["FI"] > df["FI_EMA13"]) & (df["FI_EMA2"] < 0)
        sig = state & (~state.shift(1).fillna(False))
    elif event_key == "D_FI_SAT":
        state = (df["FI"] < df["FI_EMA13"]) & (df["FI_EMA2"] > 0)
        sig = state & (~state.shift(1).fillna(False))
    elif event_key == "D_RSI13_AL":
        sig = (df["RSI13"] < 30) & (df["RSI13"].shift(1) >= 30)
    elif event_key == "D_RSI13_SAT":
        sig = (df["RSI13"] > 70) & (df["RSI13"].shift(1) <= 70)
    elif event_key == "D_RSI_DIV_BULL":
        sig = _rolling_divergence_flags(df["Close"], df["RSI13"], kind="bull", lookback=30, recent_bars=2, volume=df["Volume"])
    elif event_key == "D_RSI_DIV_BEAR":
        sig = _rolling_divergence_flags(df["Close"], df["RSI13"], kind="bear", lookback=30, recent_bars=2, volume=df["Volume"])
    elif event_key == "D_STOCH5_AL":
        sig = (df["STOCH5"] < 20) & (df["STOCH5"].shift(1) >= 20)
    elif event_key == "D_STOCH5_SAT":
        sig = (df["STOCH5"] > 80) & (df["STOCH5"].shift(1) <= 80)
    elif event_key == "D_STOCH_DIV_BULL":
        sig = _rolling_divergence_flags(df["Close"], df["STOCH5"], kind="bull", lookback=30, recent_bars=2, volume=df["Volume"])
    elif event_key == "D_STOCH_DIV_BEAR":
        sig = _rolling_divergence_flags(df["Close"], df["STOCH5"], kind="bear", lookback=30, recent_bars=2, volume=df["Volume"])
    elif event_key == "D_ELDERRAY_AL":
        state = _rising(df["ER_EMA13"]) & (df["BEAR_POWER"] < 0) & (df["BEAR_POWER"] > df["BEAR_POWER"].shift(1))
        sig = state & (~state.shift(1).fillna(False))
    elif event_key == "D_ELDERRAY_SAT":
        state = _falling(df["ER_EMA13"]) & (df["BULL_POWER"] > 0) & (df["BULL_POWER"] < df["BULL_POWER"].shift(1))
        sig = state & (~state.shift(1).fillna(False))
    elif event_key == "D_ELDERRAY_DIV_BULL":
        sig = _rolling_divergence_flags(df["Close"], df["BEAR_POWER"], kind="bull", lookback=30, recent_bars=2, volume=df["Volume"])
    elif event_key == "D_ELDERRAY_DIV_BEAR":
        sig = _rolling_divergence_flags(df["Close"], df["BULL_POWER"], kind="bear", lookback=30, recent_bars=2, volume=df["Volume"])
    elif event_key == "D_ADX_AL":
        state = (df["ADX14"] >= 25) & (df["PDI14"] > df["MDI14"])
        sig = state & (~state.shift(1).fillna(False))
    elif event_key == "D_ADX_SAT":
        state = (df["ADX14"] >= 25) & (df["MDI14"] > df["PDI14"])
        sig = state & (~state.shift(1).fillna(False))

    return sig.fillna(False), df, timeframe, direction



def compute_indicator_signal_statistics(df: pd.DataFrame, signal_mask: pd.Series, direction: str = "bull", max_bars: int = 20, move_threshold: float = 0.02, timeframe: str = "1d") -> Tuple[Dict[str, Any], pd.DataFrame]:
    if df is None or df.empty or signal_mask is None or signal_mask.empty:
        return {"occurrences": 0, "worked": 0, "win_rate": np.nan}, pd.DataFrame()

    def _trend_bars_until_reversal(entry_price: float, future_close: pd.Series, sig_direction: str, fallback_direction: str = "Yukarı") -> Tuple[int, str]:
        if future_close is None or future_close.empty or not np.isfinite(entry_price):
            return 0, fallback_direction

        closes = pd.to_numeric(future_close, errors="coerce").dropna()
        if closes.empty:
            return 0, fallback_direction

        if sig_direction == "bull":
            eval_direction = "Yukarı"
        elif sig_direction == "bear":
            eval_direction = "Aşağı"
        else:
            eval_direction = fallback_direction

        prev_close = float(entry_price)
        trend_bars = 0

        if eval_direction == "Yukarı":
            for close_val in closes:
                close_val = float(close_val)
                if close_val >= prev_close:
                    trend_bars += 1
                    prev_close = close_val
                else:
                    break
        else:
            for close_val in closes:
                close_val = float(close_val)
                if close_val <= prev_close:
                    trend_bars += 1
                    prev_close = close_val
                else:
                    break

        return int(trend_bars), eval_direction

    event_positions = np.where(signal_mask.fillna(False).values)[0]
    rows = []
    bar_to_day = 7 if timeframe == "1wk" else 1

    for pos in event_positions:
        if pos >= len(df) - 2:
            continue

        entry_price = float(df["Close"].iloc[pos])
        future = df.iloc[pos + 1 : pos + 1 + int(max_bars)].copy()
        if future.empty or not np.isfinite(entry_price) or entry_price <= 0:
            continue

        up_path = (future["High"] / entry_price) - 1.0
        down_path = (future["Low"] / entry_price) - 1.0
        max_up = float(up_path.max()) if not up_path.empty else 0.0
        max_down = float(down_path.min()) if not down_path.empty else 0.0
        up_idx = int(np.argmax(up_path.values)) + 1 if len(up_path) else 0
        down_idx = int(np.argmin(down_path.values)) + 1 if len(down_path) else 0
        down_abs = abs(max_down)

        if direction == "bull":
            worked = (max_up >= move_threshold) and (max_up >= down_abs)
            dominant_dir = "Yukarı" if max_up >= down_abs else "Aşağı"
            dominant_move = max_up if max_up >= down_abs else max_down
        elif direction == "bear":
            worked = (down_abs >= move_threshold) and (down_abs >= max_up)
            dominant_dir = "Aşağı" if down_abs >= max_up else "Yukarı"
            dominant_move = -down_abs if down_abs >= max_up else max_up
        else:
            if max_up >= down_abs:
                worked = max_up >= move_threshold
                dominant_dir = "Yukarı"
                dominant_move = max_up
            else:
                worked = down_abs >= move_threshold
                dominant_dir = "Aşağı"
                dominant_move = -down_abs

        trend_bars, trend_direction = _trend_bars_until_reversal(
            entry_price=entry_price,
            future_close=future["Close"],
            sig_direction=direction,
            fallback_direction=dominant_dir,
        )

        rows.append({
            "Tarih": pd.to_datetime(df.index[pos]),
            "Giriş Fiyatı": entry_price,
            "Çalıştı": bool(worked),
            "Baskın Yön": dominant_dir,
            "Trend Yönü": trend_direction,
            "Trend Bar": int(trend_bars),
            "Trend Gün": int(trend_bars * bar_to_day),
            "Maks. Yükseliş %": max_up * 100.0,
            "Maks. Düşüş %": max_down * 100.0,
            "Baskın Hareket %": dominant_move * 100.0,
            "Yukarı Zirve Barı": int(up_idx),
            "Aşağı Dip Barı": int(down_idx),
        })

    res_df = pd.DataFrame(rows)
    if res_df.empty:
        return {
            "occurrences": 0,
            "worked": 0,
            "win_rate": np.nan,
            "avg_days": np.nan,
            "median_days": np.nan,
            "avg_up": np.nan,
            "avg_down": np.nan,
            "avg_dom": np.nan,
            "best": np.nan,
            "worst": np.nan,
        }, res_df

    summary = {
        "occurrences": int(len(res_df)),
        "worked": int(res_df["Çalıştı"].sum()),
        "win_rate": float(res_df["Çalıştı"].mean() * 100.0),
        "avg_days": float(res_df["Trend Gün"].mean()),
        "median_days": float(res_df["Trend Gün"].median()),
        "avg_up": float(res_df["Maks. Yükseliş %"].mean()),
        "avg_down": float(res_df["Maks. Düşüş %"].mean()),
        "avg_dom": float(res_df["Baskın Hareket %"].mean()),
        "best": float(res_df["Baskın Hareket %"].max()),
        "worst": float(res_df["Baskın Hareket %"].min()),
    }
    return summary, res_df.sort_values("Tarih", ascending=False)


def build_indicator_occurrence_chart(df: pd.DataFrame, signal_mask: pd.Series, title: str = "İndikatör Oluşumları") -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=df.index,
        open=df["Open"],
        high=df["High"],
        low=df["Low"],
        close=df["Close"],
        name="Fiyat",
    ))

    if "EMA13_Close" in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df["EMA13_Close"], name="EMA13 Close", line=dict(width=1.8, color="darkorange")))
    if "EMA13_Low" in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df["EMA13_Low"], name="EMA13 Low", line=dict(width=1, color="rgba(255,165,0,0.7)")))
    if "EMA13_High" in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df["EMA13_High"], name="EMA13 High", line=dict(width=1, color="rgba(255,165,0,0.7)")))

    mask = signal_mask.fillna(False)
    if mask.any():
        fig.add_trace(go.Scatter(
            x=df.index[mask],
            y=df.loc[mask, "Close"],
            mode="markers",
            name="Oluşum",
            marker=dict(size=10, color="limegreen", symbol="circle"),
        ))

    fig.update_layout(height=520, title=title, xaxis_rangeslider_visible=False, yaxis_title="Fiyat", xaxis_title="Tarih")
    return fig

# =============================
# Tabs
# =============================


# =============================
# Historical Range Analysis Helpers
# =============================
def _slice_df_by_date_range(df_in: pd.DataFrame, start_date: datetime.date, end_date: datetime.date) -> pd.DataFrame:
    if df_in is None or df_in.empty:
        return pd.DataFrame()
    start_ts = pd.Timestamp(start_date)
    end_ts = pd.Timestamp(end_date) + pd.Timedelta(days=1) - pd.Timedelta(microseconds=1)
    return df_in.loc[(df_in.index >= start_ts) & (df_in.index <= end_ts)].copy()

def build_history_range_price_figure(plot_df: pd.DataFrame, show_patterns: bool = True, show_ema13: bool = False, show_sr_lines: bool = False) -> go.Figure:
    fig = go.Figure()
    if plot_df is None or plot_df.empty:
        fig.update_layout(height=850, title="Geçmiş Tarih Aralığı Grafiği")
        return fig

    fig.add_trace(go.Candlestick(
        x=plot_df.index,
        open=plot_df["Open"],
        high=plot_df["High"],
        low=plot_df["Low"],
        close=plot_df["Close"],
        name="Price"
    ))
    if "EMA50" in plot_df.columns:
        fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df["EMA50"], name="EMA Fast"))
    if "EMA200" in plot_df.columns:
        fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df["EMA200"], name="EMA Slow"))
    if "BB_upper" in plot_df.columns:
        fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df["BB_upper"], name="BB Upper", line=dict(dash="dot")))
    if "BB_mid" in plot_df.columns:
        fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df["BB_mid"], name="BB Mid", line=dict(dash="dot")))
    if "BB_lower" in plot_df.columns:
        fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df["BB_lower"], name="BB Lower", line=dict(dash="dot")))

    if show_ema13 and {"EMA13_High", "EMA13_Low", "EMA13_Close"}.issubset(plot_df.columns):
        fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df["EMA13_High"], name="13 EMA High", line=dict(color='rgba(255, 165, 0, 0.8)', width=1)))
        fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df["EMA13_Low"], name="13 EMA Low", fill='tonexty', fillcolor='rgba(255, 165, 0, 0.2)', line=dict(color='rgba(255, 165, 0, 0.8)', width=1)))
        fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df["EMA13_Close"], name="13 EMA Close", line=dict(color='darkorange', width=2)))

    if "ENTRY" in plot_df.columns:
        entries = plot_df[plot_df["ENTRY"] == 1]
        if not entries.empty:
            fig.add_trace(go.Scatter(x=entries.index, y=entries["Close"], mode="markers", name="ENTRY", marker=dict(symbol="triangle-up", size=10)))
    if "EXIT" in plot_df.columns:
        exits = plot_df[plot_df["EXIT"] == 1]
        if not exits.empty:
            fig.add_trace(go.Scatter(x=exits.index, y=exits["Close"], mode="markers", name="EXIT", marker=dict(symbol="triangle-down", size=10)))

    if show_patterns:
        bull_patterns = {
            "KANGAROO_BULL": "🟩🦘 LONG KANGURU",
            "PATTERN_HAMMER": "🟩🔨 HAMMER",
            "PATTERN_INV_HAMMER": "🟩🔨 INV HAMMER",
            "PATTERN_ENGULFING_BULL": "🟢 ENGULFING",
            "PATTERN_HARAMI_BULL": "🟢🤰 HARAMI",
            "PATTERN_MARUBOZU_BULL": "🟩 MARUBOZU",
            "PATTERN_TWEEZER_BOTTOM": "🟢✌️ TWEEZER",
            "PATTERN_PIERCING": "🟢🗡️ PIERCING",
            "PATTERN_MORNING_STAR": "🟢🌅 M.STAR",
            "PATTERN_LL_DOJI": "🟢⚖️ LL DOJI",
        }
        bear_patterns = {
            "KANGAROO_BEAR": "🟥🦘 SHORT KANGURU",
            "PATTERN_HANGING_MAN": "🟥🪢 HANGING M.",
            "PATTERN_SHOOTING_STAR": "🟥🌠 S.STAR",
            "PATTERN_ENGULFING_BEAR": "🔴 ENGULFING",
            "PATTERN_HARAMI_BEAR": "🔴🤰 HARAMI",
            "PATTERN_MARUBOZU_BEAR": "🟥 MARUBOZU",
            "PATTERN_TWEEZER_TOP": "🔴✌️ TWEEZER",
            "PATTERN_DARK_CLOUD": "🔴🌩️ D.CLOUD",
            "PATTERN_EVENING_STAR": "🔴🌃 E.STAR"
        }

        bull_texts = pd.Series("", index=plot_df.index)
        bear_texts = pd.Series("", index=plot_df.index)
        for col, name in bull_patterns.items():
            if col in plot_df.columns:
                mask = plot_df[col] == 1
                bull_texts[mask] += name + "<br>"
        for col, name in bear_patterns.items():
            if col in plot_df.columns:
                mask = plot_df[col] == 1
                bear_texts[mask] += name + "<br>"

        bull_texts = bull_texts.str.rstrip("<br>")
        bear_texts = bear_texts.str.rstrip("<br>")

        bull_mask = bull_texts != ""
        if bull_mask.any():
            fig.add_trace(go.Scatter(
                x=plot_df.index[bull_mask], y=plot_df["Low"][bull_mask], mode="markers+text", name="Boğa Formasyonları",
                text=bull_texts[bull_mask], textposition="bottom center",
                textfont=dict(color="green", size=10, family="Arial Black"),
                marker=dict(symbol="triangle-up", size=10, color="green", line=dict(width=1, color="DarkSlateGrey"))
            ))

        bear_mask = bear_texts != ""
        if bear_mask.any():
            fig.add_trace(go.Scatter(
                x=plot_df.index[bear_mask], y=plot_df["High"][bear_mask], mode="markers+text", name="Ayı Formasyonları",
                text=bear_texts[bear_mask], textposition="top center",
                textfont=dict(color="red", size=10, family="Arial Black"),
                marker=dict(symbol="triangle-down", size=10, color="red", line=dict(width=1, color="DarkSlateGrey"))
            ))

    if show_sr_lines:
        fig = add_support_resistance_trend_overlays(fig, plot_df)

    fig.update_layout(
        height=850,
        xaxis_rangeslider_visible=False,
        title="Geçmiş Tarih Aralığı Fiyat Grafiği + EMA + Bollinger + Sinyaller & Formasyonlar",
        yaxis_title="Fiyat",
        xaxis_title="Tarih",
    )
    return fig

def build_history_range_indicator_figures(plot_df: pd.DataFrame, benchmark_df_full: Optional[pd.DataFrame], benchmark_symbol: str):
    fig_rsi_local = go.Figure()
    fig_rsi_local.add_trace(go.Scatter(x=plot_df.index, y=plot_df["RSI"], name="RSI"))
    fig_rsi_local.add_hline(y=70, line_dash="dash", line_color="red", annotation_text="Aşırı Alım")
    fig_rsi_local.add_hline(y=30, line_dash="dash", line_color="green", annotation_text="Aşırı Satım")
    fig_rsi_local.update_layout(height=260, title="RSI (Göreceli Güç Endeksi)", yaxis_title="RSI", xaxis_title="Tarih")

    fig_macd_local = go.Figure()
    fig_macd_local.add_trace(go.Scatter(x=plot_df.index, y=plot_df["MACD"], name="MACD"))
    fig_macd_local.add_trace(go.Scatter(x=plot_df.index, y=plot_df["MACD_signal"], name="Signal"))
    fig_macd_local.add_trace(go.Bar(x=plot_df.index, y=plot_df["MACD_hist"], name="Hist"))
    fig_macd_local.update_layout(height=260, title="MACD (Moving Average Convergence Divergence)", yaxis_title="MACD", xaxis_title="Tarih")

    fig_atr_local = go.Figure()
    fig_atr_local.add_trace(go.Scatter(x=plot_df.index, y=plot_df["ATR_PCT"] * 100, name="ATR%"))
    fig_atr_local.update_layout(height=260, title="ATR % (Ortalama Gerçek Aralık / Fiyat)", yaxis_title="%", xaxis_title="Tarih")

    fig_stoch_local = go.Figure()
    if "STOCH_RSI_K" in plot_df.columns and "STOCH_RSI_D" in plot_df.columns:
        fig_stoch_local.add_trace(go.Scatter(x=plot_df.index, y=plot_df["STOCH_RSI_K"], name="Stochastic RSI K"))
        fig_stoch_local.add_trace(go.Scatter(x=plot_df.index, y=plot_df["STOCH_RSI_D"], name="Stochastic RSI D"))
        fig_stoch_local.add_hline(y=80, line_dash="dash", line_color="red", annotation_text="Aşırı Alım")
        fig_stoch_local.add_hline(y=20, line_dash="dash", line_color="green", annotation_text="Aşırı Satım")
    fig_stoch_local.update_layout(height=260, title="Stochastic RSI (K & D)", yaxis_title="Değer", xaxis_title="Tarih")

    fig_bbwidth_local = go.Figure()
    if "BB_WIDTH" in plot_df.columns:
        fig_bbwidth_local.add_trace(go.Scatter(x=plot_df.index, y=plot_df["BB_WIDTH"] * 100, name="BB % Genişlik"))
        fig_bbwidth_local.add_hline(y=2, line_dash="dash", line_color="orange", annotation_text="Sıkışma Bölgesi")
    fig_bbwidth_local.update_layout(height=260, title="Bollinger Bandı Genişliği %", yaxis_title="Genişlik %", xaxis_title="Tarih")

    fig_volratio_local = go.Figure()
    if "VOL_RATIO" in plot_df.columns:
        fig_volratio_local.add_trace(go.Bar(x=plot_df.index, y=plot_df["VOL_RATIO"], name="Hacim Oranı"))
        fig_volratio_local.add_hline(y=1.5, line_dash="dash", line_color="red", annotation_text="Anormal Hacim")
    fig_volratio_local.update_layout(height=260, title="Hacim Oranı (Son Hacim / SMA)", yaxis_title="Oran", xaxis_title="Tarih")

    plot_df_local = plot_df.copy()
    if "VOL_SMA_10" not in plot_df_local.columns:
        plot_df_local["VOL_SMA_10"] = plot_df_local["Volume"].rolling(10).mean()

    if benchmark_df_full is not None and not benchmark_df_full.empty and "Volume" in benchmark_df_full.columns:
        bench_slice = benchmark_df_full.reindex(plot_df_local.index)
        bench_vol = bench_slice["Volume"].fillna(0)
    else:
        bench_vol = pd.Series(0, index=plot_df_local.index)

    fig_vol_market_local = make_subplots(specs=[[{"secondary_y": True}]])
    fig_vol_market_local.add_trace(go.Bar(x=plot_df_local.index, y=plot_df_local["Volume"], name="Hisse Hacmi", marker_color='lightblue', opacity=0.7), secondary_y=False)
    fig_vol_market_local.add_trace(go.Scatter(x=plot_df_local.index, y=bench_vol, name=f"Endeks ({benchmark_symbol})", line=dict(color='orange', width=2)), secondary_y=True)
    fig_vol_market_local.update_layout(height=260, title="Hisse vs Endeks Hacmi", yaxis_title="Hisse", yaxis2_title="Endeks", xaxis_title="Tarih", margin=dict(l=0, r=0, t=40, b=0))

    fig_vol_2wk_local = go.Figure()
    fig_vol_2wk_local.add_trace(go.Bar(x=plot_df_local.index, y=plot_df_local["Volume"], name="Hacim", marker_color='cadetblue', opacity=0.7))
    fig_vol_2wk_local.add_trace(go.Scatter(x=plot_df_local.index, y=plot_df_local["VOL_SMA_10"], name="2 Haftalık Ort. (10 Bar)", line=dict(color='red', width=2)))
    fig_vol_2wk_local.update_layout(height=260, title="Hisse Hacmi vs 2 Haftalık Ortalama", yaxis_title="Hacim", xaxis_title="Tarih", margin=dict(l=0, r=0, t=40, b=0))

    fig_obv_local = go.Figure()
    if "OBV" in plot_df_local.columns and "OBV_EMA" in plot_df_local.columns:
        fig_obv_local.add_trace(go.Scatter(x=plot_df_local.index, y=plot_df_local["OBV"], name="OBV", line=dict(color='blue')))
        fig_obv_local.add_trace(go.Scatter(x=plot_df_local.index, y=plot_df_local["OBV_EMA"], name="OBV EMA (21)", line=dict(color='orange', dash='dot')))
    fig_obv_local.update_layout(height=260, title="On-Balance Volume (OBV)", yaxis_title="OBV", xaxis_title="Tarih", margin=dict(l=0, r=0, t=40, b=0))

    return {
        "rsi": fig_rsi_local,
        "macd": fig_macd_local,
        "atr": fig_atr_local,
        "stoch": fig_stoch_local,
        "bbwidth": fig_bbwidth_local,
        "volratio": fig_volratio_local,
        "vol_market": fig_vol_market_local,
        "vol_2wk": fig_vol_2wk_local,
        "obv": fig_obv_local,
    }



def _bist_index_mapping() -> Dict[str, str]:
    return {
        "BIST 30": "XU030.IS",
        "BIST 100": "XU100.IS",
        "BIST Tüm": "XUTUM.IS",
    }


def _resolve_bist_index_benchmark(index_ticker: str) -> str:
    if index_ticker == "XU100.IS":
        return "XUTUM.IS"
    return "XU100.IS"


def _prepare_dashboard_context_for_symbol(
    symbol: str,
    cfg: dict,
    current_interval: str,
    current_period: str,
    end_date: Optional[datetime.date],
    force_latest: bool,
    live_override: bool,
    use_market_filter: bool,
    use_htf_filter: bool,
    risk_free_rate: float,
) -> Dict[str, Any]:
    raw = load_data_cached(symbol, current_period, current_interval, end_date=end_date, force_latest=force_latest)
    if raw is None or raw.empty:
        return {"error": f"Veri gelmedi: {symbol}"}

    req = {"Open", "High", "Low", "Close", "Volume"}
    if not req.issubset(set(raw.columns)):
        return {"error": f"Gerekli OHLCV kolonları eksik: {symbol}"}

    live_info = get_live_price(symbol)
    live_price_local = live_info.get("last_price", np.nan)

    if live_override and np.isfinite(live_price_local):
        raw = apply_live_last_override_to_df(raw, live_price_local)

    feat = build_features(raw, cfg)

    market_series_local = get_bist_regime_series() if use_market_filter else None
    htf_series_local = get_higher_tf_trend_series(symbol, higher_tf_interval="1wk", ema_period=200) if use_htf_filter else None

    feat, checkpoints_local = signal_with_checkpoints(
        feat,
        cfg,
        market_filter_series=market_series_local,
        higher_tf_filter_series=htf_series_local,
    )

    feat["EMA13_High"] = ema(feat["High"], 13)
    feat["EMA13_Low"] = ema(feat["Low"], 13)
    feat["EMA13_Close"] = ema(feat["Close"], 13)

    latest_local = feat.iloc[-1]

    if int(latest_local["ENTRY"]) == 1:
        rec_local = "AL"
    elif int(latest_local["EXIT"]) == 1:
        rec_local = "SAT"
    else:
        rec_local = "AL (Güçlü Trend)" if latest_local["SCORE"] >= 80 else ("İZLE (Orta)" if latest_local["SCORE"] >= 60 else "UZAK DUR")

    benchmark_symbol_local = _resolve_bist_index_benchmark(symbol)
    benchmark_df_local = load_data_cached(benchmark_symbol_local, current_period, current_interval, end_date=end_date, force_latest=False)
    benchmark_returns_local = benchmark_df_local["Close"].pct_change().dropna() if benchmark_df_local is not None and not benchmark_df_local.empty and "Close" in benchmark_df_local.columns else None

    eq_local, tdf_local, metrics_local = backtest_long_only(
        feat,
        cfg,
        risk_free_annual=risk_free_rate,
        benchmark_returns=benchmark_returns_local,
    )

    tp_local = target_price_band(feat)
    rr_info_local = rr_from_atr_stop(latest_local, tp_local, cfg)
    spec_local = detect_speculation(feat)

    price_fig_local = build_history_range_price_figure(
        feat,
        show_patterns=st.session_state.get("show_chart_patterns", True),
        show_ema13=st.session_state.get("show_ema13_channel", False),
        show_sr_lines=st.session_state.get("show_sr_lines_index_center", False),
    )
    indicator_figs_local = build_history_range_indicator_figures(feat, benchmark_df_local, benchmark_symbol_local)
    vpvr_fig_local, poc_price_local = build_volume_profile_figure(feat, bins=28, lookback=min(len(feat), 220))

    return {
        "error": None,
        "df": feat,
        "raw": raw,
        "latest": latest_local,
        "live_info": live_info,
        "live_price": live_price_local,
        "checkpoints": checkpoints_local,
        "recommendation": rec_local,
        "eq": eq_local,
        "tdf": tdf_local,
        "metrics": metrics_local,
        "tp": tp_local,
        "rr_info": rr_info_local,
        "spec": spec_local,
        "price_fig": price_fig_local,
        "indicator_figs": indicator_figs_local,
        "vpvr_fig": vpvr_fig_local,
        "poc_price": poc_price_local,
        "benchmark_symbol": benchmark_symbol_local,
        "benchmark_df": benchmark_df_local,
    }


def _render_dashboard_like_context(ctx: Dict[str, Any], display_name: str, interval_label: str, period_label: str):
    if not ctx or ctx.get("error"):
        st.warning(ctx.get("error", "Veri hazırlanamadı."))
        return

    df_local = ctx["df"]
    latest_local = ctx["latest"]
    live_price_local = ctx["live_price"]
    checkpoints_local = ctx["checkpoints"]
    rec_local = ctx["recommendation"]
    metrics_local = ctx["metrics"]
    tp_local = ctx["tp"]
    rr_info_local = ctx["rr_info"]
    spec_local = ctx["spec"]
    vpvr_fig_local = ctx["vpvr_fig"]
    poc_price_local = ctx["poc_price"]

    st.subheader(f"📊 {display_name} Dashboard")
    render_page_education_expander([("dashboard_page","Dashboard Nasıl Okunur?"),("ema","EMA"),("bollinger","Bollinger Bantları"),("rsi","RSI"),("macd","MACD"),("atr_pct","ATR%"),("volume_ratio","Hacim Oranı"),("obv","OBV"),("support_resistance","Destek/Direnç"),("target_band","Hedef Fiyat Bandı"),("backtest","Backtest"),("monte_carlo","Monte Carlo"),("vpvr","VPVR"),("poc","POC"),("poc_distance","POC Uzaklık %"),("sector_relative","Sektöre Göre Pahalı/Ucuz")])
    c1, c2, c3, c4, c5, c6, c7, c8 = st.columns(8)
    c1.metric("Endeks", display_name)
    c2.metric("Interval", interval_label)
    c3.metric("Periyot", period_label)
    c4.metric("Daily Close", f"{latest_local['Close']:.2f}")
    c5.metric("Live/Last", f"{live_price_local:.2f}" if np.isfinite(live_price_local) else "N/A")
    c6.metric("Skor", f"{latest_local['SCORE']:.0f}/100")
    c7.metric("Sinyal", rec_local)
    c8.metric("ATR%", f"{latest_local['ATR_PCT']*100:.2f}%" if pd.notna(latest_local.get("ATR_PCT", np.nan)) else "N/A")

    sma_fast_local = latest_local.get("SMA_FAST", np.nan)
    sma_slow_local = latest_local.get("SMA_SLOW", np.nan)
    sma_rel_local = "Üstünde" if pd.notna(sma_fast_local) and pd.notna(sma_slow_local) and sma_fast_local > sma_slow_local else ("Altında" if pd.notna(sma_fast_local) and pd.notna(sma_slow_local) else "N/A")
    sma_bias_local = "LONG ✅" if sma_rel_local == "Üstünde" else ("SHORT ⚠️" if sma_rel_local == "Altında" else "N/A")
    sm1, sm2 = st.columns(2)
    sm1.metric("SMA Hızlı/Yavaş", sma_rel_local, delta=sma_bias_local)
    sm2.metric("ATR Rejimi", "Yüksek" if pd.notna(latest_local.get("ATR_PCT", np.nan)) and latest_local.get("ATR_PCT", 0) > 0.04 else "Normal")

    st.subheader("📊 Aşırı Alım / Spekülasyon Göstergeleri")
    ob1, ob2, ob3, ob4 = st.columns(4)
    ob1.metric("Aşırı Alım Skoru", f"{spec_local['overbought_score']}/100")
    ob2.metric("Aşırı Satım Skoru", f"{spec_local['oversold_score']}/100")
    ob3.metric("Spekülasyon Skoru", f"{spec_local['speculation_score']}/100")
    ob4.metric("Genel Karar", spec_local["verdict"])

    with st.expander("Detaylı Aşırı Alım/Spekülasyon Analizi", expanded=False):
        for _, v in spec_local.get("details", {}).items():
            st.write(f"• {v}")

    st.subheader("✅ Kontrol Noktaları (Son Bar)")
    cp_cols = st.columns(3)
    cp_items = list(checkpoints_local.items())
    for i, (k, v) in enumerate(cp_items):
        with cp_cols[i % 3]:
            st.metric(k, "✅" if v else "❌")

    st.subheader("🎯 Hedef Fiyat Bandı (Senaryo)")
    base_px = float(tp_local["base"])
    rr_str = fmt_rr(rr_info_local.get("rr"))
    bull = tp_local.get("bull")
    bear = tp_local.get("bear")
    b1, b2, b3 = st.columns(3)
    b1.metric("Base", f"{base_px:.2f}")
    b2.metric("Bull Band", f"{bull[0]:.2f} → {bull[1]:.2f}" if bull else "N/A")
    b3.metric("Bear Band", f"{bear[0]:.2f} → {bear[1]:.2f} | RR {rr_str}" if bear else f"N/A | RR {rr_str}")

    st.subheader("📊 Fiyat + EMA + Bollinger + Sinyaller")
    st.plotly_chart(ctx["price_fig"], use_container_width=True)

    st.subheader("🛠️ Grafik Analiz Araçları")
    idx_tools_col1 = st.columns(1)[0]
    if idx_tools_col1.button(
        "Destek / Direnç Trendini Aç / Kapat",
        key=f"idx_sr_toggle_{display_name}",
        use_container_width=True,
        help="Grafiğe otomatik destek, direnç ve trend çizgilerini ekler veya kaldırır.",
    ):
        st.session_state.show_sr_lines_index_center = not st.session_state.show_sr_lines_index_center
        st.rerun()

    figs = ctx["indicator_figs"]
    st.subheader("📉 RSI / MACD / ATR%")
    colA, colB, colC = st.columns(3)
    with colA:
        st.plotly_chart(figs["rsi"], use_container_width=True)
    with colB:
        st.plotly_chart(figs["macd"], use_container_width=True)
    with colC:
        st.plotly_chart(figs["atr"], use_container_width=True)

    st.subheader("📊 Stochastic RSI / Bollinger Genişliği / Hacim Oranı")
    colD, colE, colF = st.columns(3)
    with colD:
        st.plotly_chart(figs["stoch"], use_container_width=True)
    with colE:
        st.plotly_chart(figs["bbwidth"], use_container_width=True)
    with colF:
        st.plotly_chart(figs["volratio"], use_container_width=True)

    st.subheader("📊 Hacim ve Trend Karşılaştırmaları")
    colV1, colV2, colV3 = st.columns(3)
    with colV1:
        st.plotly_chart(figs["vol_market"], use_container_width=True)
    with colV2:
        st.plotly_chart(figs["vol_2wk"], use_container_width=True)
    with colV3:
        st.plotly_chart(figs["obv"], use_container_width=True)

    st.subheader("🧪 Backtest Özeti (Long-only + Scale Out + Time Stop)")
    render_help_badges([("backtest","Backtest"),("monte_carlo","Monte Carlo"),("sharpe","Sharpe"),("sortino","Sortino"),("calmar","Calmar"),("ulcer","Ulcer"),("kelly","Kelly"),("beta","Beta"),("information_ratio","Info Ratio")])
    m1, m2, m3, m4, m5, m6, m7, m8 = st.columns(8)
    m1.metric("Total Return", f"{metrics_local['Total Return']*100:.1f}%")
    m2.metric("Ann Return", f"{metrics_local['Annualized Return']*100:.1f}%")
    m3.metric("Sharpe", f"{metrics_local['Sharpe']:.2f}")
    m4.metric("Max DD", f"{metrics_local['Max Drawdown']*100:.1f}%")
    m5.metric("Trades", f"{metrics_local['Trades']}")
    m6.metric("Win Rate", f"{metrics_local['Win Rate']*100:.1f}%")
    m7.metric("Beta", f"{metrics_local['Beta']:.2f}")
    m8.metric("Info Ratio", f"{metrics_local['Information Ratio']:.2f}")

    eq_fig_local = go.Figure()
    eq_fig_local.add_trace(go.Scatter(x=ctx["eq"].index, y=ctx["eq"].values, name="Equity"))
    eq_fig_local.update_layout(height=320, title=f"{display_name} Backtest Sermaye Eğrisi", yaxis_title="Sermaye", xaxis_title="Tarih")
    st.plotly_chart(eq_fig_local, use_container_width=True)

    mc_local = backtest_monte_carlo(ctx["eq"], n_sims=250)
    if not mc_local.get("error"):
        im1, im2, im3, im4 = st.columns(4)
        im1.metric("MC Median", f"{mc_local['median_return_pct']:.2f}%")
        im2.metric("MC P10", f"{mc_local['p10_return_pct']:.2f}%")
        im3.metric("MC P90", f"{mc_local['p90_return_pct']:.2f}%")
        im4.metric("MC Median DD", f"{mc_local['median_max_dd_pct']:.2f}%")
        st.plotly_chart(mc_local["figure"], use_container_width=True)

    with st.expander("Trade listesi", expanded=False):
        st.dataframe(ctx["tdf"], use_container_width=True, height=240)

    st.subheader("📦 Hacim Profili (VPVR / POC)")
    vp1, vp2, vp3 = st.columns(3)
    vp1.metric("POC", f"{poc_price_local:.2f}" if np.isfinite(poc_price_local) else "N/A")
    close_ref_local = float(df_local["Close"].iloc[-1]) if not df_local.empty else np.nan
    poc_dist_local = ((poc_price_local / close_ref_local) - 1.0) * 100.0 if np.isfinite(poc_price_local) and np.isfinite(close_ref_local) and close_ref_local != 0 else np.nan
    vp2.metric("POC Uzaklık %", f"{poc_dist_local:+.2f}%" if np.isfinite(poc_dist_local) else "N/A")
    vp3.metric("Profil Bar Sayısı", f"{min(len(df_local), 220)}")
    st.plotly_chart(vpvr_fig_local, use_container_width=True)


def _render_future_price_panel_for_df(
    local_df: pd.DataFrame,
    symbol_label: str,
    current_interval: str,
    current_period: str,
    session_prefix: str,
):
    st.header("🔮 Future Price")
    st.caption("Makine öğrenmesi tabanlı çoklu model karşılaştırması ile seçili sembolün mevcut zaman diliminde ileri bar kapanış fiyat tahmini üretir.")

    if local_df is None or local_df.empty:
        st.info("Future Price için önce veri hazırlanmalıdır.")
        return

    st.markdown(f"**Aktif sembol:** `{symbol_label}`  |  **Aktif zaman dilimi:** `{current_interval}`  |  **Aktif periyot:** `{current_period}`")
    future_horizon = st.number_input(
        "Kaç bar/gün sonrası tahmin yapılsın?",
        min_value=1,
        max_value=250,
        value=5,
        step=1,
        help="Mevcut seçili zaman dilimine göre çalışır. Örn. interval 1d ise 5 = 5 gün/bar sonrası kapanış tahmini.",
        key=f"{session_prefix}_future_horizon_input",
    )

    run_future_price = st.button("🤖 Future Price Tahmini Yap", key=f"{session_prefix}_run_future_price", use_container_width=True)

    result_key = f"{session_prefix}_future_price_result"
    if run_future_price:
        with st.spinner("Makine öğrenmesi modelleri eğitiliyor, walk-forward test yapılıyor ve tahmin üretiliyor..."):
            st.session_state[result_key] = future_price_ml_forecast(local_df, int(future_horizon))
            st.session_state[f"{session_prefix}_future_price_horizon"] = int(future_horizon)

    fp_result = st.session_state.get(result_key)
    if fp_result and not fp_result.get("error"):
        compare_df = fp_result["compare_df"].copy()
        best_model_name = fp_result["best_model_name"]

        st.subheader("🏁 Model Karşılaştırma Panosu")
        st.dataframe(compare_df, use_container_width=True, height=220)

        model_options = compare_df["Model"].tolist()
        best_idx = model_options.index(best_model_name) if best_model_name in model_options else 0
        selected_model_name = st.selectbox(
            "Grafik ve detay için model seç",
            options=model_options,
            index=best_idx,
            key=f"{session_prefix}_future_price_selected_model",
            help="Sayfadaki tüm modeller görünür. Buradan seçtiğin modele göre grafik, olasılıklar ve önem sıralaması güncellenir.",
        )
        models_map = fp_result.get("models", {})
        if not isinstance(models_map, dict):
            models_map = {}
        selected_model = models_map.get(selected_model_name, fp_result.get("best_model_result", {}))
        if not selected_model:
            st.warning("Seçilen modelin detay kaydı bulunamadı; en iyi modelin sonucu gösteriliyor.")
            selected_model = fp_result.get("best_model_result", {})
        horizon_value = int(fp_result.get("horizon_bars", future_horizon))

        fp_current_price = _fp_safe_value(selected_model, fp_result, "current_price", np.nan)
        fp_pred_price = _fp_safe_value(selected_model, fp_result, "predicted_price", np.nan)
        fp_delta_pct = _fp_safe_value(selected_model, fp_result, "delta_pct", np.nan)
        fp_pred_low = _fp_safe_value(selected_model, fp_result, "predicted_low", np.nan)
        fp_pred_high = _fp_safe_value(selected_model, fp_result, "predicted_high", np.nan)
        fp_confidence = _fp_safe_value(selected_model, fp_result, "confidence_score", np.nan)
        fp_mae = _fp_safe_value(selected_model, fp_result, "mae", np.nan)
        fp_rmse = _fp_safe_value(selected_model, fp_result, "rmse", np.nan)
        fp_mape = _fp_safe_value(selected_model, fp_result, "mape", np.nan)
        fp_train_rows = _fp_safe_value(selected_model, fp_result, "train_rows", np.nan)
        fp_test_rows = _fp_safe_value(selected_model, fp_result, "test_rows", np.nan)

        fp_c1, fp_c2, fp_c3, fp_c4, fp_c5 = st.columns(5)
        fp_c1.metric("Mevcut Kapanış", _fp_fmt_num(fp_current_price, 2))
        fp_c2.metric(f"{int(horizon_value)} Bar Sonrası Tahmin", _fp_fmt_num(fp_pred_price, 2))
        fp_c3.metric("Tahmini Değişim", _fp_fmt_num(fp_delta_pct, 2, "%") if pd.notna(fp_delta_pct) else "N/A")
        fp_c4.metric("Tahmin Bandı", f"{_fp_fmt_num(fp_pred_low, 2)} - {_fp_fmt_num(fp_pred_high, 2)}" if pd.notna(fp_pred_low) and pd.notna(fp_pred_high) else "N/A")
        fp_c5.metric("Güven Skoru", _fp_fmt_num(fp_confidence, 1, "%") if pd.notna(fp_confidence) else "N/A")

        fp_m1, fp_m2, fp_m3, fp_m4 = st.columns(4)
        fp_m1.metric("MAE", _fp_fmt_num(fp_mae, 4))
        fp_m2.metric("RMSE", _fp_fmt_num(fp_rmse, 4))
        fp_m3.metric("MAPE", "%" + _fp_fmt_num(fp_mape, 2) if pd.notna(fp_mape) else "N/A")
        fp_m4.metric("Eğitim/Test Satırı", f"{int(fp_train_rows) if pd.notna(fp_train_rows) else 'N/A'}/{int(fp_test_rows) if pd.notna(fp_test_rows) else 'N/A'}")

        fp_p1, fp_p2, fp_p3, fp_p4 = st.columns(4)
        fp_p1.metric("Yükseliş Olasılığı", f"%{selected_model['up_prob']:.1f}" if pd.notna(selected_model['up_prob']) else "N/A")
        fp_p2.metric("Düşüş Olasılığı", f"%{selected_model['down_prob']:.1f}" if pd.notna(selected_model['down_prob']) else "N/A")
        fp_p3.metric("Yatay Olasılığı", f"%{selected_model['flat_prob']:.1f}" if pd.notna(selected_model['flat_prob']) else "N/A")
        fp_p4.metric("Yön Doğruluğu", f"%{selected_model['direction_acc']:.1f}" if pd.notna(selected_model['direction_acc']) else "N/A")

        st.subheader("🌡️ Rejim Özeti")
        rg1, rg2 = st.columns(2)
        rg1.metric("Trend Rejimi", fp_result.get("trend_regime", "N/A"))
        rg2.metric("Volatilite Rejimi", fp_result.get("vol_regime", "N/A"))

        st.subheader("📈 Seçili Modele Göre Grafik")
        future_fig = go.Figure()
        recent_actual = selected_model.get("recent_actual")
        test_actual_series = selected_model.get("test_actual_series")
        test_pred_series = selected_model.get("test_pred_series")
        if recent_actual is not None and not recent_actual.empty:
            future_fig.add_trace(go.Scatter(x=recent_actual.index, y=recent_actual.values, name="Geçmiş Kapanış", line=dict(color="royalblue", width=2)))
        if test_actual_series is not None and not test_actual_series.empty:
            future_fig.add_trace(go.Scatter(x=test_actual_series.index, y=test_actual_series.values, name="Test Gerçek", line=dict(color="seagreen", width=2)))
        if test_pred_series is not None and not test_pred_series.empty:
            future_fig.add_trace(go.Scatter(x=test_pred_series.index, y=test_pred_series.values, name=f"{selected_model_name} Test Tahmini", line=dict(color="darkorange", width=2, dash="dot")))

        future_x = recent_actual.index[-1] if recent_actual is not None and not recent_actual.empty else pd.Timestamp.now()
        future_fig.add_trace(go.Scatter(
            x=[future_x],
            y=[selected_model['predicted_price']],
            mode="markers",
            name="Tahmini Gelecek Fiyat",
            marker=dict(size=14, color="darkorange", symbol="diamond"),
        ))
        if pd.notna(selected_model['predicted_low']) and pd.notna(selected_model['predicted_high']):
            future_fig.add_trace(go.Scatter(
                x=[future_x, future_x],
                y=[selected_model['predicted_low'], selected_model['predicted_high']],
                mode="lines",
                name="Tahmin Bandı",
                line=dict(width=8, color="rgba(255,140,0,0.35)"),
            ))
            future_fig.add_hrect(
                y0=selected_model['predicted_low'],
                y1=selected_model['predicted_high'],
                fillcolor="rgba(255,165,0,0.08)",
                line_width=0,
                annotation_text="Olası Fiyat Bandı",
                annotation_position="top left",
            )
        future_fig.add_hline(y=selected_model['predicted_price'], line_dash="dash", line_color="darkorange", annotation_text=f"Tahmin: {selected_model['predicted_price']:.2f}")
        future_fig.update_layout(height=460, title=f"Future Price Tahmini - {selected_model_name} ({int(fp_result['horizon_bars'])} bar sonrası)", xaxis_title="Tarih", yaxis_title="Fiyat")
        st.plotly_chart(future_fig, use_container_width=True)

        st.subheader("🧠 Feature Importance / Etki Sıralaması")
        fi_df = selected_model.get("feature_importance_df")
        if fi_df is not None and not fi_df.empty:
            fi_fig = go.Figure()
            fi_top = fi_df.head(12).iloc[::-1]
            fi_fig.add_trace(go.Bar(x=fi_top["Importance"], y=fi_top["Feature"], orientation="h", name="Önem"))
            fi_fig.update_layout(height=420, title=f"{selected_model_name} - En Etkili Özellikler", xaxis_title="Önem", yaxis_title="Feature")
            st.plotly_chart(fi_fig, use_container_width=True)
            st.dataframe(fi_df, use_container_width=True, height=260)
        else:
            st.info("Bu model için yorumlanabilir feature importance bilgisi üretilemedi.")

        st.subheader("📏 Horizon Kalite Özeti")
        horizon_quality_df = fp_result.get("horizon_quality_df")
        if horizon_quality_df is not None and not horizon_quality_df.empty:
            st.dataframe(horizon_quality_df, use_container_width=True, height=220)

        st.info(
            f"Seçili model `{selected_model_name}` son kullanılabilir barı `{selected_model.get('last_feature_index', 'N/A')}` üzerinden tahmin üretti. "
            "Bu çıktı eğitim amaçlıdır; kesin fiyat bilgisi değildir. Sayfadaki ⭐ işaretli model, mevcut karşılaştırmada en düşük hata ile öne çıkmıştır."
        )
    elif fp_result and fp_result.get("error"):
        st.warning(fp_result["error"])


def _render_triple_screen_panel_for_symbol(
    symbol: str,
    display_name: str,
    session_prefix: str,
    live_override: bool,
):
    st.header("📺 Üçlü Ekran Trading Sistemi (Triple Screen)")
    st.caption(f"{display_name} için Dr. Alexander Elder'in 3 Ekranlı sistemine dayanan trend, osilatör ve giriş seviyesi analizi.")

    run_key = f"{session_prefix}_run_triple_screen"
    if st.button("Üçlü Ekran Verilerini Getir ve Analiz Et", key=f"{session_prefix}_run_triple_btn"):
        st.session_state[run_key] = True

    if st.session_state.get(run_key, False):
        with st.spinner("3 Ekran verileri hesaplanıyor (1W, 1D, 1H)..."):
            df_1w = load_data_cached(symbol, "2y", "1wk")
            df_1d = load_data_cached(symbol, "1y", "1d", force_latest=force_latest_candle)
            df_1h = load_data_cached(symbol, "60d", "1h")

            live_info_local = get_live_price(symbol)
            live_price_local = live_info_local.get("last_price", np.nan)
            if live_override:
                df_1w = apply_live_last_override_to_df(df_1w, live_price_local)
                df_1d = apply_live_last_override_to_df(df_1d, live_price_local)
                df_1h = apply_live_last_override_to_df(df_1h, live_price_local)

            if df_1w.empty or df_1d.empty or df_1h.empty:
                st.error("Bazı zaman dilimleri için veri çekilemedi (API gecikmesi veya sembol kaynaklı sorun olabilir).")
            else:
                t_screen1, t_screen2, t_screen3 = st.tabs(["1. Ekran (Haftalık)", "2. Ekran (Günlük)", "3. Ekran (1 Saatlik)"])

                with t_screen1:
                    st.subheader("1. Ekran: Haftalık (Ana Trend)")
                    _, _, m_hist = macd(df_1w["Close"])

                    ema_1w_13 = ema(df_1w["Close"], 13)
                    ema_1w_26 = ema(df_1w["Close"], 26)

                    last_close_1w = df_1w["Close"].iloc[-1]
                    if ema_1w_13.iloc[-1] > ema_1w_26.iloc[-1] and last_close_1w > ema_1w_13.iloc[-1]:
                        ema1w_sig = "AL"
                    elif ema_1w_13.iloc[-1] < ema_1w_26.iloc[-1] and last_close_1w < ema_1w_13.iloc[-1]:
                        ema1w_sig = "SAT"
                    else:
                        ema1w_sig = "BEKLE"

                    last_hist = float(m_hist.iloc[-1])
                    prev_hist = float(m_hist.iloc[-2])
                    slope_up = last_hist > prev_hist

                    div_macd, macd_ago = check_bullish_divergence(df_1w["Close"], m_hist, volume=df_1w["Volume"])

                    adx_1w, pdi_1w, mdi_1w = adx_indicator(df_1w["High"], df_1w["Low"], df_1w["Close"])
                    adx_val_1w = adx_1w.iloc[-1]
                    pdi_val_1w = pdi_1w.iloc[-1]
                    mdi_val_1w = mdi_1w.iloc[-1]

                    if adx_val_1w >= 25 and pdi_val_1w > mdi_val_1w:
                        adx_sig_1w = "AL (Güçlü Trend)"
                    elif adx_val_1w >= 25 and mdi_val_1w > pdi_val_1w:
                        adx_sig_1w = "SAT (Güçlü Trend)"
                    else:
                        adx_sig_1w = "BEKLE (Zayıf Trend)"

                    c1w_1, c1w_2, c1w_3 = st.columns(3)
                    c1w_1.metric("MACD Histogram Eğimi", "YUKARI (AL Sinyali)" if slope_up else "AŞAĞI (SAT Sinyali)", f"{last_hist - prev_hist:.2f}")
                    c1w_2.metric("Haftalık EMA (13-26)", ema1w_sig, f"EMA13: {ema_1w_13.iloc[-1]:.2f} | EMA26: {ema_1w_26.iloc[-1]:.2f}")
                    c1w_3.metric("ADX (14)", adx_sig_1w, f"ADX: {adx_val_1w:.1f} | +DI: {pdi_val_1w:.1f} | -DI: {mdi_val_1w:.1f}")

                    if div_macd:
                        st.success(f"🚀 Sistem Haftalık MACD Histogramında **Pozitif Uyumsuzluk** tespit etti! ({macd_ago} bar önce)")

                    fig1_price = go.Figure()
                    fig1_price.add_trace(go.Candlestick(x=df_1w.index, open=df_1w["Open"], high=df_1w["High"], low=df_1w["Low"], close=df_1w["Close"], name="Fiyat"))
                    fig1_price.add_trace(go.Scatter(x=df_1w.index, y=ema_1w_13, name="EMA 13", line=dict(color='blue')))
                    fig1_price.add_trace(go.Scatter(x=df_1w.index, y=ema_1w_26, name="EMA 26", line=dict(color='red')))
                    fig1_price.update_layout(title="Haftalık Fiyat ve EMA (13 & 26)", height=350, xaxis_rangeslider_visible=False)
                    st.plotly_chart(fig1_price, use_container_width=True)

                    fig1 = go.Figure()
                    colors = ['green' if x > 0 else 'red' for x in m_hist.diff()]
                    fig1.add_trace(go.Bar(x=df_1w.index, y=m_hist, name="MACD Hist", marker_color=colors))
                    fig1.update_layout(title="Haftalık MACD Histogramı", height=250)
                    st.plotly_chart(fig1, use_container_width=True)

                    fig1_adx = go.Figure()
                    fig1_adx.add_trace(go.Scatter(x=df_1w.index, y=adx_1w, name="ADX", line=dict(color='black', width=2.5)))
                    fig1_adx.add_trace(go.Scatter(x=df_1w.index, y=pdi_1w, name="+DI", line=dict(color='green')))
                    fig1_adx.add_trace(go.Scatter(x=df_1w.index, y=mdi_1w, name="-DI", line=dict(color='red')))
                    fig1_adx.add_hline(y=25, line_dash="dash", line_color="gray", annotation_text="Trend Başlangıcı (25)")
                    fig1_adx.add_hline(y=50, line_dash="dot", line_color="purple", annotation_text="Aşırı Güçlü Trend (50)")
                    fig1_adx.add_hrect(y0=25, y1=100, fillcolor="rgba(0, 255, 0, 0.05)", layer="below", line_width=0)
                    fig1_adx.add_hrect(y0=0, y1=25, fillcolor="rgba(255, 0, 0, 0.05)", layer="below", line_width=0)
                    fig1_adx.update_layout(title="Haftalık ADX ve Yön Göstergeleri (+DI / -DI)", height=250)
                    st.plotly_chart(fig1_adx, use_container_width=True)

                with t_screen2:
                    st.subheader("2. Ekran: Günlük (Osilatörler ve Sapmalar)")
                    ema_1d_11 = ema(df_1d["Close"], 11)
                    ema_1d_22 = ema(df_1d["Close"], 22)

                    last_close_1d = df_1d["Close"].iloc[-1]
                    if ema_1d_11.iloc[-1] > ema_1d_22.iloc[-1] and last_close_1d > ema_1d_11.iloc[-1]:
                        ema1d_sig = "AL"
                    elif ema_1d_11.iloc[-1] < ema_1d_22.iloc[-1] and last_close_1d < ema_1d_11.iloc[-1]:
                        ema1d_sig = "SAT"
                    else:
                        ema1d_sig = "BEKLE"

                    st.metric("Günlük EMA (11-22)", ema1d_sig, f"EMA11: {ema_1d_11.iloc[-1]:.2f} | EMA22: {ema_1d_22.iloc[-1]:.2f}")

                    fi = force_index(df_1d["Close"], df_1d["Volume"])
                    fi_ema13 = ema(fi, 13)
                    fi_ema2 = ema(fi, 2)

                    rsi13 = rsi(df_1d["Close"], 13)
                    stoch_k, _ = stochastic(df_1d["High"], df_1d["Low"], df_1d["Close"], k_period=5, d_period=3)
                    er_ema, bull_p, bear_p = elder_ray(df_1d["High"], df_1d["Low"], df_1d["Close"], 13)

                    fi_al = (fi.iloc[-1] > fi_ema13.iloc[-1]) and (fi_ema2.iloc[-1] < 0)
                    rsi_al = (rsi13.iloc[-1] < 30)
                    stoch_al = (stoch_k.iloc[-1] < 20)

                    er_ema_up = (er_ema.iloc[-1] > er_ema.iloc[-2])
                    bp_neg_but_rising = (bear_p.iloc[-1] < 0) and (bear_p.iloc[-1] > bear_p.iloc[-2])
                    er_al = er_ema_up and bp_neg_but_rising

                    div_rsi, rsi_ago = check_bullish_divergence(df_1d["Close"], rsi13, volume=df_1d["Volume"])
                    div_stoch, stoch_ago = check_bullish_divergence(df_1d["Close"], stoch_k, volume=df_1d["Volume"])
                    div_er, er_ago = check_bullish_divergence(df_1d["Close"], bear_p, volume=df_1d["Volume"])
                    div_er_bear, er_bear_ago = check_bearish_divergence(df_1d["Close"], bull_p, volume=df_1d["Volume"])

                    adx_1d, pdi_1d, mdi_1d = adx_indicator(df_1d["High"], df_1d["Low"], df_1d["Close"])
                    adx_val_1d = adx_1d.iloc[-1]
                    pdi_val_1d = pdi_1d.iloc[-1]
                    mdi_val_1d = mdi_1d.iloc[-1]

                    if adx_val_1d >= 25 and pdi_val_1d > mdi_val_1d:
                        adx_sig_1d = "AL (Güçlü Trend)"
                    elif adx_val_1d >= 25 and mdi_val_1d > pdi_val_1d:
                        adx_sig_1d = "SAT (Güçlü Trend)"
                    else:
                        adx_sig_1d = "BEKLE (Zayıf Trend)"

                    c1, c2, c3, c4, c5 = st.columns(5)
                    c1.metric("Kuvvet Endeksi (FI)", "AL" if fi_al else "BEKLE", "2 EMA Negatif & Yukarı Dönüş" if fi_al else "")
                    c2.metric("RSI (13)", "AL" if rsi_al else "BEKLE", f"{rsi13.iloc[-1]:.1f}")
                    c3.metric("Stokastik (5)", "AL" if stoch_al else "BEKLE", f"{stoch_k.iloc[-1]:.1f}")
                    c4.metric("Elder-Ray", "AL" if er_al else "BEKLE")
                    c5.metric("ADX (14)", adx_sig_1d, f"ADX: {adx_val_1d:.1f} | +DI: {pdi_val_1d:.1f}")

                    if div_rsi:
                        st.success(f"🚀 RSI(13)'te **Pozitif Uyumsuzluk** tespit edildi! ({rsi_ago} bar önce)")
                    if div_stoch:
                        st.success(f"🚀 Stokastik(5)'te **Pozitif Uyumsuzluk** tespit edildi! ({stoch_ago} bar önce)")
                    if div_er:
                        st.success(f"🚀 Elder-Ray Bear Power'da **Pozitif Uyumsuzluk (Boğa Uyumsuzluğu)** tespit edildi! ({er_ago} bar önce)")
                    if div_er_bear:
                        st.warning(f"⚠️ Elder-Ray Bull Power'da **Negatif Uyumsuzluk (Ayı Uyumsuzluğu)** tespit edildi! ({er_bear_ago} bar önce)")

                    fig2_price = go.Figure()
                    fig2_price.add_trace(go.Candlestick(x=df_1d.index, open=df_1d["Open"], high=df_1d["High"], low=df_1d["Low"], close=df_1d["Close"], name="Fiyat"))
                    fig2_price.add_trace(go.Scatter(x=df_1d.index, y=ema_1d_11, name="EMA 11", line=dict(color='blue')))
                    fig2_price.add_trace(go.Scatter(x=df_1d.index, y=ema_1d_22, name="EMA 22", line=dict(color='red')))
                    fig2_price.update_layout(title="Günlük Fiyat ve EMA (11 & 22)", height=350, xaxis_rangeslider_visible=False)
                    st.plotly_chart(fig2_price, use_container_width=True)

                    fig2_fi = go.Figure()
                    fig2_fi.add_trace(go.Scatter(x=df_1d.index, y=fi_ema13, name="FI 13 EMA", line=dict(color='orange')))
                    fig2_fi.add_trace(go.Bar(x=df_1d.index, y=fi_ema2, name="FI 2 EMA", marker_color='gray'))
                    fig2_fi.update_layout(title="Kuvvet Endeksi (Force Index)", height=250)
                    st.plotly_chart(fig2_fi, use_container_width=True)

                    fig2_er = go.Figure()
                    fig2_er.add_trace(go.Bar(x=df_1d.index, y=bull_p, name="Bull Power", marker_color='green'))
                    fig2_er.add_trace(go.Bar(x=df_1d.index, y=bear_p, name="Bear Power", marker_color='red'))
                    fig2_er.update_layout(title="Elder-Ray (Bull & Bear Power)", height=250)
                    st.plotly_chart(fig2_er, use_container_width=True)

                    fig2_adx = go.Figure()
                    fig2_adx.add_trace(go.Scatter(x=df_1d.index, y=adx_1d, name="ADX", line=dict(color='black', width=2.5)))
                    fig2_adx.add_trace(go.Scatter(x=df_1d.index, y=pdi_1d, name="+DI", line=dict(color='green')))
                    fig2_adx.add_trace(go.Scatter(x=df_1d.index, y=mdi_1d, name="-DI", line=dict(color='red')))
                    fig2_adx.add_hline(y=25, line_dash="dash", line_color="gray", annotation_text="Trend Başlangıcı (25)")
                    fig2_adx.add_hline(y=50, line_dash="dot", line_color="purple", annotation_text="Aşırı Güçlü Trend (50)")
                    fig2_adx.add_hrect(y0=25, y1=100, fillcolor="rgba(0, 255, 0, 0.05)", layer="below", line_width=0)
                    fig2_adx.add_hrect(y0=0, y1=25, fillcolor="rgba(255, 0, 0, 0.05)", layer="below", line_width=0)
                    fig2_adx.update_layout(title="Günlük ADX ve Yön Göstergeleri (+DI / -DI)", height=250)
                    st.plotly_chart(fig2_adx, use_container_width=True)

                with t_screen3:
                    st.subheader("3. Ekran: 1 Saatlik (Giriş / Çıkış ve Hedefler)")
                    adx_1h, pdi_1h, mdi_1h = adx_indicator(df_1h["High"], df_1h["Low"], df_1h["Close"])
                    adx_val_1h = adx_1h.iloc[-1]
                    pdi_val_1h = pdi_1h.iloc[-1]
                    mdi_val_1h = mdi_1h.iloc[-1]

                    if adx_val_1h >= 25 and pdi_val_1h > mdi_val_1h:
                        adx_sig_1h = "AL (Güçlü Trend)"
                    elif adx_val_1h >= 25 and mdi_val_1h > pdi_val_1h:
                        adx_sig_1h = "SAT (Güçlü Trend)"
                    else:
                        adx_sig_1h = "BEKLE (Zayıf Trend)"

                    st.metric("1 Saatlik ADX (14)", adx_sig_1h, f"ADX: {adx_val_1h:.1f} | +DI: {pdi_val_1h:.1f} | -DI: {mdi_val_1h:.1f}")

                    ema_1h = ema(df_1h["Close"], 13)
                    atr_1h = atr(df_1h["High"], df_1h["Low"], df_1h["Close"], 14)
                    last_atr_1h = float(atr_1h.iloc[-1]) if not pd.isna(atr_1h.iloc[-1]) else 0.0

                    pens = ema_1h - df_1h["Low"]
                    pens_positive = pens[pens > 0]
                    avg_pen = float(pens_positive.mean()) if not pens_positive.empty else 0.0

                    up_pens = df_1h["High"] - ema_1h
                    up_pens_positive = up_pens[up_pens > 0]
                    avg_up_pen = float(up_pens_positive.mean()) if not up_pens_positive.empty else 0.0

                    ema_today = float(ema_1h.iloc[-1])
                    ema_yest = float(ema_1h.iloc[-2])
                    ema_delta = ema_today - ema_yest
                    ema_est_tmrw = ema_today + ema_delta

                    buy_level = ema_est_tmrw - avg_pen
                    stop_loss = buy_level - (1.5 * last_atr_1h) if last_atr_1h > 0 else buy_level * 0.98
                    risk = buy_level - stop_loss

                    target_1 = ema_est_tmrw + avg_up_pen
                    target_2 = buy_level + (risk * 2)

                    st.markdown(f"""
                    **Hesaplamalar ve Strateji (Buy Limit & Hedefler):**
                    * 📌 **Güncel EMA (13):** {ema_today:.2f} | **Yarınki Tahmini EMA:** {ema_est_tmrw:.2f}
                    * 🟢 **Önerilen Alış Seviyesi (Buy Limit): {buy_level:.2f}** *(Ortalama {avg_pen:.2f} düşüş penetrasyonu ile)*
                    * 🔴 **Zarar Kes (Stop-Loss): {stop_loss:.2f}** *(Alışın 1.5 ATR altı. Risk: {risk:.2f})*
                    * 🎯 **Hedef 1 (Kısa Vade): {target_1:.2f}** *(Simetrik Yükseliş Penetrasyonu)*
                    * 🚀 **Hedef 2 (Trend - 1:2 RR): {target_2:.2f}** *(Riske edilen tutarın 2 katı kazanç)*
                    """)

                    fig3 = go.Figure()
                    fig3.add_trace(go.Candlestick(x=df_1h.index, open=df_1h["Open"], high=df_1h["High"], low=df_1h["Low"], close=df_1h["Close"], name="Price"))
                    fig3.add_trace(go.Scatter(x=df_1h.index, y=ema_1h, name="EMA 13", line=dict(color='blue')))

                    last_time = df_1h.index[-1]
                    next_time = last_time + pd.Timedelta(hours=1)
                    fig3.add_trace(go.Scatter(x=[next_time], y=[ema_est_tmrw], mode='markers', marker=dict(size=10, color='orange'), name="Tahmini EMA"))

                    fig3.add_hline(y=target_2, line_dash="dash", line_color="darkgreen", annotation_text="Hedef 2 (1:2 RR)", annotation_position="top left")
                    fig3.add_hline(y=target_1, line_dash="dashdot", line_color="cyan", annotation_text="Hedef 1 (Simetrik)", annotation_position="top left")
                    fig3.add_hline(y=buy_level, line_dash="dash", line_color="lime", annotation_text="Limit Alış Seviyesi", annotation_position="bottom left")
                    fig3.add_hline(y=stop_loss, line_dash="dot", line_color="red", annotation_text="Stop-Loss (1.5 ATR)", annotation_position="bottom left")

                    fig3.update_layout(title="1 Saatlik Giriş/Çıkış Stratejisi (Alış, Hedef ve Stop)", height=450, xaxis_rangeslider_visible=False)
                    st.plotly_chart(fig3, use_container_width=True)

                    fig3_adx = go.Figure()
                    fig3_adx.add_trace(go.Scatter(x=df_1h.index, y=adx_1h, name="ADX", line=dict(color='black', width=2.5)))
                    fig3_adx.add_trace(go.Scatter(x=df_1h.index, y=pdi_1h, name="+DI", line=dict(color='green')))
                    fig3_adx.add_trace(go.Scatter(x=df_1h.index, y=mdi_1h, name="-DI", line=dict(color='red')))
                    fig3_adx.add_hline(y=25, line_dash="dash", line_color="gray", annotation_text="Trend Başlangıcı (25)")
                    fig3_adx.add_hline(y=50, line_dash="dot", line_color="purple", annotation_text="Aşırı Güçlü Trend (50)")
                    fig3_adx.add_hrect(y0=25, y1=100, fillcolor="rgba(0, 255, 0, 0.05)", layer="below", line_width=0)
                    fig3_adx.add_hrect(y0=0, y1=25, fillcolor="rgba(255, 0, 0, 0.05)", layer="below", line_width=0)
                    fig3_adx.update_layout(title="1 Saatlik ADX ve Yön Göstergeleri (+DI / -DI)", height=250)
                    st.plotly_chart(fig3_adx, use_container_width=True)


tab_dash, tab_triple, tab_indicator_stats, tab_future, tab_chart_patterns, tab_trend_donchian, tab_financials, tab_history_range, tab_education, tab_index_center, tab_calendar, tab_social, tab_heatmap, tab_export, tab_scan = st.tabs(["📊 Dashboard", "📺 3 Ekranlı Sistem", "📈 İndikatör İstatistik", "🔮 Future Price", "📐 Grafik Formasyonları", "📡 Trend + Donchian", "📘 Bilanço Analizi", "🕰️ Tarih Aralığı Analizi", "📚 Eğitim Merkezi", "📉 BIST Endeks Merkezi", "🗓️ Ekonomik Takvim", "📣 X + YouTube Trends", "🔥 Sektörel Heatmap", "📄 Rapor (PDF/HTML)", "🔍 Tarama"])

with tab_dash:
    render_page_education_expander([("dashboard_page","Dashboard Nasıl Okunur?"),("ema","EMA"),("bollinger","Bollinger Bantları"),("rsi","RSI"),("macd","MACD"),("atr_pct","ATR%"),("stoch_rsi","Stochastic RSI"),("bb_width","Bollinger Genişliği"),("volume_ratio","Hacim Oranı"),("obv","OBV"),("support_resistance","Destek / Direnç"),("target_band","Hedef Fiyat Bandı"),("risk_reward","Risk / Ödül"),("backtest","Backtest"),("monte_carlo","Monte Carlo"),("sharpe","Sharpe"),("sortino","Sortino"),("calmar","Calmar"),("ulcer","Ulcer Index"),("kelly","Kelly"),("vpvr","VPVR"),("poc","POC"),("poc_distance","POC Uzaklık %"),("sector_relative","Sektöre Göre Pahalı / Ucuz")])

    if "app_errors" in st.session_state and st.session_state.app_errors:
        for err in st.session_state.app_errors:
            st.error(f"⚠️ {err}")
        st.session_state.app_errors = []

    if interval == "1d" and not force_latest_candle and not use_live_last_override and not df.empty:
        last_date = df.index[-1].date()
        today_date = datetime.date.today()
        if today_date > last_date and today_date.weekday() < 5:
            st.warning(
                f"⚠️ **Gecikmeli Veri Uyarısı:** Grafikteki son mum dünün ({last_date.strftime('%d.%m.%Y')}) tarihine ait. "
                "Yahoo Finance bugünün günlük mumunu henüz kapatmamış/güncellememiş görünüyor. "
                "Sol menüden **'Eksik Güncel Mumu Zorla Ekle'** seçeneğini işaretleyerek bugünün güncel fiyatını grafiğe dahil edebilirsiniz."
            )

    if use_fa and not st.session_state.screener_df.empty:
        st.subheader(f"🧾 Fundamental Screener Sonuçları ({market})")
        sdf = st.session_state.screener_df.copy()
        show_cols = [
            "ticker", "longName", "FA_pass", "FA_score", "FA_ok_count", "FA_coverage",
            "sector", "industry", "trailingPE", "forwardPE", "pegRatio",
            "priceToSalesTrailing12Months", "priceToBook", "returnOnEquity",
            "operatingMargins", "profitMargins", "debtToEquity",
            "revenueGrowth", "earningsGrowth", "marketCap",
        ]
        sdf_show = sdf[[c for c in show_cols if c in sdf.columns]].copy()
        st.dataframe(sdf_show, use_container_width=True, height=360)

    st.subheader("📊 Aşırı Alım / Spekülasyon Göstergeleri")
    col_ob1, col_ob2, col_ob3, col_ob4, col_ob5, col_ob6 = st.columns(6)
    
    col_ob1.metric("Aşırı Alım Skoru", f"{overbought_result['overbought_score']}/100")
    col_ob2.metric("Aşırı Satım Skoru", f"{overbought_result['oversold_score']}/100")
    col_ob3.metric("Spekülasyon Skoru", f"{overbought_result['speculation_score']}/100")
    col_ob4.metric("Genel Karar", overbought_result["verdict"])

    sf_val = overbought_result.get("short_percent_float")
    sr_val = overbought_result.get("short_ratio")
    sf_str = f"{sf_val * 100:.2f}%" if pd.notna(sf_val) else "N/A"
    sr_str = f"{sr_val:.2f}" if pd.notna(sr_val) else "N/A"
    
    col_ob5.metric("Kısa Poz. % (Short Float)", sf_str)
    col_ob6.metric("Kapatma Gün (Days to Cover)", sr_str)

    with st.expander("Detaylı Aşırı Alım/Spekülasyon Analizi"):
        for _, v in overbought_result["details"].items():
            st.write(f"• {v}")

    c1, c2, c3, c4, c5, c6, c7, c8 = st.columns(8)
    c1.metric("Market", market)
    c2.metric("Sembol", ticker)
    c3.metric("Daily Close", f"{(live_price if use_live_last_override and np.isfinite(live_price) else latest['Close']):.2f}")
    c4.metric("Live/Last", f"{live_price:.2f}" if np.isfinite(live_price) else "N/A")
    c5.metric("Skor", f"{latest['SCORE']:.0f}/100")
    c6.metric("Sinyal", rec)
    c7.metric("Piyasa Filtresi", "BULL ✅" if checkpoints.get("Market Filter OK", True) else "BEAR ❌")
    c8.metric("Haftalık Trend", "BULL ✅" if checkpoints.get("Higher TF Filter OK", True) else "BEAR ❌")

    sector_rel = get_sector_relative_value_summary(ticker, market, universe)
    sma_fast_last = latest.get("SMA_FAST", np.nan)
    sma_slow_last = latest.get("SMA_SLOW", np.nan)
    sma_rel = "Üstünde" if pd.notna(sma_fast_last) and pd.notna(sma_slow_last) and sma_fast_last > sma_slow_last else ("Altında" if pd.notna(sma_fast_last) and pd.notna(sma_slow_last) else "N/A")
    sma_bias = "LONG ✅" if sma_rel == "Üstünde" else ("SHORT ⚠️" if sma_rel == "Altında" else "N/A")

    extra_m1, extra_m2 = st.columns(2)
    extra_m1.metric("Sektöre Göre Değer", sector_rel["label"], delta=(f"{sector_rel['delta_pct']:.1f}%" if pd.notna(sector_rel["delta_pct"]) else None))
    extra_m1.caption(sector_rel["detail"])
    extra_m2.metric("SMA Hızlı/Yavaş", sma_rel, delta=sma_bias)
    
    st.subheader("🕯️ Fiyat Aksiyonu (Price Action) Mum Formasyonları - Son Bar")
    
    is_bull_tail = latest.get("KANGAROO_BULL", 0) == 1
    is_bear_tail = latest.get("KANGAROO_BEAR", 0) == 1
    tail_val = "BOĞA 🦘" if is_bull_tail else ("AYI 🦘" if is_bear_tail else "YOK")
    tail_delta = "AL Yönlü" if is_bull_tail else ("-SAT Yönlü" if is_bear_tail else None)
    
    pa_c1, pa_c2, pa_c3, pa_c4, pa_c5, pa_c6 = st.columns(6)
    pa_c1.metric("1. Kanguru", tail_val, delta=tail_delta)
    pa_c2.metric("2. Engulfing", "Boğa 🟢" if latest.get("PATTERN_ENGULFING_BULL") else ("Ayı 🔴" if latest.get("PATTERN_ENGULFING_BEAR") else "Yok"))
    pa_c3.metric("3. Hammer / Star", "Çekiç 🟢" if latest.get("PATTERN_HAMMER") else ("Kayan Yıldız 🔴" if latest.get("PATTERN_SHOOTING_STAR") else "Yok"))
    pa_c4.metric("4. Doji", "Uzun Bacak ⚪" if latest.get("PATTERN_LL_DOJI") else ("Doji ⚪" if latest.get("PATTERN_DOJI") else "Yok"))
    pa_c5.metric("5. Marubozu", "Boğa 🟢" if latest.get("PATTERN_MARUBOZU_BULL") else ("Ayı 🔴" if latest.get("PATTERN_MARUBOZU_BEAR") else "Yok"))
    pa_c6.metric("6. Harami", "Boğa 🟢" if latest.get("PATTERN_HARAMI_BULL") else ("Ayı 🔴" if latest.get("PATTERN_HARAMI_BEAR") else "Yok"))
    
    pa2_c1, pa2_c2, pa2_c3, pa2_c4, pa2_c5, pa2_c6 = st.columns(6)
    pa2_c1.metric("7. Tweezer", "Dip 🟢" if latest.get("PATTERN_TWEEZER_BOTTOM") else ("Tepe 🔴" if latest.get("PATTERN_TWEEZER_TOP") else "Yok"))
    pa2_c2.metric("8. M./E. Star", "Sabah 🟢" if latest.get("PATTERN_MORNING_STAR") else ("Akşam 🔴" if latest.get("PATTERN_EVENING_STAR") else "Yok"))
    pa2_c3.metric("9. Piercing / Dark", "Delen 🟢" if latest.get("PATTERN_PIERCING") else ("Kara Bulut 🔴" if latest.get("PATTERN_DARK_CLOUD") else "Yok"))
    pa2_c4.metric("10. Inv. H / Hang", "Ters Çekiç 🟢" if latest.get("PATTERN_INV_HAMMER") else ("Asılı Adam 🔴" if latest.get("PATTERN_HANGING_MAN") else "Yok"))
    pa2_c5.metric("11. Filtre Durumu", "Aktif ✅", help="Formasyonlar (2-3 günlük trendler ve EMA) gürültüyü azaltmak için filtrelendi.")
    pa2_c6.write("")

    st.subheader("✅ Kontrol Noktaları (Son Bar)")
    cp_cols = st.columns(3)
    cp_items = list(checkpoints.items())
    for i, (k, v) in enumerate(cp_items):
        with cp_cols[i % 3]:
            st.metric(k, "✅" if v else "❌")

    st.subheader("🎯 Hedef Fiyat Bandı (Senaryo)")
    base_px = float(tp["base"])
    rr_str = fmt_rr(rr_info.get("rr"))
    r1 = None
    s1 = None

    bcol1, bcol2, bcol3 = st.columns(3)
    bcol1.metric("Base", f"{base_px:.2f}", help="Referans alınan anlık/kapanış fiyat.")

    if tp.get("bull"):
        bull1, bull2, r1 = tp["bull"]
        bcol2.metric("Bull Band", f"{bull1:.2f} → {bull2:.2f}", help="ATR bazlı dinamik hedef.")
        if r1 is not None and np.isfinite(r1):
            r1_info = tp.get("r1_dict") or {}
            if r1_info.get("is_synthetic", False):
                bcol2.caption(f"Yakın direnç: {r1:.2f} ({pct_dist(r1, base_px):+.2f}%)\n\n**Hisse Zirvede (Geçmiş Direnç Yok).**\n*Sentetik Pivot Direnci hesaplandı.*")
            else:
                dur = r1_info.get("duration_bars", 0)
                vol_pct = r1_info.get("vol_diff_pct", 0)
                str_pct = r1_info.get("strength_pct", 0)
                bcol2.caption(f"Yakın direnç: {r1:.2f} ({pct_dist(r1, base_px):+.2f}%)\n\n**Güç:** %{str_pct:.0f} | **Uzunluk:** {dur} Bar | **Hacim:** %{vol_pct:+.1f} (Ort.)")
        else:
            bcol2.caption("Yakın direnç: YOK")
    else:
        bcol2.metric("Bull Band", "N/A")

    if tp.get("bear"):
        bear1, bear2, s1 = tp["bear"]
        target_info = f" | Hedef: {rr_info.get('target_type','')}" if rr_info.get('target_type') else ""
        bcol3.metric("Bear Band", f"{bear1:.2f} → {bear2:.2f}  |  RR {rr_str}{target_info}", help="ATR bazlı stop ve Risk/Ödül oranı.")
        if s1 is not None and np.isfinite(s1):
            s1_info = tp.get("s1_dict") or {}
            if s1_info.get("is_synthetic", False):
                bcol3.caption(f"Yakın destek: {s1:.2f} ({pct_dist(s1, base_px):+.2f}%)\n\n**Hisse Diplerde (Geçmiş Destek Yok).**\n*Sentetik Pivot Desteği hesaplandı.*")
            else:
                dur = s1_info.get("duration_bars", 0)
                vol_pct = s1_info.get("vol_diff_pct", 0)
                str_pct = s1_info.get("strength_pct", 0)
                bcol3.caption(f"Yakın destek: {s1:.2f} ({pct_dist(s1, base_px):+.2f}%)\n\n**Güç:** %{str_pct:.0f} | **Uzunluk:** {dur} Bar | **Hacim:** %{vol_pct:+.1f} (Ort.)")
        else:
            bcol3.caption("Yakın destek: YOK")
    else:
        bcol3.metric("Bear Band", f"N/A  |  RR {rr_str}")

    def render_levels_marked(levels: List[dict], base: float, s1, r1):
        lines = []
        for lv_dict in (levels or []):
            lv = float(lv_dict["price"])
            dur = lv_dict["duration_bars"]
            vol_pct = lv_dict["vol_diff_pct"]
            str_pct = lv_dict["strength_pct"]

            tag = ""
            if s1 is not None and np.isfinite(s1) and abs(lv - float(s1)) < 1e-9:
                tag = " 🟩 Yakın Destek"
            if r1 is not None and np.isfinite(r1) and abs(lv - float(r1)) < 1e-9:
                tag = " 🟥 Yakın Direnç"

            dist = pct_dist(lv, base)
            dist_txt = f"{dist:+.2f}%" if dist is not None else ""
            lines.append(f"- **{lv:.2f}** ({dist_txt}) | Güç: %{str_pct:.0f} | Uzunluk: {dur} Bar | Hacim: %{vol_pct:+.1f} {tag}")
        return "\n".join(lines) if lines else "_Seviye yok_"

    with st.expander("Seviye listesi (yaklaşık) — işaretli + fiyata uzaklık %", expanded=False):
        st.markdown(render_levels_marked(tp.get("levels", []), base_px, s1, r1))

    st.subheader("📊 Fiyat + EMA + Bollinger + Sinyaller")
    st.plotly_chart(fig_price, use_container_width=True)

    st.subheader("🛠️ Grafik Analiz Araçları")
    tools_col1, tools_col2, tools_col3, tools_col4 = st.columns(4)
    
    if tools_col1.button("Sadece Grafiği Analiz Et", use_container_width=True, help="Grafikteki tüm formasyon işaretlerini (Kanguru vb.) kaldırır."):
        st.session_state.show_chart_patterns = False
        st.rerun()
        
    if tools_col2.button("Formasyonları Geri Getir", use_container_width=True, help="Kaldırılan tüm formasyon işaretlerini grafiğe geri ekler."):
        st.session_state.show_chart_patterns = True
        st.rerun()
        
    if tools_col3.button("13 EMA Kanalını Aç / Kapat", use_container_width=True, help="Grafiğe 13 EMA High, Low ve Close kanallarını ekler veya kaldırır."):
        st.session_state.show_ema13_channel = not st.session_state.show_ema13_channel
        st.rerun()

    if tools_col4.button("Destek / Direnç Trendini Aç / Kapat", use_container_width=True, help="Grafiğe otomatik destek, direnç ve trend çizgilerini ekler veya kaldırır."):
        st.session_state.show_sr_lines_dashboard = not st.session_state.show_sr_lines_dashboard
        st.rerun()

    st.subheader("📦 Hacim Profili (VPVR / POC)")
    vp1, vp2, vp3 = st.columns(3)
    vp1.metric("POC", f"{vp_poc_price:.2f}" if np.isfinite(vp_poc_price) else "N/A")
    vp2.metric("POC Uzaklık %", f"{pct_dist(vp_poc_price, float(latest["Close"])):+.2f}%" if np.isfinite(vp_poc_price) else "N/A")
    vp3.metric("Profil Bar Sayısı", f"{min(len(df), 220)}")
    st.plotly_chart(fig_vpvr, use_container_width=True)

    st.subheader("📉 RSI / MACD / ATR%")
    colA, colB, colC = st.columns(3)
    with colA: st.plotly_chart(fig_rsi, use_container_width=True)
    with colB: st.plotly_chart(fig_macd, use_container_width=True)
    with colC: st.plotly_chart(fig_atr, use_container_width=True)

    st.subheader("📊 Stochastic RSI / Bollinger Genişliği / Hacim Oranı")
    colD, colE, colF = st.columns(3)
    with colD: st.plotly_chart(fig_stoch, use_container_width=True)
    with colE: st.plotly_chart(fig_bbwidth, use_container_width=True)
    with colF: st.plotly_chart(fig_volratio, use_container_width=True)

    st.subheader("📊 Hacim ve Trend Karşılaştırmaları")
    colV1, colV2, colV3 = st.columns(3)
    with colV1: st.plotly_chart(fig_vol_market, use_container_width=True)
    with colV2: st.plotly_chart(fig_vol_2wk, use_container_width=True)
    with colV3: st.plotly_chart(fig_obv, use_container_width=True)

    st.subheader("🧪 Backtest Özeti (Long-only + Scale Out + Time Stop)")
    m1, m2, m3, m4, m5, m6, m7, m8 = st.columns(8)
    m1.metric("Total Return", f"{metrics['Total Return']*100:.1f}%")
    m2.metric("Ann Return", f"{metrics['Annualized Return']*100:.1f}%")
    m3.metric("Sharpe", f"{metrics['Sharpe']:.2f}")
    m4.metric("Max DD", f"{metrics['Max Drawdown']*100:.1f}%")
    m5.metric("Trades", f"{metrics['Trades']}")
    m6.metric("Win Rate", f"{metrics['Win Rate']*100:.1f}%")
    m7.metric("Beta", f"{metrics['Beta']:.2f}")
    m8.metric("Info Ratio", f"{metrics['Information Ratio']:.2f}")

    with st.expander("Trade listesi (Detaylı Kâr/Zarar ve Çıkış Nedenleri)", expanded=False):
        st.dataframe(tdf, use_container_width=True, height=240)

    with st.expander("Equity curve (Sermaye Eğrisi)", expanded=False):
        st.plotly_chart(fig_eq, use_container_width=True)

    st.subheader("🎲 Monte Carlo Simülasyonu")
    mc_res = backtest_monte_carlo(eq, n_sims=300)
    if mc_res.get("error"):
        st.info(mc_res["error"])
    else:
        mc1, mc2, mc3, mc4 = st.columns(4)
        mc1.metric("Median Sonuç", f"{mc_res['median_return_pct']:.2f}%")
        mc2.metric("P10", f"{mc_res['p10_return_pct']:.2f}%")
        mc3.metric("P90", f"{mc_res['p90_return_pct']:.2f}%")
        mc4.metric("Median Max DD", f"{mc_res['median_max_dd_pct']:.2f}%")
        st.plotly_chart(mc_res["figure"], use_container_width=True)

    if sentiment_summary:
        st.subheader("📰 Haber Duygu Analizi (Google News + Gemini)")
        st.info(sentiment_summary)
        if st.session_state.sentiment_items:
            st.markdown("**Kaynak Haberler:**")
            for item in st.session_state.sentiment_items:
                st.markdown(f"- [{item['title']}]({item['link']})")

    st.subheader("🤖 Gemini Multimodal AI — Grafik + Price Action + Spekülasyon Analizi")
    if not ai_on:
        st.info("Gemini kapalı (sol menüden açabilirsiniz).")
    else:
        pa = price_action_pack(df, last_n=20)
        st.session_state.pa_pack = pa

        user_msg = st.text_area(
            "Gemini'ye sor/talimat ver (spekülasyon sorusu eklendi):",
            value="Ekteki fiyat grafiği resmini ve aşağıdaki JSON'da bulunan son 20 barlık price-action verilerini incele. Ayrıca aşırı alım/spekülasyon göstergelerini de değerlendir (RSI, Bollinger, hacim sıçraması, fiyatın EMA'dan uzaklığı, Stochastic RSI). Bu hisse aşırı değerli mi, spekülatif bir hareket mi var? AL/SAT/İZLE önerisi ve stratejinin bozulacağı şartları yaz. Analizin sonunda aşağıdaki formatta bir tablo ekle:\n\n| Hedef | Fiyat |\n|-------|-------|\n| Alış Fiyatı (önerilen giriş) | ... |\n| Hedef Satış Fiyatı (ilk hedef) | ... |\n| Stop Loss (ATR bazlı) | ... |\n\n",
            height=150,
        )

        col_g1, col_g2 = st.columns([1, 1])

        with col_g1:
            if st.button("🖼️ Gemini'ye Sor (Görsel + Tüm Veriler)", use_container_width=True):
                snap20 = df_snapshot_for_llm(df, n=25)
                fa_row_local = None
                if use_fa and not st.session_state.screener_df.empty:
                    screener_row = find_screener_row(st.session_state.screener_df, ticker)
                    f_single = fetch_fundamentals_generic(ticker, market=market)
                    f_score, f_breakdown, f_pass = fundamental_score_row(f_single, fa_mode, thresholds)
                    fa_eval = {
                        "mode": fa_mode, "score": f_score, "passed": f_pass,
                        "ok_cnt": sum(1 for v in f_breakdown.values() if v.get("available") and v.get("ok")),
                        "coverage": sum(1 for v in f_breakdown.values() if v.get("available")),
                    }
                    fa_row_local = merge_fa_row(screener_row, f_single, fa_eval)

                sector_comp = ""
                if fa_row_local and fa_row_local.get("trailingPE") and fa_row_local.get("sector"):
                    sector_comp = f"Sektör: {fa_row_local['sector']}, F/K: {fa_row_local['trailingPE']:.2f}"

                sentiment_info = st.session_state.get("sentiment_summary", "")

                prompt = f"""
Sen bir price-action, formasyon okuma, aşırı alım/spekülasyon tespiti ve risk yönetimi odaklı kıdemli finansal analiz asistanısın. Kesin yatırım tavsiyesi verme, sadece objektif ve eğitim amaçlı analiz yap. Lütfen aşağıdaki adımları takip ederek analiz yap:
1. Genel trendi değerlendir (EMA50, EMA200, fiyatın bu ortalamalara göre konumu).
2. Temel destek/direnç seviyelerini belirle (price action paketindeki swing high/low, order block).
3. Aşırı alım/spekülasyon göstergelerini incele:
   - RSI (>70 aşırı alım, <30 aşırı satım)
   - Bollinger Bandı (fiyat üst bandın üstünde mi?)
   - Hacim sıçraması (son hacim normalin 1.5 katından fazla mı?)
   - Fiyatın EMA50'den uzaklığı (%20'den fazla mı?)
   - Stokastik RSI (>80 aşırı alım)
   - Fiyat yükselirken hacim düşüyor mu? (zayıflama)
4. Volatiliteyi (ATR), Kanguru Kuyruğu (Kangaroo Tail) formasyonu olup olmadığını ve stop seviyesi için fikir ver.
5. Temel analiz skorunu (FA) değerlendir (eğer varsa).
6. Haber duyarlılığını dikkate al (eğer varsa).
7. Tüm bu bilgileri sentezleyerek:
   - Hisse aşırı değerli mi, aşırı değersiz mi, yoksa normal bölgede mi?
   - Spekülatif bir hareket var mı? (ani hacim, zayıf trend)
   - AL/SAT/İZLE önerisi, hedef bant ve stratejinin bozulacağı şartlar.
Ekte sana analiz edilen hissenin grafiğinin GÖRSELİNİ (image) gönderdim. Görseli detaylıca incele. Ek olarak algoritmamızın ürettiği aşağıdaki JSON verilerini de referans al:

JSON:
{json.dumps({
    "ticker": ticker,
    "market": market,
    "algo_signal": rec,
    "latest_close": float(latest["Close"]),
    "target_band": tp,
    "rr_info": rr_info,
    "pa_pack": pa,
    "data_snapshot": snap20,
    "fundamental_score": fa_row_local.get("FA_score") if fa_row_local else None,
    "fundamental_pass": fa_row_local.get("FA_pass") if fa_row_local else None,
    "sector_info": sector_comp,
    "sentiment_analysis": sentiment_info,
    "overbought_analysis": overbought_result
}, ensure_ascii=False, default=str)}

Kullanıcının Sorusu: {user_msg}

Analizin sonunda aşağıdaki gibi bir tablo ile hedef alış ve satış fiyatlarını göster (fiyatları kendi hesapladığın seviyelerle doldur):

| Hedef | Fiyat |
|-------|-------|
| Alış Fiyatı (önerilen giriş) | ... |
| Hedef Satış Fiyatı (ilk hedef) | ... |
| Stop Loss (ATR bazlı) | ... |
"""

                image_bytes = _plotly_fig_to_png_bytes(fig_price)

                text = gemini_generate_text(
                    prompt=prompt,
                    model=gemini_model,
                    temperature=gemini_temp,
                    max_output_tokens=gemini_max_tokens,
                    image_bytes=image_bytes,
                )
                st.session_state.gemini_text = text

        with col_g2:
            if st.button("Temizle", use_container_width=True):
                st.session_state.gemini_text = ""

        if st.session_state.gemini_text:
            st.markdown(st.session_state.gemini_text)


# =============================
# HEATMAP TAB
# =============================
with tab_heatmap:
    st.header("🔥 Sektörel Treemap (Heatmap)")
    st.write("Belirtilen pazar veya Screener sonuçları üzerinden şirketlerin günlük, haftalık ve aylık performanslarını görselleştirir.")

    if st.button("Heatmap Verilerini Getir ve Oluştur (1D, 1W, 1M)", type="primary"):
        with st.spinner("Toplu veri çekiliyor ve hesaplanıyor..."):
            if not st.session_state.screener_df.empty:
                hm_tickers = st.session_state.screener_df["ticker"].tolist()
            else:
                hm_tickers = universe[:100]

            use_tickers = [normalize_ticker(t, market) for t in hm_tickers]

            try:
                df_all = yf.download(use_tickers, period="1mo", interval="1d", auto_adjust=True, group_by="ticker", progress=False)
            except Exception:
                df_all = pd.DataFrame()

            hm_data = []
            for t in use_tickers:
                try:
                    if len(use_tickers) == 1:
                        df_t = df_all.copy()
                    else:
                        df_t = df_all[t].copy()

                    df_t = df_t.dropna()
                    if len(df_t) >= 2:
                        c_last = float(df_t["Close"].iloc[-1])
                        c_prev_1d = float(df_t["Close"].iloc[-2])
                        c_prev_1wk = float(df_t["Close"].iloc[-6]) if len(df_t) >= 6 else float(df_t["Close"].iloc[0])
                        c_prev_1mo = float(df_t["Close"].iloc[0])

                        ret_1d = (c_last / c_prev_1d - 1) * 100
                        ret_1wk = (c_last / c_prev_1wk - 1) * 100
                        ret_1mo = (c_last / c_prev_1mo - 1) * 100

                        sector = "Genel"
                        if not st.session_state.screener_df.empty:
                            row_match = find_screener_row(st.session_state.screener_df, t)
                            if row_match and pd.notna(row_match.get("sector")) and str(row_match.get("sector")).strip():
                                sector = str(row_match.get("sector"))

                        hm_data.append({"Ticker": t, "Sector": sector, "1 Günlük %": ret_1d, "1 Haftalık %": ret_1wk, "1 Aylık %": ret_1mo})
                except Exception:
                    pass

            df_hm = pd.DataFrame(hm_data)

        if not df_hm.empty:
            df_hm["Abs_1D"] = df_hm["1 Günlük %"].abs()

            st.subheader("GÜNLÜK Performans")
            fig_hm_1d = px.treemap(
                df_hm, path=[px.Constant("Tüm Pazar"), "Sector", "Ticker"], values="Abs_1D",
                color="1 Günlük %", color_continuous_scale="RdYlGn", color_continuous_midpoint=0,
                custom_data=["1 Günlük %", "1 Haftalık %", "1 Aylık %"],
            )
            fig_hm_1d.update_traces(hovertemplate="<b>%{label}</b><br>1 Günlük: %{customdata[0]:.2f}%<br>1 Haftalık: %{customdata[1]:.2f}%<br>1 Aylık: %{customdata[2]:.2f}%")
            st.plotly_chart(fig_hm_1d, use_container_width=True)

            st.subheader("HAFTALIK Performans")
            df_hm["Abs_1W"] = df_hm["1 Haftalık %"].abs()
            fig_hm_1w = px.treemap(df_hm, path=[px.Constant("Tüm Pazar"), "Sector", "Ticker"], values="Abs_1W", color="1 Haftalık %", color_continuous_scale="RdYlGn", color_continuous_midpoint=0)
            st.plotly_chart(fig_hm_1w, use_container_width=True)

            st.subheader("AYLIK Performans")
            df_hm["Abs_1M"] = df_hm["1 Aylık %"].abs()
            fig_hm_1m = px.treemap(df_hm, path=[px.Constant("Tüm Pazar"), "Sector", "Ticker"], values="Abs_1M", color="1 Aylık %", color_continuous_scale="RdYlGn", color_continuous_midpoint=0)
            st.plotly_chart(fig_hm_1m, use_container_width=True)
        else:
            st.error("Heatmap için yeterli veri çekilemedi.")


# =============================
# EXPORT TAB
# =============================
with tab_export:
    st.subheader("📄 Rapor İndir (En sorunsuz: HTML → tarayıcıdan PDF)")
    include_charts = st.checkbox("Rapor grafikleri dahil et", value=True)
    include_trades = st.checkbox("Trade listesi dahil et (ilk 25)", value=True)
    include_gemini = st.checkbox("Gemini çıktısını rapora ekle", value=True)
    include_pa = st.checkbox("Price Action Pack'i rapora ekle", value=True)
    include_sentiment = st.checkbox("Haber duygu analizini rapora ekle", value=True)
    include_overbought = st.checkbox("Aşırı alım/spekülasyon analizini rapora ekle", value=True)

    with st.spinner("Fundamental + screener satırı hazırlanıyor..."):
        try:
            f_single = fetch_fundamentals_generic(ticker, market=market)
            f_score, f_breakdown, f_pass = fundamental_score_row(f_single, fa_mode, thresholds)
            fa_eval = {
                "mode": fa_mode, "score": f_score, "passed": f_pass,
                "ok_cnt": sum(1 for v in f_breakdown.values() if v.get("available") and v.get("ok")),
                "coverage": sum(1 for v in f_breakdown.values() if v.get("available")),
            }
            screener_row = find_screener_row(st.session_state.get("screener_df", pd.DataFrame()), ticker)
            fa_row = merge_fa_row(screener_row, f_single, fa_eval)
        except Exception:
            fa_row = None

    meta = {
        "market": market, "ticker": ticker, "interval": interval, "period": period,
        "currency": analysis_currency,
        "preset": preset_name, "ema_fast": ema_fast, "ema_slow": ema_slow,
        "rsi_period": rsi_period, "bb_period": bb_period, "bb_std": bb_std,
        "atr_period": atr_period, "vol_sma": vol_sma,
    }

    ta_summary = {
        "rec": rec, "close": fmt_num(float(latest["Close"]), 2),
        "live": fmt_num(float(live_price), 2) if np.isfinite(live_price) else "N/A",
        "score": fmt_num(float(latest.get("SCORE", np.nan)), 0),
        "rsi": fmt_num(float(latest.get("RSI", np.nan)), 2),
        "ema50": fmt_num(float(latest.get("EMA50", np.nan)), 2),
        "ema200": fmt_num(float(latest.get("EMA200", np.nan)), 2),
        "atr_pct": fmt_pct(float(latest.get("ATR_PCT", np.nan))) if pd.notna(latest.get("ATR_PCT", np.nan)) else "N/A",
    }

    gemini_text = st.session_state.gemini_text if include_gemini else None
    pa_pack_export = st.session_state.pa_pack if include_pa else None
    sentiment_export = st.session_state.sentiment_summary if include_sentiment else None
    sentiment_items_export = st.session_state.sentiment_items if include_sentiment else None
    overbought_export = overbought_result if include_overbought else None

    html_bytes = build_html_report(
        title=f"FA→TA Trading Report - {ticker}", meta=meta, checkpoints=checkpoints, metrics=metrics,
        tp=tp, rr_info=rr_info, figs=(figs_for_report if include_charts else {}), fa_row=fa_row,
        gemini_insight=gemini_text, pa_pack=pa_pack_export, sentiment_summary=sentiment_export,
        sentiment_items=sentiment_items_export, overbought_result=overbought_export,
    )
    st.download_button("⬇️ HTML İndir (Önerilen)", data=html_bytes, file_name=f"{ticker}_FA_TA_report.html", mime="text/html", use_container_width=True)

    st.divider()

    if not REPORTLAB_OK:
        st.warning("Doğrudan PDF için 'reportlab' gerekli. requirements.txt içine `reportlab` ekleyip redeploy edersen PDF butonu da aktif olur.")
    else:
        if st.button("🧾 PDF Oluştur (reportlab)", use_container_width=True):
            with st.spinner("PDF oluşturuluyor..."):
                pdf_bytes = generate_pdf_report(
                    title=f"FA→TA Trading Report - {ticker}", subtitle="Educational analysis.", meta=meta,
                    checkpoints=checkpoints, ta_summary=ta_summary, target_band=tp, rr_info=rr_info,
                    backtest_metrics=metrics, fa_row=fa_row, levels=tp.get("levels", []),
                    trades_df=(tdf if include_trades else None), figs=(figs_for_report if include_charts else None),
                    include_charts=include_charts, gemini_insight=gemini_text, pa_pack=pa_pack_export,
                    sentiment_summary=sentiment_export, sentiment_items=sentiment_items_export, overbought_result=overbought_export,
                )

            if pdf_bytes:
                st.success("PDF hazır ✅")
                st.download_button("⬇️ PDF İndir", data=pdf_bytes, file_name=f"{ticker}_FA_TA_report.pdf", mime="application/pdf", use_container_width=True)
                st.info("Grafikler gelmiyorsa `kaleido` kütüphanesini requirements'a eklemelisin.")
            else:
                st.error("PDF üretilemedi.")


# =============================
# TRIPLE SCREEN TAB
# =============================

with tab_scan:
    timeframe_options = ["Aylık", "Haftalık", "Günlük", "Saatlik"]

    if not st.session_state.ta_ran:
        st.info("Sol menüden 'Teknik Analizi Çalıştır' butonuna basarak sistemi aktifleştirmelisin.")
    else:
        if not st.session_state.screener_df.empty and "ticker" in st.session_state.screener_df.columns:
            scan_symbol_options_raw = st.session_state.screener_df["ticker"].astype(str).str.upper().tolist()
        else:
            scan_symbol_options_raw = [str(x).upper() for x in universe]

        if ticker:
            scan_symbol_options_raw = [ticker] + scan_symbol_options_raw

        scan_symbol_options = []
        seen_scan_symbols = set()
        for opt in scan_symbol_options_raw:
            naked_opt = naked_ticker(opt)
            if naked_opt not in seen_scan_symbols:
                seen_scan_symbols.add(naked_opt)
                scan_symbol_options.append(opt)

        default_scan_symbols = [ticker] if ticker else (scan_symbol_options[:1] if scan_symbol_options else [])

        st.markdown("---")
        st.subheader("🚀 Dashboard AL / GÜÇLÜ AL Taraması (1W / 2Y)")
        st.caption("Seçtiğin BIST veya USA universe listesini haftalık mum ve 2 yıllık veriyle tarar. Dashboard'daki build_features + signal_with_checkpoints skor mantığını kullanır; GÜÇLÜ AL ve AL veren hisseleri listeler.")

        # Kullanıcının taranacak evreni Tarama sekmesi içinden açıkça seçebilmesi için
        # sidebar market seçimine dokunmadan bağımsız universe seçimi eklendi.
        strong_scan_universe_options = ["BIST100", "USA - S&P 500", "USA - Nasdaq 100", "Mevcut liste", "Sadece seçili sembol"]
        if market == "BIST":
            strong_scan_default_idx = 0
        elif usa_bucket == "Nasdaq 100":
            strong_scan_default_idx = 2
        else:
            strong_scan_default_idx = 1

        sb_col1, sb_col2, sb_col3, sb_col4 = st.columns([1.25, 0.75, 0.75, 0.75])
        with sb_col1:
            strong_scan_source = st.selectbox(
                "Taranacak universe",
                strong_scan_universe_options,
                index=strong_scan_default_idx,
                key="dashboard_strong_scan_source",
                help="Hangi evrenin taranacağını buradan seçebilirsin. BIST100, USA S&P 500, USA Nasdaq 100, mevcut liste veya sadece seçili sembol.",
            )

        if strong_scan_source == "BIST100":
            strong_scan_market = "BIST"
            strong_scan_symbols_base = load_universe_file(pjoin("universes", "bist100.txt"))
        elif strong_scan_source == "USA - S&P 500":
            strong_scan_market = "USA"
            strong_scan_symbols_base = load_universe_file(pjoin("universes", "sp500.txt"))
        elif strong_scan_source == "USA - Nasdaq 100":
            strong_scan_market = "USA"
            strong_scan_symbols_base = load_universe_file(pjoin("universes", "nasdaq100.txt"))
        elif strong_scan_source == "Sadece seçili sembol":
            strong_scan_market = market
            strong_scan_symbols_base = [ticker] if ticker else []
        else:
            strong_scan_market = market
            strong_scan_symbols_base = scan_symbol_options

        strong_scan_currency = "TL"
        with sb_col2:
            if strong_scan_market == "BIST":
                strong_scan_currency = st.selectbox(
                    "BIST para birimi",
                    ["TL", "USD"],
                    index=0 if bist_price_currency == "TL" else 1,
                    key="dashboard_strong_scan_bist_currency",
                    help="BIST taramasında fiyatların TL mi USD bazlı mı hesaplanacağını seçer.",
                )
            else:
                st.caption("Para birimi: USD")

        strong_scan_available_count = max(1, len(strong_scan_symbols_base))
        with sb_col3:
            max_strong_scan_symbols = st.number_input(
                "Maksimum hisse",
                min_value=1,
                max_value=strong_scan_available_count,
                value=min(100, strong_scan_available_count),
                step=10,
                key="dashboard_strong_scan_max_symbols",
                help="Çok büyük evrenlerde tarama uzun sürebilir. İstersen artırabilirsin.",
            )
        with sb_col4:
            strong_scan_workers = st.number_input(
                "Paralel işçi",
                min_value=1,
                max_value=10,
                value=6,
                step=1,
                key="dashboard_strong_scan_workers",
                help="Veri çekimini hızlandırır. API limitlerinde hata olursa düşür.",
            )

        st.caption(f"Seçili tarama: **{strong_scan_source}** | Market: **{strong_scan_market}** | Hisse sayısı: **{len(strong_scan_symbols_base)}**")

        run_dashboard_strong_scan = st.button(
            "🚀 1W / 2Y Dashboard AL + GÜÇLÜ AL Taramasını Çalıştır",
            key="run_dashboard_strong_buy_scan",
            use_container_width=True,
            type="secondary",
        )

        clear_dashboard_strong_scan = st.button(
            "Dashboard AL / GÜÇLÜ AL Sonuçlarını Temizle",
            key="clear_dashboard_strong_buy_scan",
            use_container_width=True,
        )

        if clear_dashboard_strong_scan:
            st.session_state.dashboard_strong_buy_scan_results = pd.DataFrame()

        if run_dashboard_strong_scan:
            strong_symbols_to_scan = strong_scan_symbols_base[:int(max_strong_scan_symbols)]

            if not strong_symbols_to_scan:
                st.warning("Dashboard AL / GÜÇLÜ AL taraması için hisse bulunamadı.")
            else:
                with st.spinner(f"1W / 2Y Dashboard AL / GÜÇLÜ AL taraması yapılıyor... ({len(strong_symbols_to_scan)} hisse)"):
                    strong_scan_df = dashboard_strong_buy_scan_many(
                        strong_symbols_to_scan,
                        strong_scan_market,
                        cfg.copy(),
                        strong_scan_currency,
                        max_workers=int(strong_scan_workers),
                    )

                if strong_scan_df.empty:
                    st.session_state.dashboard_strong_buy_scan_results = pd.DataFrame()
                else:
                    signal_series = strong_scan_df.get("Dashboard Sinyali", pd.Series(dtype=str)).astype(str)
                    signal_norm = signal_series.str.strip().str.upper()
                    strong_ok = strong_scan_df[
                        signal_norm.eq("GÜÇLÜ AL") |
                        signal_norm.eq("GÜÇLÜ AL (TREND)") |
                        signal_norm.eq("AL") |
                        signal_norm.eq("AL (GÜÇLÜ TREND)")
                    ].copy()

                    if not strong_ok.empty:
                        strong_ok["Sinyal Öncelik"] = strong_ok["Dashboard Sinyali"].astype(str).str.strip().str.upper().map(
                            {"GÜÇLÜ AL": 1, "AL (GÜÇLÜ TREND)": 2, "GÜÇLÜ AL (TREND)": 2, "AL": 3}
                        ).fillna(9)
                        strong_ok = strong_ok.sort_values(
                            ["Sinyal Öncelik", "Skor", "Hacim Oranı", "RSI"],
                            ascending=[True, False, False, False],
                            na_position="last",
                        ).reset_index(drop=True)

                    st.session_state.dashboard_strong_buy_scan_results = strong_ok

                    failed_count = int(strong_scan_df.get("Hata", pd.Series(dtype=str)).astype(str).str.len().gt(0).sum()) if "Hata" in strong_scan_df.columns else 0
                    if failed_count > 0:
                        st.caption(f"Not: {failed_count} sembolde veri/API kaynaklı hata veya yetersiz veri oluştu.")

        dashboard_strong_results = st.session_state.dashboard_strong_buy_scan_results.copy()
        if not dashboard_strong_results.empty:
            if "Dashboard Sinyali" in dashboard_strong_results.columns:
                result_signal_norm = dashboard_strong_results["Dashboard Sinyali"].astype(str).str.strip().str.upper()
                strong_count = int(result_signal_norm.isin(["GÜÇLÜ AL", "GÜÇLÜ AL (TREND)"]).sum())
                al_trend_count = int(result_signal_norm.eq("AL (GÜÇLÜ TREND)").sum())
                buy_count = int(result_signal_norm.eq("AL").sum())
            else:
                strong_count = 0
                al_trend_count = 0
                buy_count = 0
            st.success(f"1W / 2Y Dashboard sonucu: {strong_count} GÜÇLÜ AL, {al_trend_count} AL (Güçlü Trend), {buy_count} AL adayı")
            strong_show_cols = [
                "Sembol", "Dashboard Sinyali", "Skor", "Son Kapanış", "RSI", "MACD Hist",
                "ATR%", "Hacim Oranı", "Trend OK", "BB OK", "OBV OK", "Para Birimi",
                "USDTRY", "Son Bar Tarihi"
            ]
            st.dataframe(
                dashboard_strong_results[[c for c in strong_show_cols if c in dashboard_strong_results.columns]],
                use_container_width=True,
                height=300,
            )

            pick_strong_symbol = st.selectbox(
                "Dashboard'a aktarılacak AL / GÜÇLÜ AL hissesi",
                dashboard_strong_results["Sembol"].astype(str).tolist(),
                key="dashboard_strong_buy_pick_symbol",
            )
            if st.button("➡️ Seçili hisseyi Dashboard'a al", key="send_dashboard_strong_buy_to_dashboard", use_container_width=True):
                st.session_state.selected_ticker = pick_strong_symbol
                st.rerun()
        elif run_dashboard_strong_scan:
            st.info("1W / 2Y Dashboard mantığına göre AL veya GÜÇLÜ AL adayı bulunmadı.")

        st.markdown("---")
        st.header("🔍 Uyumsuzluk Tarama Sekmesi")
        st.caption("Seçilen hisselerde aylık, haftalık, günlük ve saatlik bazda MACD Histogram, EMA, ADX, Kuvvet Endeksi (FI), RSI (13), Stokastik (5) ve Elder-Ray uyumsuzluklarını listeler. Aylık, haftalık ve günlük tarama kapanmış barlara göre çalışır.")

        with st.form("divergence_scan_form"):
            st.markdown("**Zaman Dilimleri**")
            tf_select_all = st.checkbox("Tüm zaman dilimlerini seç", value=True)
            tf_cols = st.columns(4)
            tf_checks = {}
            for i, tf_name in enumerate(timeframe_options):
                with tf_cols[i % 4]:
                    tf_checks[tf_name] = st.checkbox(tf_name, value=True, key=f"scan_tf_{tf_name}")

            st.markdown("**Taranacak Hisseler**")
            sym_select_all = st.checkbox("Tüm hisseleri seç", value=False)
            sym_cols = st.columns(4)
            sym_checks = {}
            default_symbol_set = set(default_scan_symbols)

            for i, sym in enumerate(scan_symbol_options):
                with sym_cols[i % 4]:
                    sym_checks[sym] = st.checkbox(
                        sym,
                        value=(sym in default_symbol_set),
                        key=f"scan_sym_{sym}",
                    )

            run_scan_submitted = st.form_submit_button("🔎 Uyumsuzluk Taramasını Çalıştır", use_container_width=True)

        clear_scan_results = st.button("Sonuçları Temizle", key="clear_divergence_scan", use_container_width=True)

        if clear_scan_results:
            st.session_state.divergence_scan_results = pd.DataFrame()

        if run_scan_submitted:
            selected_scan_timeframes = timeframe_options.copy() if tf_select_all else [tf for tf, is_checked in tf_checks.items() if is_checked]
            selected_scan_symbols = scan_symbol_options.copy() if sym_select_all else [sym for sym, is_checked in sym_checks.items() if is_checked]

            if not selected_scan_symbols:
                st.warning("Lütfen en az bir hisse seçin.")
            elif not selected_scan_timeframes:
                st.warning("Lütfen en az bir zaman dilimi seçin.")
            else:
                scan_frames = []
                with st.spinner("Uyumsuzluk taraması yapılıyor..."):
                    for scan_symbol in selected_scan_symbols:
                        scan_symbol_norm = normalize_ticker(scan_symbol, market)
                        for timeframe_name in selected_scan_timeframes:
                            scan_df = scan_divergences_for_symbol(
                                scan_symbol_norm,
                                timeframe_name,
                                force_latest_daily=False,
                            )
                            if not scan_df.empty:
                                scan_frames.append(scan_df)

                if scan_frames:
                    scan_results_df = pd.concat(scan_frames, ignore_index=True)
                    scan_results_df = scan_results_df.sort_values(
                        ["Zaman Dilimi", "Sembol", "Uyumsuzluk", "Gösterge", "Kaç Bar Önce"],
                        ascending=[True, True, True, True, True],
                    ).reset_index(drop=True)
                    st.session_state.divergence_scan_results = scan_results_df
                else:
                    st.session_state.divergence_scan_results = pd.DataFrame(
                        columns=["Sembol", "Zaman Dilimi", "Gösterge", "Uyumsuzluk", "Kaç Bar Önce", "Son Kapanış"]
                    )

        scan_results = st.session_state.divergence_scan_results.copy()

        if scan_results.empty:
            st.info("Henüz sonuç yok. Hisse ve zaman dilimlerini seçip taramayı başlatın.")
        else:
            st.success(f"Toplam {len(scan_results)} uyumsuzluk bulundu.")
            st.dataframe(scan_results, use_container_width=True, height=260)

            display_timeframes = [tf for tf in timeframe_options if tf in scan_results["Zaman Dilimi"].unique().tolist()]
            if display_timeframes:
                tf_tabs = st.tabs(display_timeframes)
                for tf_name, tf_tab in zip(display_timeframes, tf_tabs):
                    with tf_tab:
                        tf_df = scan_results[scan_results["Zaman Dilimi"] == tf_name].copy()

                        pos_df = tf_df[tf_df["Uyumsuzluk"] == "Pozitif"].copy()
                        neg_df = tf_df[tf_df["Uyumsuzluk"] == "Negatif"].copy()

                        scan_res_col1, scan_res_col2 = st.columns(2)

                        with scan_res_col1:
                            st.subheader("Pozitif Uyumsuzluklar")
                            if pos_df.empty:
                                st.info("Pozitif uyumsuzluk bulunmadı.")
                            else:
                                st.dataframe(pos_df, use_container_width=True, height=320)

                        with scan_res_col2:
                            st.subheader("Negatif Uyumsuzluklar")
                            if neg_df.empty:
                                st.info("Negatif uyumsuzluk bulunmadı.")
                            else:
                                st.dataframe(neg_df, use_container_width=True, height=320)



with tab_history_range:
    st.header("🕰️ Tarih Aralığı Analizi")
    render_page_education_expander([("history_page","Tarih Aralığı Analizi Nasıl Okunur?"),("ema","EMA"),("rsi","RSI"),("macd","MACD"),("atr_pct","ATR%"),("stoch_rsi","Stochastic RSI"),("bb_width","Bollinger Genişliği"),("volume_ratio","Hacim Oranı"),("backtest","Backtest"),("vpvr","VPVR"),("poc","POC"),("poc_distance","POC Uzaklık %")])
    st.caption("Yüklü veri penceresi içinden geçmiş bir tarih aralığı seçerek büyük boy grafik ve ilgili teknik panelleri görüntüleyebilirsin. Daha eski tarihleri görmek için sol menüden periyodu genişlet.")

    if df is None or df.empty:
        st.info("Bu sekme için önce ana teknik verinin yüklenmiş olması gerekir.")
    else:
        available_dates = pd.to_datetime(df.index).normalize()
        min_hist_date = available_dates.min().date()
        max_hist_date = available_dates.max().date()

        default_start = max(min_hist_date, max_hist_date - datetime.timedelta(days=60))
        hr1, hr2 = st.columns(2)
        with hr1:
            history_start_date = st.date_input(
                "Başlangıç Tarihi",
                value=default_start,
                min_value=min_hist_date,
                max_value=max_hist_date,
                key="history_range_start_date",
            )
        with hr2:
            history_end_date = st.date_input(
                "Bitiş Tarihi",
                value=max_hist_date,
                min_value=min_hist_date,
                max_value=max_hist_date,
                key="history_range_end_date",
            )

        if history_start_date > history_end_date:
            st.warning("Başlangıç tarihi bitiş tarihinden büyük olamaz.")
        else:
            history_df = _slice_df_by_date_range(df, history_start_date, history_end_date)

            if history_df.empty or len(history_df) < 10:
                st.warning("Seçilen tarih aralığında yeterli veri bulunamadı. Daha geniş bir aralık seç.")
            else:
                history_price_fig = build_history_range_price_figure(
                    history_df,
                    show_patterns=st.session_state.get("show_chart_patterns", True),
                    show_ema13=st.session_state.get("show_ema13_channel", False),
                )
                st.plotly_chart(history_price_fig, use_container_width=True)

                hist_figs = build_history_range_indicator_figures(history_df, benchmark_df, benchmark_ticker)

                st.subheader("📉 RSI / MACD / ATR%")
                hcol1, hcol2, hcol3 = st.columns(3)
                with hcol1:
                    st.plotly_chart(hist_figs["rsi"], use_container_width=True)
                with hcol2:
                    st.plotly_chart(hist_figs["macd"], use_container_width=True)
                with hcol3:
                    st.plotly_chart(hist_figs["atr"], use_container_width=True)

                st.subheader("📊 Stochastic RSI / Bollinger Genişliği / Hacim Oranı")
                hcol4, hcol5, hcol6 = st.columns(3)
                with hcol4:
                    st.plotly_chart(hist_figs["stoch"], use_container_width=True)
                with hcol5:
                    st.plotly_chart(hist_figs["bbwidth"], use_container_width=True)
                with hcol6:
                    st.plotly_chart(hist_figs["volratio"], use_container_width=True)

                st.subheader("📊 Hacim ve Trend Karşılaştırmaları")
                hcol7, hcol8, hcol9 = st.columns(3)
                with hcol7:
                    st.plotly_chart(hist_figs["vol_market"], use_container_width=True)
                with hcol8:
                    st.plotly_chart(hist_figs["vol_2wk"], use_container_width=True)
                with hcol9:
                    st.plotly_chart(hist_figs["obv"], use_container_width=True)

                st.subheader("🧪 Backtest Özeti (Long-only + Scale Out + Time Stop)")
                history_benchmark_returns = None
                if benchmark_df is not None and not benchmark_df.empty and "Close" in benchmark_df.columns:
                    history_benchmark_slice = _slice_df_by_date_range(benchmark_df, history_start_date, history_end_date)
                    if not history_benchmark_slice.empty:
                        history_benchmark_returns = history_benchmark_slice["Close"].pct_change().dropna()

                hist_eq, hist_tdf, hist_metrics = backtest_long_only(
                    history_df,
                    cfg,
                    risk_free_annual=risk_free_annual,
                    benchmark_returns=history_benchmark_returns,
                )

                hm1, hm2, hm3, hm4, hm5, hm6, hm7, hm8 = st.columns(8)
                hm1.metric("Total Return", f"{hist_metrics['Total Return']*100:.1f}%")
                hm2.metric("Ann Return", f"{hist_metrics['Annualized Return']*100:.1f}%")
                hm3.metric("Sharpe", f"{hist_metrics['Sharpe']:.2f}")
                hm4.metric("Max DD", f"{hist_metrics['Max Drawdown']*100:.1f}%")
                hm5.metric("Trades", f"{hist_metrics['Trades']}")
                hm6.metric("Win Rate", f"{hist_metrics['Win Rate']*100:.1f}%")
                hm7.metric("Beta", f"{hist_metrics['Beta']:.2f}")
                hm8.metric("Info Ratio", f"{hist_metrics['Information Ratio']:.2f}")

                hist_eq_fig = go.Figure()
                hist_eq_fig.add_trace(go.Scatter(x=hist_eq.index, y=hist_eq.values, name="Equity"))
                hist_eq_fig.update_layout(height=320, title="Seçilen Tarih Aralığı Backtest Sermaye Eğrisi", yaxis_title="Sermaye", xaxis_title="Tarih")
                st.plotly_chart(hist_eq_fig, use_container_width=True)

                with st.expander("Trade listesi (Seçilen tarih aralığı)", expanded=False):
                    st.dataframe(hist_tdf, use_container_width=True, height=240)

                st.subheader("📦 Hacim Profili (VPVR / POC)")
                hist_vpvr_fig, hist_poc_price = build_volume_profile_figure(history_df, bins=28, lookback=len(history_df))
                hp1, hp2, hp3 = st.columns(3)
                hp1.metric("POC", f"{hist_poc_price:.2f}" if np.isfinite(hist_poc_price) else "N/A")
                hist_close_ref = float(history_df["Close"].iloc[-1]) if not history_df.empty else np.nan
                hist_poc_dist = ((hist_poc_price / hist_close_ref) - 1.0) * 100.0 if np.isfinite(hist_poc_price) and np.isfinite(hist_close_ref) and hist_close_ref != 0 else np.nan
                hp2.metric("POC Uzaklık %", f"{hist_poc_dist:+.2f}%" if np.isfinite(hist_poc_dist) else "N/A")
                hp3.metric("Profil Bar Sayısı", f"{len(history_df)}")
                st.plotly_chart(hist_vpvr_fig, use_container_width=True)


with tab_education:
    render_education_center_tab()

with tab_index_center:
    st.header("📉 BIST Endeks Merkezi")
    render_page_education_expander([("index_center_page","BIST Endeks Merkezi Nasıl Okunur?"),("ema","EMA"),("rsi","RSI"),("macd","MACD"),("atr_pct","ATR%"),("vpvr","VPVR"),("poc","POC"),("triple_screen","Triple Screen"),("future_price","Future Price")])
    st.caption("BIST 30, BIST 100 ve BIST Tüm endeksleri için dashboard benzeri teknik görünüm, 3 Ekranlı Sistem ve Future Price panelleri.")

    index_map = _bist_index_mapping()
    index_labels = list(index_map.keys())
    default_index_label = "BIST 100" if "BIST 100" in index_labels else index_labels[0]

    selected_index_label = st.selectbox(
        "Endeks Seç",
        options=index_labels,
        index=index_labels.index(default_index_label),
        key="bist_index_center_select",
    )
    selected_index_ticker = index_map[selected_index_label]

    state_symbol_key = "bist_index_center_last_symbol"
    if st.session_state.get(state_symbol_key) != selected_index_ticker:
        st.session_state[state_symbol_key] = selected_index_ticker
        for k in [
            "bist_index_center_future_price_result",
            "bist_index_center_future_price_horizon",
            "bist_index_center_future_price_selected_model",
            "bist_index_center_run_triple_screen",
        ]:
            st.session_state.pop(k, None)

    with st.spinner(f"{selected_index_label} verileri hazırlanıyor..."):
        index_ctx = _prepare_dashboard_context_for_symbol(
            selected_index_ticker,
            cfg=cfg,
            current_interval=interval,
            current_period=period,
            end_date=custom_end_date,
            force_latest=force_latest_candle,
            live_override=use_live_last_override,
            use_market_filter=use_bist_filter,
            use_htf_filter=use_higher_tf_filter,
            risk_free_rate=risk_free_annual,
        )

    idx_sub_dash, idx_sub_triple, idx_sub_future = st.tabs(["📊 Dashboard", "📺 3 Ekranlı Sistem", "🔮 Future Price"])

    with idx_sub_dash:
        render_page_education_expander([("dashboard_page","Endeks Dashboard Nasıl Okunur?"),("ema","EMA"),("bollinger","Bollinger"),("rsi","RSI"),("macd","MACD"),("atr_pct","ATR%"),("volume_ratio","Hacim Oranı"),("obv","OBV"),("vpvr","VPVR"),("poc","POC"),("support_resistance","Destek/Direnç")])
        _render_dashboard_like_context(index_ctx, selected_index_label, interval, period)

    with idx_sub_triple:
        render_page_education_expander([("triple_page","3 Ekranlı Sistem Nasıl Okunur?"),("triple_screen","Triple Screen"),("macd","Haftalık MACD"),("ema","EMA"),("adx","ADX"),("rsi","RSI"),("stochastic","Stokastik"),("force_index","Force Index"),("elder_ray","Elder-Ray"),("divergence","Uyumsuzluk")])
        _render_triple_screen_panel_for_symbol(
            selected_index_ticker,
            selected_index_label,
            session_prefix="bist_index_center",
            live_override=use_live_last_override,
        )

    with idx_sub_future:
        render_page_education_expander([("future_page","Future Price Nasıl Okunur?"),("future_price","Future Price"),("horizon_bars","Tahmin Ufku"),("mae","MAE"),("rmse","RMSE"),("mape","MAPE"),("direction_acc","Yön Doğruluğu"),("confidence","Güven Skoru"),("train_test","Eğitim/Test"),("trend_regime","Trend Rejimi"),("vol_regime","Volatilite Rejimi")])
        if index_ctx.get("error"):
            st.warning(index_ctx["error"])
        else:
            _render_future_price_panel_for_df(
                index_ctx["df"],
                selected_index_label,
                interval,
                period,
                session_prefix="bist_index_center",
            )

with tab_triple:
    st.header("📺 Üçlü Ekran Trading Sistemi (Triple Screen)")
    render_page_education_expander([("triple_page","3 Ekranlı Sistem Nasıl Okunur?"),("triple_screen","Triple Screen"),("macd","Haftalık MACD"),("ema","EMA"),("adx","ADX"),("rsi","RSI"),("stochastic","Stokastik"),("force_index","Force Index"),("elder_ray","Elder-Ray"),("divergence","Uyumsuzluk")])
    st.caption("Dr. Alexander Elder'in 3 Ekranlı sistemine dayanan, trend, osilatör ve giriş seviyesi analizleri.")
    
    if not st.session_state.ta_ran:
        st.info("Sol menüden 'Teknik Analizi Çalıştır' butonuna basarak sistemi aktifleştirmelisin.")
    else:
        if st.button("Üçlü Ekran Verilerini Getir ve Analiz Et", key="run_triple"):
            st.session_state.run_triple_screen = True
            
        if st.session_state.get("run_triple_screen", False):
            with st.spinner("3 Ekran verileri hesaplanıyor (1W, 1D, 1H)..."):
                
                df_1w = load_data_cached(ticker, "2y", "1wk")
                df_1d = load_data_cached(ticker, "1y", "1d", force_latest=force_latest_candle)
                df_1h = load_data_cached(ticker, "60d", "1h")

                if use_live_last_override:
                    df_1w = apply_live_last_override_to_df(df_1w, live_price)
                    df_1d = apply_live_last_override_to_df(df_1d, live_price)
                    df_1h = apply_live_last_override_to_df(df_1h, live_price)
                
                if df_1w.empty or df_1d.empty or df_1h.empty:
                    st.error("Bazı zaman dilimleri için veri çekilemedi (Hisse senedi yeni olabilir veya API gecikmesi var).")
                else:
                    t_screen1, t_screen2, t_screen3 = st.tabs(["1. Ekran (Haftalık)", "2. Ekran (Günlük)", "3. Ekran (1 Saatlik)"])
                    
                    with t_screen1:
                        st.subheader("1. Ekran: Haftalık (Ana Trend)")
                        m_line, m_sig, m_hist = macd(df_1w["Close"])
                        
                        ema_1w_13 = ema(df_1w["Close"], 13)
                        ema_1w_26 = ema(df_1w["Close"], 26)
                        
                        last_close_1w = df_1w["Close"].iloc[-1]
                        if ema_1w_13.iloc[-1] > ema_1w_26.iloc[-1] and last_close_1w > ema_1w_13.iloc[-1]:
                            ema1w_sig = "AL"
                        elif ema_1w_13.iloc[-1] < ema_1w_26.iloc[-1] and last_close_1w < ema_1w_13.iloc[-1]:
                            ema1w_sig = "SAT"
                        else:
                            ema1w_sig = "BEKLE"
                        
                        last_hist = float(m_hist.iloc[-1])
                        prev_hist = float(m_hist.iloc[-2])
                        slope_up = last_hist > prev_hist
                        
                        div_macd, macd_ago = check_bullish_divergence(df_1w["Close"], m_hist, volume=df_1w["Volume"])

                        adx_1w, pdi_1w, mdi_1w = adx_indicator(df_1w["High"], df_1w["Low"], df_1w["Close"])
                        adx_val_1w = adx_1w.iloc[-1]
                        pdi_val_1w = pdi_1w.iloc[-1]
                        mdi_val_1w = mdi_1w.iloc[-1]
                        
                        if adx_val_1w >= 25 and pdi_val_1w > mdi_val_1w:
                            adx_sig_1w = "AL (Güçlü Trend)"
                        elif adx_val_1w >= 25 and mdi_val_1w > pdi_val_1w:
                            adx_sig_1w = "SAT (Güçlü Trend)"
                        else:
                            adx_sig_1w = "BEKLE (Zayıf Trend)"
                        
                        c1w_1, c1w_2, c1w_3 = st.columns(3)
                        c1w_1.metric("MACD Histogram Eğimi", "YUKARI (AL Sinyali)" if slope_up else "AŞAĞI (SAT Sinyali)", f"{last_hist - prev_hist:.2f}")
                        c1w_2.metric("Haftalık EMA (13-26)", ema1w_sig, f"EMA13: {ema_1w_13.iloc[-1]:.2f} | EMA26: {ema_1w_26.iloc[-1]:.2f}")
                        c1w_3.metric("ADX (14)", adx_sig_1w, f"ADX: {adx_val_1w:.1f} | +DI: {pdi_val_1w:.1f} | -DI: {mdi_val_1w:.1f}")
                        
                        if div_macd:
                            st.success(f"🚀 Sistem Haftalık MACD Histogramında **Pozitif Uyumsuzluk** tespit etti! ({macd_ago} bar önce)")
                            
                        fig1_price = go.Figure()
                        fig1_price.add_trace(go.Candlestick(x=df_1w.index, open=df_1w["Open"], high=df_1w["High"], low=df_1w["Low"], close=df_1w["Close"], name="Fiyat"))
                        fig1_price.add_trace(go.Scatter(x=df_1w.index, y=ema_1w_13, name="EMA 13", line=dict(color='blue')))
                        fig1_price.add_trace(go.Scatter(x=df_1w.index, y=ema_1w_26, name="EMA 26", line=dict(color='red')))
                        fig1_price.update_layout(title="Haftalık Fiyat ve EMA (13 & 26)", height=350, xaxis_rangeslider_visible=False)
                        st.plotly_chart(fig1_price, use_container_width=True)

                        fig1 = go.Figure()
                        colors = ['green' if x > 0 else 'red' for x in m_hist.diff()]
                        fig1.add_trace(go.Bar(x=df_1w.index, y=m_hist, name="MACD Hist", marker_color=colors))
                        fig1.update_layout(title="Haftalık MACD Histogramı", height=250)
                        st.plotly_chart(fig1, use_container_width=True)

                        fig1_adx = go.Figure()
                        fig1_adx.add_trace(go.Scatter(x=df_1w.index, y=adx_1w, name="ADX", line=dict(color='black', width=2.5)))
                        fig1_adx.add_trace(go.Scatter(x=df_1w.index, y=pdi_1w, name="+DI", line=dict(color='green')))
                        fig1_adx.add_trace(go.Scatter(x=df_1w.index, y=mdi_1w, name="-DI", line=dict(color='red')))
                        fig1_adx.add_hline(y=25, line_dash="dash", line_color="gray", annotation_text="Trend Başlangıcı (25)")
                        fig1_adx.add_hline(y=50, line_dash="dot", line_color="purple", annotation_text="Aşırı Güçlü Trend (50)")
                        fig1_adx.add_hrect(y0=25, y1=100, fillcolor="rgba(0, 255, 0, 0.05)", layer="below", line_width=0)
                        fig1_adx.add_hrect(y0=0, y1=25, fillcolor="rgba(255, 0, 0, 0.05)", layer="below", line_width=0)
                        fig1_adx.update_layout(title="Haftalık ADX ve Yön Göstergeleri (+DI / -DI)", height=250)
                        st.plotly_chart(fig1_adx, use_container_width=True)

                    with t_screen2:
                        st.subheader("2. Ekran: Günlük (Osilatörler ve Sapmalar)")
                        
                        ema_1d_11 = ema(df_1d["Close"], 11)
                        ema_1d_22 = ema(df_1d["Close"], 22)
                        
                        last_close_1d = df_1d["Close"].iloc[-1]
                        if ema_1d_11.iloc[-1] > ema_1d_22.iloc[-1] and last_close_1d > ema_1d_11.iloc[-1]:
                            ema1d_sig = "AL"
                        elif ema_1d_11.iloc[-1] < ema_1d_22.iloc[-1] and last_close_1d < ema_1d_11.iloc[-1]:
                            ema1d_sig = "SAT"
                        else:
                            ema1d_sig = "BEKLE"
                        
                        st.metric("Günlük EMA (11-22)", ema1d_sig, f"EMA11: {ema_1d_11.iloc[-1]:.2f} | EMA22: {ema_1d_22.iloc[-1]:.2f}")

                        fi = force_index(df_1d["Close"], df_1d["Volume"])
                        fi_ema13 = ema(fi, 13)
                        fi_ema2 = ema(fi, 2)
                        
                        rsi13 = rsi(df_1d["Close"], 13)
                        
                        stoch_k, stoch_d = stochastic(df_1d["High"], df_1d["Low"], df_1d["Close"], k_period=5, d_period=3)
                        
                        er_ema, bull_p, bear_p = elder_ray(df_1d["High"], df_1d["Low"], df_1d["Close"], 13)
                        
                        fi_al = (fi_ema2.iloc[-1] < 0) and (fi_ema2.iloc[-1] > fi_ema2.iloc[-2])
                        rsi_al = (rsi13.iloc[-1] < 30)
                        stoch_al = (stoch_k.iloc[-1] < 20)
                        
                        er_ema_up = (er_ema.iloc[-1] > er_ema.iloc[-2])
                        bp_neg_but_rising = (bear_p.iloc[-1] < 0) and (bear_p.iloc[-1] > bear_p.iloc[-2])
                        er_al = er_ema_up and bp_neg_but_rising
                        
                        div_rsi, rsi_ago = check_bullish_divergence(df_1d["Close"], rsi13, volume=df_1d["Volume"])
                        div_stoch, stoch_ago = check_bullish_divergence(df_1d["Close"], stoch_k, volume=df_1d["Volume"])
                        div_er, er_ago = check_bullish_divergence(df_1d["Close"], bear_p, volume=df_1d["Volume"])
                        div_er_bear, er_bear_ago = check_bearish_divergence(df_1d["Close"], bull_p, volume=df_1d["Volume"])
                        
                        adx_1d, pdi_1d, mdi_1d = adx_indicator(df_1d["High"], df_1d["Low"], df_1d["Close"])
                        adx_val_1d = adx_1d.iloc[-1]
                        pdi_val_1d = pdi_1d.iloc[-1]
                        mdi_val_1d = mdi_1d.iloc[-1]
                        
                        if adx_val_1d >= 25 and pdi_val_1d > mdi_val_1d:
                            adx_sig_1d = "AL (Güçlü Trend)"
                        elif adx_val_1d >= 25 and mdi_val_1d > pdi_val_1d:
                            adx_sig_1d = "SAT (Güçlü Trend)"
                        else:
                            adx_sig_1d = "BEKLE (Zayıf Trend)"

                        c1, c2, c3, c4, c5 = st.columns(5)
                        c1.metric("Kuvvet Endeksi (FI)", "AL" if fi_al else "BEKLE", "2 EMA Negatif & Yukarı Dönüş" if fi_al else "")
                        c2.metric("RSI (13)", "AL" if rsi_al else "BEKLE", f"{rsi13.iloc[-1]:.1f}")
                        c3.metric("Stokastik (5)", "AL" if stoch_al else "BEKLE", f"{stoch_k.iloc[-1]:.1f}")
                        c4.metric("Elder-Ray", "AL" if er_al else "BEKLE")
                        c5.metric("ADX (14)", adx_sig_1d, f"ADX: {adx_val_1d:.1f} | +DI: {pdi_val_1d:.1f}")
                        
                        if div_rsi:
                            st.success(f"🚀 RSI(13)'te **Pozitif Uyumsuzluk** tespit edildi! ({rsi_ago} bar önce)")
                        if div_stoch:
                            st.success(f"🚀 Stokastik(5)'te **Pozitif Uyumsuzluk** tespit edildi! ({stoch_ago} bar önce)")
                        if div_er:
                            st.success(f"🚀 Elder-Ray Bear Power'da **Pozitif Uyumsuzluk (Boğa Uyumsuzluğu)** tespit edildi! ({er_ago} bar önce)")
                        if div_er_bear:
                            st.warning(f"⚠️ Elder-Ray Bull Power'da **Negatif Uyumsuzluk (Ayı Uyumsuzluğu)** tespit edildi! ({er_bear_ago} bar önce)")
                        
                        if st.session_state.sentiment_summary:
                            st.info(f"**Haber Etkisi Modülü:** {st.session_state.sentiment_summary}")
                            
                        fig2_price = go.Figure()
                        fig2_price.add_trace(go.Candlestick(x=df_1d.index, open=df_1d["Open"], high=df_1d["High"], low=df_1d["Low"], close=df_1d["Close"], name="Fiyat"))
                        fig2_price.add_trace(go.Scatter(x=df_1d.index, y=ema_1d_11, name="EMA 11", line=dict(color='blue')))
                        fig2_price.add_trace(go.Scatter(x=df_1d.index, y=ema_1d_22, name="EMA 22", line=dict(color='red')))
                        fig2_price.update_layout(title="Günlük Fiyat ve EMA (11 & 22)", height=350, xaxis_rangeslider_visible=False)
                        st.plotly_chart(fig2_price, use_container_width=True)
                        
                        fig2_fi = go.Figure()
                        fig2_fi.add_trace(go.Scatter(x=df_1d.index, y=fi_ema13, name="FI 13 EMA", line=dict(color='orange')))
                        fig2_fi.add_trace(go.Bar(x=df_1d.index, y=fi_ema2, name="FI 2 EMA", marker_color='gray'))
                        fig2_fi.update_layout(title="Kuvvet Endeksi (Force Index)", height=250)
                        st.plotly_chart(fig2_fi, use_container_width=True)
                        
                        fig2_er = go.Figure()
                        fig2_er.add_trace(go.Bar(x=df_1d.index, y=bull_p, name="Bull Power", marker_color='green'))
                        fig2_er.add_trace(go.Bar(x=df_1d.index, y=bear_p, name="Bear Power", marker_color='red'))
                        fig2_er.update_layout(title="Elder-Ray (Bull & Bear Power)", height=250)
                        st.plotly_chart(fig2_er, use_container_width=True)

                        fig2_adx = go.Figure()
                        fig2_adx.add_trace(go.Scatter(x=df_1d.index, y=adx_1d, name="ADX", line=dict(color='black', width=2.5)))
                        fig2_adx.add_trace(go.Scatter(x=df_1d.index, y=pdi_1d, name="+DI", line=dict(color='green')))
                        fig2_adx.add_trace(go.Scatter(x=df_1d.index, y=mdi_1d, name="-DI", line=dict(color='red')))
                        
                        fig2_adx.add_hline(y=25, line_dash="dash", line_color="gray", annotation_text="Trend Başlangıcı (25)")
                        fig2_adx.add_hline(y=50, line_dash="dot", line_color="purple", annotation_text="Aşırı Güçlü Trend (50)")
                        fig2_adx.add_hrect(y0=25, y1=100, fillcolor="rgba(0, 255, 0, 0.05)", layer="below", line_width=0)
                        fig2_adx.add_hrect(y0=0, y1=25, fillcolor="rgba(255, 0, 0, 0.05)", layer="below", line_width=0)
                        
                        fig2_adx.update_layout(title="Günlük ADX ve Yön Göstergeleri (+DI / -DI)", height=250)
                        st.plotly_chart(fig2_adx, use_container_width=True)

                    with t_screen3:
                        st.subheader("3. Ekran: 1 Saatlik (Giriş / Çıkış ve Hedefler)")
                        
                        adx_1h, pdi_1h, mdi_1h = adx_indicator(df_1h["High"], df_1h["Low"], df_1h["Close"])
                        adx_val_1h = adx_1h.iloc[-1]
                        pdi_val_1h = pdi_1h.iloc[-1]
                        mdi_val_1h = mdi_1h.iloc[-1]
                        
                        if adx_val_1h >= 25 and pdi_val_1h > mdi_val_1h:
                            adx_sig_1h = "AL (Güçlü Trend)"
                        elif adx_val_1h >= 25 and mdi_val_1h > pdi_val_1h:
                            adx_sig_1h = "SAT (Güçlü Trend)"
                        else:
                            adx_sig_1h = "BEKLE (Zayıf Trend)"
                        
                        st.metric("1 Saatlik ADX (14)", adx_sig_1h, f"ADX: {adx_val_1h:.1f} | +DI: {pdi_val_1h:.1f} | -DI: {mdi_val_1h:.1f}")

                        ema_1h = ema(df_1h["Close"], 13)
                        atr_1h = atr(df_1h["High"], df_1h["Low"], df_1h["Close"], 14)
                        last_atr_1h = float(atr_1h.iloc[-1]) if not pd.isna(atr_1h.iloc[-1]) else 0.0
                        
                        pens = ema_1h - df_1h["Low"]
                        pens_positive = pens[pens > 0]
                        avg_pen = float(pens_positive.mean()) if not pens_positive.empty else 0.0
                        
                        up_pens = df_1h["High"] - ema_1h
                        up_pens_positive = up_pens[up_pens > 0]
                        avg_up_pen = float(up_pens_positive.mean()) if not up_pens_positive.empty else 0.0
                        
                        ema_today = float(ema_1h.iloc[-1])
                        ema_yest = float(ema_1h.iloc[-2])
                        ema_delta = ema_today - ema_yest
                        ema_est_tmrw = ema_today + ema_delta
                        
                        buy_level = ema_est_tmrw - avg_pen
                        
                        stop_loss = buy_level - (1.5 * last_atr_1h) if last_atr_1h > 0 else buy_level * 0.98
                        risk = buy_level - stop_loss
                        
                        target_1 = ema_est_tmrw + avg_up_pen
                        target_2 = buy_level + (risk * 2)
                        
                        st.markdown(f"""
                        **Hesaplamalar ve Strateji (Buy Limit & Hedefler):**
                        * 📌 **Güncel EMA (13):** {ema_today:.2f} | **Bir Sonraki Tahmini EMA:** {ema_est_tmrw:.2f}
                        * 🟢 **Önerilen Alış Seviyesi (Buy Limit): {buy_level:.2f}** *(Ortalama {avg_pen:.2f} düşüş penetrasyonu ile)*
                        * 🔴 **Zarar Kes (Stop-Loss): {stop_loss:.2f}** *(Alışın 1.5 ATR altı. Risk: {risk:.2f})*
                        * 🎯 **Hedef 1 (Kısa Vade): {target_1:.2f}** *(Simetrik Yükseliş Penetrasyonu)*
                        * 🚀 **Hedef 2 (Trend - 1:2 RR): {target_2:.2f}** *(Riske edilen tutarın 2 katı kazanç)*
                        """)
                        
                        fig3 = go.Figure()
                        fig3.add_trace(go.Candlestick(x=df_1h.index, open=df_1h["Open"], high=df_1h["High"], low=df_1h["Low"], close=df_1h["Close"], name="Price"))
                        fig3.add_trace(go.Scatter(x=df_1h.index, y=ema_1h, name="EMA 13", line=dict(color='blue')))
                        
                        last_time = df_1h.index[-1]
                        next_time = last_time + pd.Timedelta(hours=1)
                        fig3.add_trace(go.Scatter(x=[next_time], y=[ema_est_tmrw], mode='markers', marker=dict(size=10, color='orange'), name="Tahmini EMA"))
                        
                        fig3.add_hline(y=target_2, line_dash="dash", line_color="darkgreen", annotation_text="Hedef 2 (1:2 RR)", annotation_position="top left")
                        fig3.add_hline(y=target_1, line_dash="dashdot", line_color="cyan", annotation_text="Hedef 1 (Simetrik)", annotation_position="top left")
                        fig3.add_hline(y=buy_level, line_dash="dash", line_color="lime", annotation_text="Limit Alış Seviyesi", annotation_position="bottom left")
                        fig3.add_hline(y=stop_loss, line_dash="dot", line_color="red", annotation_text="Stop-Loss (1.5 ATR)", annotation_position="bottom left")
                        
                        fig3.update_layout(title="1 Saatlik Giriş/Çıkış Stratejisi (Alış, Hedef ve Stop)", height=450, xaxis_rangeslider_visible=False)
                        st.plotly_chart(fig3, use_container_width=True)

                        fig3_adx = go.Figure()
                        fig3_adx.add_trace(go.Scatter(x=df_1h.index, y=adx_1h, name="ADX", line=dict(color='black', width=2.5)))
                        fig3_adx.add_trace(go.Scatter(x=df_1h.index, y=pdi_1h, name="+DI", line=dict(color='green')))
                        fig3_adx.add_trace(go.Scatter(x=df_1h.index, y=mdi_1h, name="-DI", line=dict(color='red')))
                        
                        fig3_adx.add_hline(y=25, line_dash="dash", line_color="gray", annotation_text="Trend Başlangıcı (25)")
                        fig3_adx.add_hline(y=50, line_dash="dot", line_color="purple", annotation_text="Aşırı Güçlü Trend (50)")
                        fig3_adx.add_hrect(y0=25, y1=100, fillcolor="rgba(0, 255, 0, 0.05)", layer="below", line_width=0)
                        fig3_adx.add_hrect(y0=0, y1=25, fillcolor="rgba(255, 0, 0, 0.05)", layer="below", line_width=0)
                        
                        fig3_adx.update_layout(title="1 Saatlik ADX ve Yön Göstergeleri (+DI / -DI)", height=250)
                        st.plotly_chart(fig3_adx, use_container_width=True)



with tab_indicator_stats:
    st.header("📈 İndikatör İstatistik")
    render_page_education_expander([("indicator_stats_page","İndikatör İstatistik Nasıl Okunur?"),("divergence","Uyumsuzluk"),("rsi","RSI"),("macd","MACD"),("adx","ADX"),("elder_ray","Elder-Ray"),("force_index","Force Index"),("stochastic","Stokastik")])
    st.caption("Seçilen hisse ve indikatör için geçmiş oluşumları tarar; kaç kez oluştuğunu, kaç kez çalıştığını, trendin ters yöne dönene kadar ortalama kaç gün sürdüğünü ve oluşum sonrası yüzde kaç yükselip/düştüğünü istatistiksel olarak verir.")

    if not st.session_state.ta_ran:
        st.info("Sol menüden 'Teknik Analizi Çalıştır' butonuna basarak sistemi aktifleştirmelisin.")
    else:
        stats_symbol_options_raw = [str(x).upper() for x in universe]
        if not st.session_state.screener_df.empty and "ticker" in st.session_state.screener_df.columns:
            stats_symbol_options_raw = st.session_state.screener_df["ticker"].astype(str).str.upper().tolist() + stats_symbol_options_raw
        if ticker:
            stats_symbol_options_raw = [ticker] + stats_symbol_options_raw

        stats_symbol_options = []
        seen_stats = set()
        for opt in stats_symbol_options_raw:
            nopt = naked_ticker(opt)
            if nopt not in seen_stats:
                seen_stats.add(nopt)
                stats_symbol_options.append(opt)

        default_stats_symbol = ticker if ticker in stats_symbol_options else (stats_symbol_options[0] if stats_symbol_options else "")
        catalog = get_indicator_stats_catalog()
        label_to_key = {v["label"]: k for k, v in catalog.items()}
        ordered_labels = list(label_to_key.keys())

        s1, s2, s3, s4 = st.columns([2.2, 3.2, 1.2, 1.2])
        with s1:
            stats_symbol_raw = st.selectbox("Hisse Seç", options=stats_symbol_options, index=stats_symbol_options.index(default_stats_symbol) if default_stats_symbol in stats_symbol_options else 0, key="stats_symbol_raw")
        with s2:
            selected_indicator_label = st.selectbox("İndikatör / Formasyon Seç", options=ordered_labels, index=0, key="selected_indicator_label")
        with s3:
            max_bars_forward = st.slider("İleri Bakış (bar)", min_value=5, max_value=60, value=20, step=1, key="stats_max_bars_forward")
        with s4:
            move_threshold_pct = st.slider("Çalıştı Eşiği %", min_value=1.0, max_value=10.0, value=2.0, step=0.5, key="stats_move_threshold_pct")

        run_indicator_stats = st.button("📈 İstatistik Analizini Çalıştır", key="run_indicator_stats", use_container_width=True)

        if run_indicator_stats:
            selected_key = label_to_key[selected_indicator_label]
            with st.spinner("İndikatör geçmişi hazırlanıyor ve istatistik hesaplanıyor..."):
                prep = _prepare_indicator_stats_frames(stats_symbol_raw, market, cfg)
                if prep.get("error"):
                    st.session_state.indicator_stats_result = {"error": prep.get("error")}
                else:
                    sig_mask, src_df, src_tf, src_dir = build_indicator_signal_series(selected_key, prep["daily"], prep["weekly"])
                    summary, details_df = compute_indicator_signal_statistics(
                        src_df,
                        sig_mask,
                        direction=src_dir,
                        max_bars=int(max_bars_forward),
                        move_threshold=float(move_threshold_pct) / 100.0,
                        timeframe=src_tf,
                    )
                    occ_chart = build_indicator_occurrence_chart(src_df, sig_mask, title=f"{prep['ticker']} — {selected_indicator_label}") if src_df is not None and not src_df.empty else None
                    st.session_state.indicator_stats_result = {
                        "error": None,
                        "ticker": prep["ticker"],
                        "indicator_label": selected_indicator_label,
                        "indicator_key": selected_key,
                        "timeframe": src_tf,
                        "direction": src_dir,
                        "summary": summary,
                        "details_df": details_df,
                        "chart": occ_chart,
                    }

        stats_result = st.session_state.get("indicator_stats_result")
        if stats_result:
            if stats_result.get("error"):
                st.warning(stats_result["error"])
            else:
                summary = stats_result.get("summary", {})
                st.success(f"Analiz tamamlandı: {stats_result.get('ticker', '')} | {stats_result.get('indicator_label', '')} | Zaman dilimi: {stats_result.get('timeframe', '')}")

                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Toplam Oluşum", f"{summary.get('occurrences', 0)}")
                m2.metric("Çalışan Oluşum", f"{summary.get('worked', 0)}")
                m3.metric("Başarı Oranı", f"%{summary.get('win_rate', np.nan):.1f}" if np.isfinite(summary.get('win_rate', np.nan)) else "N/A")
                m4.metric("Ort. Trend Süresi", f"{summary.get('avg_days', np.nan):.1f} gün" if np.isfinite(summary.get('avg_days', np.nan)) else "N/A")

                m5, m6, m7, m8 = st.columns(4)
                m5.metric("Medyan Trend Süresi", f"{summary.get('median_days', np.nan):.1f} gün" if np.isfinite(summary.get('median_days', np.nan)) else "N/A")
                m6.metric("Ort. Maks. Yükseliş", f"%{summary.get('avg_up', np.nan):.2f}" if np.isfinite(summary.get('avg_up', np.nan)) else "N/A")
                m7.metric("Ort. Maks. Düşüş", f"%{summary.get('avg_down', np.nan):.2f}" if np.isfinite(summary.get('avg_down', np.nan)) else "N/A")
                m8.metric("Ort. Baskın Hareket", f"%{summary.get('avg_dom', np.nan):.2f}" if np.isfinite(summary.get('avg_dom', np.nan)) else "N/A")

                m9, m10 = st.columns(2)
                m9.metric("En İyi Baskın Hareket", f"%{summary.get('best', np.nan):.2f}" if np.isfinite(summary.get('best', np.nan)) else "N/A")
                m10.metric("En Kötü Baskın Hareket", f"%{summary.get('worst', np.nan):.2f}" if np.isfinite(summary.get('worst', np.nan)) else "N/A")

                if stats_result.get("chart") is not None:
                    st.plotly_chart(stats_result["chart"], use_container_width=True)

                with st.expander("Detaylı Oluşum Tablosu", expanded=False):
                    details_df = stats_result.get("details_df", pd.DataFrame())
                    if details_df is not None and not details_df.empty:
                        st.dataframe(details_df, use_container_width=True, height=360)
                    else:
                        st.info("Bu indikatör için yeterli geçmiş oluşum bulunamadı.")

with tab_future:
    st.header("🔮 Future Price")
    render_page_education_expander([("future_page","Future Price Nasıl Okunur?"),("future_price","Future Price"),("horizon_bars","Tahmin Ufku"),("mae","MAE"),("rmse","RMSE"),("mape","MAPE"),("direction_acc","Yön Doğruluğu"),("confidence","Güven Skoru"),("train_test","Eğitim/Test"),("trend_regime","Trend Rejimi"),("vol_regime","Volatilite Rejimi")])
    st.caption("Makine öğrenmesi tabanlı çoklu model karşılaştırması ile seçili sembolün mevcut zaman diliminde ileri bar kapanış fiyat tahmini üretir.")

    if not st.session_state.ta_ran:
        st.info("Sol menüden 'Teknik Analizi Çalıştır' butonuna basarak sistemi aktifleştirmelisin.")
    else:
        st.markdown(f"**Aktif sembol:** `{ticker}`  |  **Aktif zaman dilimi:** `{interval}`  |  **Aktif periyot:** `{period}`")
        future_horizon = st.number_input(
            "Kaç bar/gün sonrası tahmin yapılsın?",
            min_value=1,
            max_value=250,
            value=5,
            step=1,
            help="Mevcut seçili zaman dilimine göre çalışır. Örn. interval 1d ise 5 = 5 gün/bar sonrası kapanış tahmini.",
            key="future_horizon_input",
        )

        run_future_price = st.button("🤖 Future Price Tahmini Yap", key="run_future_price", use_container_width=True)

        if run_future_price:
            with st.spinner("Makine öğrenmesi modelleri eğitiliyor, walk-forward test yapılıyor ve tahmin üretiliyor..."):
                st.session_state.future_price_result = future_price_ml_forecast(df, int(future_horizon))
                st.session_state.future_price_horizon = int(future_horizon)

        fp_result = st.session_state.get("future_price_result")
        if fp_result and not fp_result.get("error"):
            compare_df = fp_result["compare_df"].copy()
            best_model_name = fp_result["best_model_name"]

            st.subheader("🏁 Model Karşılaştırma Panosu")
            st.dataframe(compare_df, use_container_width=True, height=220)

            model_options = compare_df["Model"].tolist()
            best_idx = model_options.index(best_model_name) if best_model_name in model_options else 0
            selected_model_name = st.selectbox(
                "Grafik ve detay için model seç",
                options=model_options,
                index=best_idx,
                key="future_price_selected_model",
                help="Sayfadaki tüm modeller görünür. Buradan seçtiğin modele göre grafik, olasılıklar ve önem sıralaması güncellenir.",
            )
            models_map = fp_result.get("models", {})
            if not isinstance(models_map, dict):
                models_map = {}
            selected_model = models_map.get(selected_model_name, fp_result.get("best_model_result", {}))
            if not selected_model:
                st.warning("Seçilen modelin detay kaydı bulunamadı; en iyi modelin sonucu gösteriliyor.")
                selected_model = fp_result.get("best_model_result", {})
            horizon_value = int(fp_result.get("horizon_bars", future_horizon))

            fp_current_price = _fp_safe_value(selected_model, fp_result, "current_price", np.nan)
            fp_pred_price = _fp_safe_value(selected_model, fp_result, "predicted_price", np.nan)
            fp_delta_pct = _fp_safe_value(selected_model, fp_result, "delta_pct", np.nan)
            fp_pred_low = _fp_safe_value(selected_model, fp_result, "predicted_low", np.nan)
            fp_pred_high = _fp_safe_value(selected_model, fp_result, "predicted_high", np.nan)
            fp_confidence = _fp_safe_value(selected_model, fp_result, "confidence_score", np.nan)
            fp_mae = _fp_safe_value(selected_model, fp_result, "mae", np.nan)
            fp_rmse = _fp_safe_value(selected_model, fp_result, "rmse", np.nan)
            fp_mape = _fp_safe_value(selected_model, fp_result, "mape", np.nan)
            fp_train_rows = _fp_safe_value(selected_model, fp_result, "train_rows", np.nan)
            fp_test_rows = _fp_safe_value(selected_model, fp_result, "test_rows", np.nan)

            fp_c1, fp_c2, fp_c3, fp_c4, fp_c5 = st.columns(5)
            fp_c1.metric("Mevcut Kapanış", _fp_fmt_num(fp_current_price, 2))
            fp_c2.metric(f"{int(horizon_value)} Bar Sonrası Tahmin", _fp_fmt_num(fp_pred_price, 2))
            fp_c3.metric("Tahmini Değişim", _fp_fmt_num(fp_delta_pct, 2, "%") if pd.notna(fp_delta_pct) else "N/A")
            fp_c4.metric("Tahmin Bandı", f"{_fp_fmt_num(fp_pred_low, 2)} - {_fp_fmt_num(fp_pred_high, 2)}" if pd.notna(fp_pred_low) and pd.notna(fp_pred_high) else "N/A")
            fp_c5.metric("Güven Skoru", _fp_fmt_num(fp_confidence, 1, "%") if pd.notna(fp_confidence) else "N/A")

            fp_m1, fp_m2, fp_m3, fp_m4 = st.columns(4)
            fp_m1.metric("MAE", _fp_fmt_num(fp_mae, 4))
            fp_m2.metric("RMSE", _fp_fmt_num(fp_rmse, 4))
            fp_m3.metric("MAPE", "%" + _fp_fmt_num(fp_mape, 2) if pd.notna(fp_mape) else "N/A")
            fp_m4.metric("Eğitim/Test Satırı", f"{int(fp_train_rows) if pd.notna(fp_train_rows) else 'N/A'}/{int(fp_test_rows) if pd.notna(fp_test_rows) else 'N/A'}")

            fp_p1, fp_p2, fp_p3, fp_p4 = st.columns(4)
            fp_p1.metric("Yükseliş Olasılığı", f"%{selected_model['up_prob']:.1f}" if pd.notna(selected_model['up_prob']) else "N/A")
            fp_p2.metric("Düşüş Olasılığı", f"%{selected_model['down_prob']:.1f}" if pd.notna(selected_model['down_prob']) else "N/A")
            fp_p3.metric("Yatay Olasılığı", f"%{selected_model['flat_prob']:.1f}" if pd.notna(selected_model['flat_prob']) else "N/A")
            fp_p4.metric("Yön Doğruluğu", f"%{selected_model['direction_acc']:.1f}" if pd.notna(selected_model['direction_acc']) else "N/A")

            st.subheader("🌡️ Rejim Özeti")
            rg1, rg2 = st.columns(2)
            rg1.metric("Trend Rejimi", fp_result.get("trend_regime", "N/A"))
            rg2.metric("Volatilite Rejimi", fp_result.get("vol_regime", "N/A"))
            if st.session_state.get("sentiment_summary"):
                st.info(f"Haber Duyarlılığı Özeti: {st.session_state.get('sentiment_summary')[:300]}")

            st.subheader("📈 Seçili Modele Göre Grafik")
            future_fig = go.Figure()
            recent_actual = selected_model.get("recent_actual")
            test_actual_series = selected_model.get("test_actual_series")
            test_pred_series = selected_model.get("test_pred_series")
            if recent_actual is not None and not recent_actual.empty:
                future_fig.add_trace(go.Scatter(x=recent_actual.index, y=recent_actual.values, name="Geçmiş Kapanış", line=dict(color="royalblue", width=2)))
            if test_actual_series is not None and not test_actual_series.empty:
                future_fig.add_trace(go.Scatter(x=test_actual_series.index, y=test_actual_series.values, name="Test Gerçek", line=dict(color="seagreen", width=2)))
            if test_pred_series is not None and not test_pred_series.empty:
                future_fig.add_trace(go.Scatter(x=test_pred_series.index, y=test_pred_series.values, name=f"{selected_model_name} Test Tahmini", line=dict(color="darkorange", width=2, dash="dot")))

            future_x = recent_actual.index[-1] if recent_actual is not None and not recent_actual.empty else pd.Timestamp.now()
            future_fig.add_trace(go.Scatter(
                x=[future_x],
                y=[selected_model['predicted_price']],
                mode="markers",
                name="Tahmini Gelecek Fiyat",
                marker=dict(size=14, color="darkorange", symbol="diamond"),
            ))
            if pd.notna(selected_model['predicted_low']) and pd.notna(selected_model['predicted_high']):
                future_fig.add_trace(go.Scatter(
                    x=[future_x, future_x],
                    y=[selected_model['predicted_low'], selected_model['predicted_high']],
                    mode="lines",
                    name="Tahmin Bandı",
                    line=dict(width=8, color="rgba(255,140,0,0.35)"),
                ))
                future_fig.add_hrect(
                    y0=selected_model['predicted_low'],
                    y1=selected_model['predicted_high'],
                    fillcolor="rgba(255,165,0,0.08)",
                    line_width=0,
                    annotation_text="Olası Fiyat Bandı",
                    annotation_position="top left",
                )
            future_fig.add_hline(y=selected_model['predicted_price'], line_dash="dash", line_color="darkorange", annotation_text=f"Tahmin: {selected_model['predicted_price']:.2f}")
            future_fig.update_layout(height=460, title=f"Future Price Tahmini - {selected_model_name} ({int(fp_result['horizon_bars'])} bar sonrası)", xaxis_title="Tarih", yaxis_title="Fiyat")
            st.plotly_chart(future_fig, use_container_width=True)

            st.subheader("🧠 Feature Importance / Etki Sıralaması")
            fi_df = selected_model.get("feature_importance_df")
            if fi_df is not None and not fi_df.empty:
                fi_fig = go.Figure()
                fi_top = fi_df.head(12).iloc[::-1]
                fi_fig.add_trace(go.Bar(x=fi_top["Importance"], y=fi_top["Feature"], orientation="h", name="Önem"))
                fi_fig.update_layout(height=420, title=f"{selected_model_name} - En Etkili Özellikler", xaxis_title="Önem", yaxis_title="Feature")
                st.plotly_chart(fi_fig, use_container_width=True)
                st.dataframe(fi_df, use_container_width=True, height=260)
            else:
                st.info("Bu model için yorumlanabilir feature importance bilgisi üretilemedi.")

            st.subheader("📏 Horizon Kalite Özeti")
            horizon_quality_df = fp_result.get("horizon_quality_df")
            if horizon_quality_df is not None and not horizon_quality_df.empty:
                st.dataframe(horizon_quality_df, use_container_width=True, height=220)

            st.info(
                f"Seçili model `{selected_model_name}` son kullanılabilir barı `{selected_model.get('last_feature_index', 'N/A')}` üzerinden tahmin üretti. "
                "Bu çıktı eğitim amaçlıdır; kesin fiyat bilgisi değildir. Sayfadaki ⭐ işaretli model, mevcut karşılaştırmada en düşük hata ile öne çıkmıştır."
            )
        elif fp_result and fp_result.get("error"):
            st.warning(fp_result["error"])



with tab_chart_patterns:
    st.header("📐 Grafik Formasyonları")
    render_page_education_expander([("chart_patterns_page","Grafik Formasyonları Nasıl Okunur?"),("support_resistance","Destek/Direnç"),("risk_reward","Risk/Ödül"),("ema","EMA"),("volume_ratio","Hacim Oranı")])
    st.caption("Seçtiğin zaman dilimlerinde seçili hisseyi klasik grafik formasyonları için tarar. Tespit edilen formasyonlar yeşil işaretlenir; tıkladığında grafikte mavi alan ile gösterilir.")

    if not st.session_state.ta_ran:
        st.info("Sol menüden 'Teknik Analizi Çalıştır' butonuna basarak sistemi aktifleştirmelisin.")
    else:
        st.markdown(f"**Aktif sembol:** `{ticker}`")

        cp_timeframe_labels = {"1d": "1 Günlük", "1wk": "1 Haftalık"}
        selected_pattern_timeframes = st.multiselect(
            "Taranacak zaman dilimleri",
            options=["1d", "1wk"],
            default=["1d"],
            format_func=lambda x: cp_timeframe_labels.get(x, x),
            key="chart_pattern_timeframes",
        )

        run_chart_pattern_scan = st.button("📐 Grafik Formasyonlarını Tara", key="run_chart_pattern_scan", use_container_width=True)

        if run_chart_pattern_scan:
            scan_results = {}
            with st.spinner("Grafik formasyonları taranıyor..."):
                for tf in selected_pattern_timeframes:
                    scan_results[tf] = scan_chart_patterns_for_symbol(ticker, tf)
            st.session_state.chart_pattern_scan_results = scan_results
            st.session_state.chart_pattern_selected = None

        chart_scan_results = st.session_state.get("chart_pattern_scan_results", {})

        if chart_scan_results:
            st.subheader("Tespit Listesi")
            detected_button_pressed = False

            for tf in selected_pattern_timeframes:
                tf_result = chart_scan_results.get(tf)
                st.markdown(f"### {cp_timeframe_labels.get(tf, tf)}")

                if not tf_result:
                    st.info("Bu zaman dilimi için henüz tarama çalıştırılmadı.")
                    continue
                if tf_result.get("error"):
                    st.warning(tf_result["error"])
                    continue

                matches = tf_result.get("matches", {})
                found_any = False

                for group_name in CHART_PATTERN_GROUPS:
                    match = matches.get(group_name)
                    left_col, right_col = st.columns([4, 1])
                    with left_col:
                        if match is not None:
                            st.markdown(f"<span style='color:#16a34a; font-weight:700;'>🟢 {group_name}</span> <span style='color:#64748b;'>({match['subtype']})</span>", unsafe_allow_html=True)
                        else:
                            st.markdown(f"<span style='color:#94a3b8;'>⚪ {group_name}</span>", unsafe_allow_html=True)
                    with right_col:
                        if match is not None:
                            found_any = True
                            btn_key = f"show_chart_pattern_{tf}_{group_name}"
                            if st.button("Göster", key=btn_key, use_container_width=True):
                                st.session_state.chart_pattern_selected = {"timeframe": tf, "group": group_name}
                                detected_button_pressed = True

                if not found_any:
                    st.info("Bu zaman diliminde tespit edilen formasyon bulunamadı.")

            selected_info = st.session_state.get("chart_pattern_selected")
            if selected_info:
                selected_tf = selected_info.get("timeframe")
                selected_group = selected_info.get("group")
                selected_result = chart_scan_results.get(selected_tf, {})
                selected_match = (selected_result.get("matches") or {}).get(selected_group)

                if selected_match is not None and selected_result.get("df") is not None and not selected_result.get("df").empty:
                    st.subheader("Grafikte Gösterim")
                    chart_fig = build_chart_pattern_figure(
                        selected_result["df"],
                        selected_match,
                        ticker,
                        cp_timeframe_labels.get(selected_tf, selected_tf),
                    )
                    st.plotly_chart(chart_fig, use_container_width=True)

                    st.info(
                        f"Tespit edilen formasyon: **{selected_match['subtype']}**. "
                        f"Mavi alan formasyonun kapsadığı bölgeyi, mavi noktalar ise formasyonu oluşturan ana salınım noktalarını gösterir. "
                        f"{selected_match.get('note', '')}"
                    )



with tab_social:
    st.header("📣 X + YouTube Trends")
    render_page_education_expander([("social_page","X + YouTube Trends Nasıl Okunur?"),("volume_ratio","İlgi Yoğunluğu"),("trend_regime","Trend Bağlamı")])
    social_tab_x, social_tab_youtube = st.tabs(["𝕏 X Trends", "▶️ YouTube Trends"])

    with social_tab_x:
        st.header("𝕏 X Trends")
        st.caption("Seçili hisse için hem sembol hem de şirketin tam adıyla X recent search yapar; pozitif/negatif post göstergeleri ve grafikler üretir.")

        x_symbol_options = [naked_ticker(x) for x in universe] if universe else [naked_ticker(ticker)]
        default_x_symbol = naked_ticker(ticker) if naked_ticker(ticker) in x_symbol_options else x_symbol_options[0]

        x_symbol_raw = st.selectbox(
            "Hisse Seç",
            options=x_symbol_options,
            index=x_symbol_options.index(default_x_symbol),
            key="x_symbol_raw",
        )
        x_ticker = normalize_ticker(x_symbol_raw, market)

        x_company_name_auto = get_company_name_for_social(x_ticker, market, st.session_state.get("screener_df", pd.DataFrame()))
        x_company_name = st.text_input(
            "Şirketin Tam Adı",
            value=x_company_name_auto,
            key="x_company_name",
            help="X aramasında sembol ile birlikte şirketin tam adı da aranır.",
        )

        xc1, xc2, xc3 = st.columns(3)
        with xc1:
            x_max_results = st.slider("Maksimum Post", min_value=10, max_value=100, value=50, step=10, key="x_max_results")
        with xc2:
            x_token_input = st.text_input("X Bearer Token (opsiyonel)", value="", type="password", key="x_token_input")
        with xc3:
            run_x = st.button("𝕏 X Analizini Çalıştır", key="run_x_analysis", use_container_width=True)

        if run_x:
            x_result = fetch_x_trends_bundle(
                naked_ticker(x_ticker),
                x_company_name.strip(),
                bearer_token_override=x_token_input,
                max_results=int(x_max_results),
            )
            st.session_state.x_trends_result = x_result
            if x_result.get("error") is None:
                st.session_state.x_trends_analysis = build_x_indicators(x_result.get("posts_df"))
            else:
                st.session_state.x_trends_analysis = {"error": x_result.get("error")}

        x_result = st.session_state.get("x_trends_result")
        x_analysis = st.session_state.get("x_trends_analysis")

        if x_result and x_result.get("error"):
            st.warning(x_result["error"])
        elif x_analysis and x_analysis.get("error"):
            st.warning(x_analysis["error"])
        elif x_result and x_analysis:
            x_posts_df = x_result.get("posts_df", pd.DataFrame())

            xg1, xg2, xg3, xg4, xg5, xg6 = st.columns(6)
            xg1.metric("Pozitif Skor", f"%{x_analysis['positive_score']:.1f}")
            xg2.metric("Negatif Skor", f"%{x_analysis['negative_score']:.1f}")
            xg3.metric("Pozitif Post", str(x_analysis["positive_count"]))
            xg4.metric("Negatif Post", str(x_analysis["negative_count"]))
            xg5.metric("Ortalama Etkileşim", f"{x_analysis['avg_engagement']:.1f}" if pd.notna(x_analysis['avg_engagement']) else "N/A")
            xg6.metric("Genel Karar", x_analysis["verdict"])

            xv1, xv2 = st.columns(2)
            with xv1:
                x_volume_df = x_analysis.get("volume_df", pd.DataFrame())
                if not x_volume_df.empty:
                    x_vol_fig = go.Figure()
                    x_vol_fig.add_trace(go.Bar(x=x_volume_df["Day"], y=x_volume_df["Post Count"], name="Post Count"))
                    x_vol_fig.update_layout(height=340, title="Günlük X Post Sayısı", xaxis_title="Tarih", yaxis_title="Post")
                    st.plotly_chart(x_vol_fig, use_container_width=True)

            with xv2:
                x_pol_df = x_analysis.get("polarity_df", pd.DataFrame())
                if not x_pol_df.empty:
                    x_pol_fig = go.Figure()
                    x_pol_fig.add_trace(go.Bar(x=x_pol_df["Polarity"], y=x_pol_df["Count"], name="Count"))
                    x_pol_fig.update_layout(height=340, title="X Polarity Dağılımı", xaxis_title="Polarity", yaxis_title="Adet")
                    st.plotly_chart(x_pol_fig, use_container_width=True)

            st.subheader("🧾 X Sonuç Tablosu")
            if not x_posts_df.empty:
                x_show = x_posts_df[["Date", "Author", "Author Name", "Polarity", "Engagement", "Text"]].copy()
                st.dataframe(x_show.sort_values("Date", ascending=False), use_container_width=True, height=360)
            else:
                st.info("X araması veri döndürmedi.")

    with social_tab_youtube:
        st.header("▶️ YouTube Trends")
        st.caption("Seçili hisse için hem sembol hem de şirketin tam adıyla YouTube araması yapar; pozitif/negatif video göstergeleri ve grafikler üretir.")

        yt_symbol_options = [naked_ticker(x) for x in universe] if universe else [naked_ticker(ticker)]
        default_yt_symbol = naked_ticker(ticker) if naked_ticker(ticker) in yt_symbol_options else yt_symbol_options[0]

        yt_symbol_raw = st.selectbox(
            "Hisse Seç",
            options=yt_symbol_options,
            index=yt_symbol_options.index(default_yt_symbol),
            key="yt_symbol_raw",
        )
        yt_ticker = normalize_ticker(yt_symbol_raw, market)

        yt_company_name_auto = get_company_name_for_social(yt_ticker, market, st.session_state.get("screener_df", pd.DataFrame()))
        yt_company_name = st.text_input(
            "Şirketin Tam Adı",
            value=yt_company_name_auto,
            key="yt_company_name",
            help="YouTube aramasında sembol ile birlikte şirketin tam adı da aranır.",
        )

        yc1, yc2, yc3, yc4 = st.columns(4)
        with yc1:
            yt_lookback = st.selectbox("Zaman Aralığı", ["Son 1 Gün", "Son 7 Gün", "Son 30 Gün", "Son 90 Gün", "Son 12 Ay"], index=2, key="yt_lookback")
        with yc2:
            yt_max_results = st.slider("Maksimum Video", min_value=5, max_value=50, value=25, step=5, key="yt_max_results")
        with yc3:
            yt_api_key_input = st.text_input("YouTube API Key (opsiyonel)", value="", type="password", key="yt_api_key_input")
        with yc4:
            run_yt = st.button("▶️ YouTube Analizini Çalıştır", key="run_youtube_analysis", use_container_width=True)

        if run_yt:
            yt_result = fetch_youtube_trends_bundle(
                naked_ticker(yt_ticker),
                yt_company_name.strip(),
                api_key_override=yt_api_key_input,
                lookback_label=yt_lookback,
                max_results=int(yt_max_results),
            )
            st.session_state.youtube_trends_result = yt_result
            if yt_result.get("error") is None:
                st.session_state.youtube_trends_analysis = build_youtube_indicators(yt_result.get("videos_df"))
            else:
                st.session_state.youtube_trends_analysis = {"error": yt_result.get("error")}

        yt_result = st.session_state.get("youtube_trends_result")
        yt_analysis = st.session_state.get("youtube_trends_analysis")

        if yt_result and yt_result.get("error"):
            st.warning(yt_result["error"])
        elif yt_analysis and yt_analysis.get("error"):
            st.warning(yt_analysis["error"])
        elif yt_result and yt_analysis:
            yt_videos_df = yt_result.get("videos_df", pd.DataFrame())

            yg1, yg2, yg3, yg4, yg5, yg6 = st.columns(6)
            yg1.metric("Pozitif Skor", f"%{yt_analysis['positive_score']:.1f}")
            yg2.metric("Negatif Skor", f"%{yt_analysis['negative_score']:.1f}")
            yg3.metric("Pozitif Video", str(yt_analysis["positive_count"]))
            yg4.metric("Negatif Video", str(yt_analysis["negative_count"]))
            yg5.metric("Ortalama İzlenme", f"{yt_analysis['avg_views']:.1f}" if pd.notna(yt_analysis['avg_views']) else "N/A")
            yg6.metric("Genel Karar", yt_analysis["verdict"])

            yv1, yv2 = st.columns(2)
            with yv1:
                yt_volume_df = yt_analysis.get("volume_df", pd.DataFrame())
                if not yt_volume_df.empty:
                    yt_vol_fig = go.Figure()
                    yt_vol_fig.add_trace(go.Bar(x=yt_volume_df["Day"], y=yt_volume_df["Video Count"], name="Video Count"))
                    yt_vol_fig.update_layout(height=340, title="Günlük YouTube Video Sayısı", xaxis_title="Tarih", yaxis_title="Video")
                    st.plotly_chart(yt_vol_fig, use_container_width=True)

            with yv2:
                yt_pol_df = yt_analysis.get("polarity_df", pd.DataFrame())
                if not yt_pol_df.empty:
                    yt_pol_fig = go.Figure()
                    yt_pol_fig.add_trace(go.Bar(x=yt_pol_df["Polarity"], y=yt_pol_df["Count"], name="Count"))
                    yt_pol_fig.update_layout(height=340, title="YouTube Polarity Dağılımı", xaxis_title="Polarity", yaxis_title="Adet")
                    st.plotly_chart(yt_pol_fig, use_container_width=True)

            st.subheader("🧾 YouTube Sonuç Tablosu")
            if not yt_videos_df.empty:
                yt_show = yt_videos_df[["Published At", "Channel", "Polarity", "View Count", "Like Count", "Comment Count", "Title", "Video URL"]].copy()
                st.dataframe(yt_show.sort_values("Published At", ascending=False), use_container_width=True, height=360)
            else:
                st.info("YouTube araması veri döndürmedi.")


with tab_trend_donchian:
    st.header("📡 Trend + Donchian Sistemleri")
    render_page_education_expander([("trend_donchian_page","Trend + Donchian Sistemleri Nasıl Okunur?"),("trend_patt","Trend with Patt Entry"),("donchian","Donchian"),("donchian_520","5&20"),("richard_dennis","Richard Dennis")])
    st.caption("Bu sekmede trend ve breakout sistemleri özetlenir. 'Trend with Patt Entry' için kamuya açık birebir orijinal kurallar doğrulanamadığı için pratik trend + price action yaklaşımı kullanılmıştır.")

    twp_tab, dc_tab, d520_tab, rd_tab = st.tabs(["📈 Trend with Patt Entry", "📦 Donchian Kanalları", "5&20", "Richard Dennis"])

    with twp_tab:
        st.subheader("Trend with Patt Entry (pratik yaklaşım)")
        render_page_education_expander([("trend_patt","Trend with Patt Entry"),("twp_trend_filter","Trend Filtresi"),("twp_long_setup","LONG Setup"),("twp_short_setup","SHORT Setup"),("twp_sma_relation","SMA50 / SMA200 İlişkisi"),("sma_bias","SMA Bias"),("sma_fast","Hızlı SMA"),("sma_slow","Yavaş SMA")])
        twp_df = trend_with_pattern_entry_signals(df.copy())
        twp_last = twp_df.iloc[-1]
        t1, t2, t3, t4 = st.columns(4)
        trend_dir = "Yukarı" if bool(twp_last.get("TWP_TREND_UP", False)) else ("Aşağı" if bool(twp_last.get("TWP_TREND_DOWN", False)) else "Nötr")
        t1.metric("Trend", trend_dir)
        t2.metric("Long Setup", "VAR ✅" if bool(twp_last.get("TWP_LONG_ENTRY", False)) else "YOK")
        t3.metric("Short Setup", "VAR ⚠️" if bool(twp_last.get("TWP_SHORT_ENTRY", False)) else "YOK")
        t4.metric("SMA50 / SMA200", "Üstünde" if pd.notna(twp_last.get("TWP_SMA50", np.nan)) and pd.notna(twp_last.get("TWP_SMA200", np.nan)) and twp_last["TWP_SMA50"] > twp_last["TWP_SMA200"] else "Altında")
        fig_twp = build_system_overlay_chart(
            twp_df.tail(220),
            "Trend with Patt Entry",
            [("TWP_SMA50", "SMA 50"), ("TWP_SMA200", "SMA 200")],
            [("TWP_LONG_ENTRY", "Long Entry", "triangle-up"), ("TWP_SHORT_ENTRY", "Short Entry", "triangle-down")],
        )
        st.plotly_chart(fig_twp, use_container_width=True)

    with dc_tab:
        st.subheader("Donchian Kanalları")
        render_page_education_expander([("donchian","Donchian Kanalları"),("donchian_upper_band","Üst Bant"),("donchian_mid_band","Orta Bant"),("donchian_lower_band","Alt Bant"),("donchian_position","Konum")])
        dc_df = df.copy()
        dc_df["DC_UPPER20"], dc_df["DC_MID20"], dc_df["DC_LOWER20"] = donchian_channels(dc_df["High"], dc_df["Low"], 20)
        dc_last = dc_df.iloc[-1]
        d1, d2, d3, d4 = st.columns(4)
        d1.metric("Üst Bant", fmt_num(dc_last.get("DC_UPPER20", np.nan)))
        d2.metric("Orta Bant", fmt_num(dc_last.get("DC_MID20", np.nan)))
        d3.metric("Alt Bant", fmt_num(dc_last.get("DC_LOWER20", np.nan)))
        d4.metric("Konum", "Üst Banda Yakın" if pd.notna(dc_last.get("DC_UPPER20", np.nan)) and pd.notna(dc_last.get("DC_LOWER20", np.nan)) and abs(dc_last["Close"] - dc_last["DC_UPPER20"]) < abs(dc_last["Close"] - dc_last["DC_LOWER20"]) else "Alt Banda Yakın")
        fig_dc = build_system_overlay_chart(dc_df.tail(220), "Donchian Kanalı (20)", [("DC_UPPER20", "Üst 20"), ("DC_MID20", "Orta 20"), ("DC_LOWER20", "Alt 20")], [])
        st.plotly_chart(fig_dc, use_container_width=True)

    with d520_tab:
        st.subheader("Donchian 5&20 / MA 5-20 Sistemi")
        render_page_education_expander([("donchian_520","Donchian 5&20"),("d520_state","Durum"),("d520_buy_sig","AL Sinyali"),("d520_sell_sig","SAT Sinyali"),("sma_bias","SMA Bias"),("sma_fast","Hızlı SMA"),("sma_slow","Yavaş SMA")])
        d520_df = donchian_5_20_system(df.copy())
        d520_last = d520_df.iloc[-1]
        k1, k2, k3, k4 = st.columns(4)
        state_520 = "LONG ✅" if bool(d520_last.get("D520_LONG", False)) else ("SHORT ⚠️" if bool(d520_last.get("D520_SHORT", False)) else "Nötr")
        k1.metric("Durum", state_520)
        k2.metric("SMA5", fmt_num(d520_last.get("D520_SMA5", np.nan)))
        k3.metric("SMA20", fmt_num(d520_last.get("D520_SMA20", np.nan)))
        k4.metric("Son Sinyal", "AL" if bool(d520_last.get("D520_BUY_SIG", False)) else ("SAT" if bool(d520_last.get("D520_SELL_SIG", False)) else "YOK"))
        fig_520 = build_system_overlay_chart(
            d520_df.tail(220),
            "5 / 20 Sistem",
            [("D520_SMA5", "SMA 5"), ("D520_SMA20", "SMA 20")],
            [("D520_BUY_SIG", "AL", "triangle-up"), ("D520_SELL_SIG", "SAT", "triangle-down")],
        )
        st.plotly_chart(fig_520, use_container_width=True)

    with rd_tab:
        st.subheader("Richard Dennis / Turtle benzeri breakout sistemi")
        render_page_education_expander([("richard_dennis","Richard Dennis / Turtle"),("rd_long_entry","LONG Giriş"),("rd_short_entry","SHORT Giriş"),("rd_exit_filter","Çıkış Filtresi"),("rd_upper20","20G Üst Seviye"),("rd_lower20","20G Alt Seviye"),("donchian","Donchian Kanalları")])
        rd_df = richard_dennis_system(df.copy())
        rd_last = rd_df.iloc[-1]
        r1, r2, r3, r4 = st.columns(4)
        state_rd = "LONG ✅" if bool(rd_last.get("RD_LONG_ENTRY", False)) else ("SHORT ⚠️" if bool(rd_last.get("RD_SHORT_ENTRY", False)) else "İzle")
        r1.metric("Durum", state_rd)
        r2.metric("20G Üst", fmt_num(rd_last.get("RD_UPPER20", np.nan)))
        r3.metric("20G Alt", fmt_num(rd_last.get("RD_LOWER20", np.nan)))
        r4.metric("Çıkış Filtresi", "Aktif" if bool(rd_last.get("RD_LONG_EXIT", False) or rd_last.get("RD_SHORT_EXIT", False)) else "Pasif")
        fig_rd = build_system_overlay_chart(
            rd_df.tail(220),
            "Richard Dennis / Turtle",
            [("RD_UPPER20", "Üst 20"), ("RD_LOWER20", "Alt 20"), ("RD_EXIT_UPPER10", "Çıkış Üst 10"), ("RD_EXIT_LOWER10", "Çıkış Alt 10")],
            [("RD_LONG_ENTRY", "Long Entry", "triangle-up"), ("RD_SHORT_ENTRY", "Short Entry", "triangle-down"), ("RD_LONG_EXIT", "Long Exit", "x"), ("RD_SHORT_EXIT", "Short Exit", "x")],
        )
        st.plotly_chart(fig_rd, use_container_width=True)


with tab_financials:
    st.header("📘 Bilanço Analizi")
    render_page_education_expander([("financials_page","Bilanço Analizi Nasıl Okunur?"),("roe","ROE"),("revenue_growth","Net Satış Büyümesi"),("ebitda","FAVÖK"),("ebitda_margin","FAVÖK Marjı"),("debt_equity","Borç / Özsermaye"),("current_ratio","Cari Oran"),("net_margin","Net Kâr Marjı"),("fcf","Serbest Nakit Akışı"),("pe","F/K"),("pb","PD/DD"),("net_debt_ebitda","Net Borç / FAVÖK"),("altman_z","Altman Z"),("piotroski_f","Piotroski F"),("dcf","DCF"),("sector_relative","Sektöre Göre Pahalı / Ucuz")])
    st.caption("Seçilen hissenin çeyreklik veya senelik son 4 bilanço dönemini gösterir. Çeyreklik modda kıyas türü olarak bir önceki dönem (legacy), bir önceki çeyrek veya geçen yılın aynı çeyreği seçilebilir; senelik modda bir önceki yıl baz alınır. Daha iyi yönde değişim yeşil, kötü yönde değişim kırmızı görünür.")

    fin_symbol_options = [naked_ticker(x) for x in universe] if universe else [naked_ticker(ticker)]
    default_fin_symbol = naked_ticker(ticker) if naked_ticker(ticker) in fin_symbol_options else fin_symbol_options[0]

    fc1, fc2, fc3, fc4 = st.columns([2, 1.2, 1.6, 1])
    with fc1:
        fin_symbol_raw = st.selectbox(
            "Hisse Seç",
            options=fin_symbol_options,
            index=fin_symbol_options.index(default_fin_symbol),
            key="financials_symbol_raw",
        )
    with fc2:
        fin_mode = st.radio(
            "Dönem Türü",
            options=["quarterly", "annual"],
            format_func=lambda x: "Çeyreklik" if x == "quarterly" else "Senelik",
            key="financials_mode",
            horizontal=True,
        )
    with fc3:
        fin_quarter_compare_mode = st.radio(
            "Çeyrek Kıyas Türü",
            options=["prev_period_legacy", "prev_quarter", "same_quarter_prev_year"],
            format_func=lambda x: (
                "Bir Önceki Dönem (Legacy)"
                if x == "prev_period_legacy"
                else ("Önceki Çeyrek" if x == "prev_quarter" else "Geçen Yıl Aynı Çeyrek")
            ),
            key="financials_quarter_compare_mode",
            horizontal=True,
            disabled=(fin_mode != "quarterly"),
        )
    with fc4:
        run_financials = st.button("📘 Bilanço Analizini Getir", key="run_financials_snapshot", use_container_width=True)

    if run_financials:
        st.session_state.financial_snapshot_result = fetch_financial_snapshot_analysis(
            fin_symbol_raw,
            selected_market=market,
            statement_mode=fin_mode,
            quarterly_compare_mode=fin_quarter_compare_mode,
        )

    fin_result = st.session_state.get("financial_snapshot_result")
    if fin_result:
        if fin_result.get("error"):
            st.warning(fin_result["error"])
        else:
            st.markdown(fin_result.get("table_html", ""), unsafe_allow_html=True)

            summary = fin_result.get("summary", {})
            st.divider()
            st.subheader("Son Bilançoya Göre Tahmini Adil Değerler")
            current_mc = summary.get("current_market_cap", np.nan)
            fair_mc = summary.get("fair_market_cap", np.nan)
            fair_share_price = summary.get("fair_share_price", np.nan)
            dcf_mc = summary.get("dcf_fair_market_cap", np.nan)
            dcf_share = summary.get("dcf_fair_share_price", np.nan)
            upside = summary.get("upside_pct", np.nan)
            latest_shares = summary.get("latest_shares", np.nan)

            fm1, fm2, fm3, fm4 = st.columns(4)
            fm1.metric("Son Dönem", summary.get("latest_period", "N/A"))
            fm2.metric("Mevcut Piyasa Değeri", _money_fmt(current_mc) if np.isfinite(current_mc) else "N/A")
            fm3.metric("Tahmini Adil Net Piyasa Değeri", _money_fmt(fair_mc) if np.isfinite(fair_mc) else "N/A")
            fm4.metric("Adil Hisse Fiyatı", f"{fair_share_price:.2f}" if np.isfinite(fair_share_price) else "N/A")

            fm5, fm6, fm7, fm8 = st.columns(4)
            fm5.metric("Potansiyel Fark", f"{upside:+.2f}%" if np.isfinite(upside) else "N/A")
            fm6.metric("Son Hisse Sayısı", _money_fmt(latest_shares) if np.isfinite(latest_shares) else "N/A")
            fm7.metric("DCF Adil Piyasa Değeri", _money_fmt(dcf_mc) if np.isfinite(dcf_mc) else "N/A")
            fm8.metric("DCF Adil Hisse Fiyatı", f"{dcf_share:.2f}" if np.isfinite(dcf_share) else "N/A")

            fm9, fm10, fm11 = st.columns(3)
            fm9.metric("Hedef F/K", f"{summary.get('target_pe', np.nan):.2f}" if np.isfinite(summary.get('target_pe', np.nan)) else "N/A")
            fm10.metric("Hedef PD/DD", f"{summary.get('target_pb', np.nan):.2f}" if np.isfinite(summary.get('target_pb', np.nan)) else "N/A")
            fm11.metric("Hedef EV/FAVÖK", f"{summary.get('target_ev_ebitda', np.nan):.2f}" if np.isfinite(summary.get('target_ev_ebitda', np.nan)) else "N/A")

            fm12, fm13, fm14, fm15 = st.columns(4)
            fm12.metric("Altman Z-Skoru", f"{summary.get('altman_z', np.nan):.2f}" if np.isfinite(summary.get('altman_z', np.nan)) else "N/A")
            fm13.metric("Altman Durumu", summary.get("altman_state", "N/A"))
            piot_val = summary.get('piotroski_f', np.nan)
            fm14.metric("Piotroski F-Skoru", f"{int(round(piot_val))}/9" if np.isfinite(piot_val) else "N/A")
            fm15.metric("Piotroski Durumu", summary.get("piotroski_state", "N/A"))

            with st.expander("Tahmini piyasa değeri hesap yöntemi", expanded=False):
                st.write("Heuristik adil değer; son finansal tabloya göre F/K, PD/DD ve EV/FAVÖK tabanlı birleşik tahmindir. Buna ek olarak eğitim amaçlı basit bir DCF yaklaşımı da ayrıca hesaplanır.")
                st.write("Altman Z-Skoru bilanço dayanıklılığı ve finansal stres riskini, Piotroski F-Skoru ise 0-9 arası temel kaliteyi özetler.")
                breakdown = summary.get("fair_breakdown", [])
                if breakdown:
                    bd_df = pd.DataFrame(breakdown)
                    bd_df["value"] = bd_df["value"].apply(lambda x: _money_fmt(x) if np.isfinite(x) else "N/A")
                    st.dataframe(bd_df, use_container_width=True, height=180)

                dcf_breakdown = summary.get("dcf_breakdown", {})
                if dcf_breakdown:
                    st.write("DCF Varsayımları:")
                    st.dataframe(
                        pd.DataFrame(
                            {
                                "Varsayım": ["Son FCF", "İskonto Oranı", "Büyüme Oranı", "Terminal Büyüme"],
                                "Değer": [
                                    _money_fmt(dcf_breakdown.get("latest_fcf", np.nan)) if np.isfinite(dcf_breakdown.get("latest_fcf", np.nan)) else "N/A",
                                    f"%{dcf_breakdown.get('discount_rate', np.nan) * 100:.2f}" if np.isfinite(dcf_breakdown.get("discount_rate", np.nan)) else "N/A",
                                    f"%{dcf_breakdown.get('growth_rate', np.nan) * 100:.2f}" if np.isfinite(dcf_breakdown.get("growth_rate", np.nan)) else "N/A",
                                    f"%{dcf_breakdown.get('terminal_growth', np.nan) * 100:.2f}" if np.isfinite(dcf_breakdown.get("terminal_growth", np.nan)) else "N/A",
                                ],
                            }
                        ),
                        use_container_width=True,
                        height=170,
                    )

with tab_calendar:
    st.header("🗓️ Ekonomik Takvim + Makro Risk Paneli")
    st.caption("Ülkeler bazlı önemli makro verileri getirir. İstediğin veri bloğunu sadece ilgili çalıştır tuşuna basınca çağırır; böylece uygulama her hisse seçiminde gereksiz yüklenmez.")


    st.subheader("0) Şirket Takvimi")
    corporate_symbol_options = [naked_ticker(x) for x in universe] if universe else [naked_ticker(ticker)]
    default_corporate_symbol = naked_ticker(ticker) if naked_ticker(ticker) in corporate_symbol_options else corporate_symbol_options[0]

    cc1, cc2 = st.columns([3, 1])
    with cc1:
        corporate_symbol_raw = st.selectbox(
            "Hisse Seç",
            options=corporate_symbol_options,
            index=corporate_symbol_options.index(default_corporate_symbol),
            key="corporate_symbol_raw_calendar",
        )
    with cc2:
        run_corporate_dates = st.button("Şirket Takvimini Getir", key="run_corporate_dates", use_container_width=True)

    if run_corporate_dates:
        st.session_state.corporate_dates_result = fetch_next_corporate_dates(corporate_symbol_raw, market)

    corporate_dates_result = st.session_state.get("corporate_dates_result")
    if corporate_dates_result:
        def _fmt_dt_local(dt):
            if dt is None or pd.isna(dt):
                return "Açıklanmadı / Veri Yok"
            try:
                return pd.to_datetime(dt).strftime("%Y-%m-%d")
            except Exception:
                return str(dt)

        next_earnings = corporate_dates_result.get("next_earnings_date")
        next_dividend = corporate_dates_result.get("next_dividend_date")
        last_dividend = corporate_dates_result.get("last_dividend_date")

        cdm1, cdm2, cdm3 = st.columns(3)
        cdm1.metric("Sonraki Bilanço Tarihi", _fmt_dt_local(next_earnings))
        cdm2.metric("Sonraki Temettü Tarihi", _fmt_dt_local(next_dividend))
        cdm3.metric("Son Bilinen Temettü Tarihi", _fmt_dt_local(last_dividend))

        source_parts = []
        if corporate_dates_result.get("earnings_source"):
            source_parts.append(f"Bilanço kaynağı: {corporate_dates_result.get('earnings_source')}")
        if corporate_dates_result.get("dividend_source"):
            source_parts.append(f"Temettü kaynağı: {corporate_dates_result.get('dividend_source')}")
        if source_parts:
            st.caption(" | ".join(source_parts))
        else:
            st.caption("Veri sağlayıcı ilgili tarihleri yayınlamadıysa alanlar boş görünebilir.")

    st.divider()

    country_options = ["united states", "euro area", "united kingdom", "turkey", "china", "japan", "germany", "france", "canada", "australia", "india", "brazil"]

    st.subheader("1) Ekonomik Takvim (FMP)")
    ec1, ec2, ec3, ec4 = st.columns(4)
    with ec1:
        selected_countries = st.multiselect(
            "Ülkeler",
            options=country_options,
            default=["turkey"] if market == "BIST" else ["united states"],
            key="econ_calendar_countries",
        )
    with ec2:
        importance_label = st.selectbox(
            "Önem Seviyesi",
            options=["Sadece Yüksek", "Orta + Yüksek", "Hepsi"],
            index=0,
            key="econ_calendar_importance_label",
        )
        importance_map = {"Sadece Yüksek": "3", "Orta + Yüksek": "2,3", "Hepsi": "1,2,3"}
        importance_value = importance_map[importance_label]
    with ec3:
        days_forward = st.slider("Kaç Gün İleri", min_value=1, max_value=30, value=14, step=1, key="econ_calendar_days_forward")
    with ec4:
        days_back = st.slider("Kaç Gün Geri", min_value=0, max_value=7, value=1, step=1, key="econ_calendar_days_back")

    econ_api_key_input = st.text_input(
        "FMP API Key",
        value="",
        type="password",
        key="econ_calendar_api_key_input",
        help="Boş bırakırsan secrets içindeki FMP_API_KEY kullanılır.",
    )

    run_econ_calendar = st.button("🗓️ Ekonomik Takvimi Getir", key="run_econ_calendar", use_container_width=True)

    if run_econ_calendar:
        st.session_state.econ_calendar_result = fetch_economic_calendar(
            tuple(selected_countries),
            importance=importance_value,
            api_key_override=econ_api_key_input,
            days_back=int(days_back),
            days_forward=int(days_forward),
        )

    econ_result = st.session_state.get("econ_calendar_result")
    if econ_result:
        if econ_result.get("error"):
            st.warning(econ_result["error"])
        else:
            econ_df = econ_result.get("df", pd.DataFrame()).copy()
            if econ_df.empty:
                st.info("Seçilen filtrelerle ekonomik takvim verisi bulunamadı.")
            else:
                st.success(f"Toplam {len(econ_df)} ekonomik etkinlik bulundu. Kaynak: {econ_result.get('source', 'N/A')}")

                m1, m2, m3, m4 = st.columns(4)
                high_count = int((econ_df["Importance"] == 3).sum()) if "Importance" in econ_df.columns else 0
                country_count = int(econ_df["Country"].nunique()) if "Country" in econ_df.columns else 0
                upcoming_count = int((econ_df["Date"] >= pd.Timestamp(datetime.datetime.utcnow())).sum()) if "Date" in econ_df.columns else 0
                released_count = max(len(econ_df) - upcoming_count, 0)

                m1.metric("Yüksek Önemli Veri", str(high_count))
                m2.metric("Ülke Sayısı", str(country_count))
                m3.metric("Yaklaşan Veri", str(upcoming_count))
                m4.metric("Açıklanmış Veri", str(released_count))

                st.subheader("Öne Çıkan Etkinlikler")
                top_cards = econ_df.head(12).copy()
                for _, row in top_cards.iterrows():
                    dt_str = row["Date"].strftime("%Y-%m-%d %H:%M") if pd.notna(row["Date"]) else "N/A"
                    info_html = make_hover_question_html(row.get("Event", ""), row.get("Category", ""), row.get("Country", ""))
                    st.markdown(
                        f"""
                        <div style="padding:10px 12px; border:1px solid #ddd; border-radius:10px; margin-bottom:8px;">
                            <div style="font-weight:700;">
                                {row.get("Country", "N/A")} — {row.get("Event", "N/A")} {info_html}
                            </div>
                            <div style="font-size:13px; margin-top:4px;">
                                Tarih: {dt_str} | Önem: {row.get("ImportanceLabel", "N/A")} | Kategori: {row.get("Category", "N/A")}
                            </div>
                            <div style="font-size:13px; margin-top:4px;">
                                Actual: {row.get("Actual", "N/A")} | Forecast: {row.get("Forecast", "N/A")} | Previous: {row.get("Previous", "N/A")}
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                gc1, gc2 = st.columns(2)
                with gc1:
                    by_country = econ_df.groupby("Country").size().reset_index(name="Event Count")
                    fig_country = go.Figure()
                    fig_country.add_trace(go.Bar(x=by_country["Country"], y=by_country["Event Count"], name="Event Count"))
                    fig_country.update_layout(height=320, title="Ülkeye Göre Veri Sayısı", xaxis_title="Ülke", yaxis_title="Adet")
                    st.plotly_chart(fig_country, use_container_width=True)

                with gc2:
                    by_cat = econ_df.groupby("Category").size().reset_index(name="Event Count").sort_values("Event Count", ascending=False).head(12)
                    fig_cat = go.Figure()
                    fig_cat.add_trace(go.Bar(x=by_cat["Category"], y=by_cat["Event Count"], name="Event Count"))
                    fig_cat.update_layout(height=320, title="Kategoriye Göre Veri Sayısı", xaxis_title="Kategori", yaxis_title="Adet")
                    st.plotly_chart(fig_cat, use_container_width=True)

                st.subheader("Takvim Tablosu")
                show_df = econ_df[["Date", "Country", "Category", "Event", "Actual", "Forecast", "Previous", "ImportanceLabel"]].copy()
                st.dataframe(show_df, use_container_width=True, height=320)

                with st.expander("Eğitim Notları — Hangi veri hangi sektörleri etkiler?", expanded=False):
                    edu_df = econ_df[["Country", "Event", "Category", "Education"]].drop_duplicates().reset_index(drop=True)
                    st.dataframe(edu_df, use_container_width=True, height=280)

    st.divider()
    st.subheader("2) BIST100 NH-NL İndikatörü (Elder)")
    st.caption("Elder yaklaşımına yakın şekilde BIST100 içindeki 52 haftalık yeni zirve ve yeni dip yapan hisseleri sayar; NH-NL ve 10 günlük EMA'sını üretir.")
    if st.button("NH-NL İndikatörünü Çalıştır", key="run_nhnl_indicator", use_container_width=True):
        st.session_state.nhnl_result = fetch_bist100_nhnl_indicator()

    nhnl_result = st.session_state.get("nhnl_result")
    if nhnl_result:
        if nhnl_result.get("error"):
            st.warning(nhnl_result["error"])
        else:
            nhnl_df = nhnl_result.get("df", pd.DataFrame()).copy()
            if not nhnl_df.empty:
                n1, n2, n3, n4 = st.columns(4)
                n1.metric("Yeni Zirve", f"{int(nhnl_df['New_Highs'].iloc[-1])}")
                n2.metric("Yeni Dip", f"{int(nhnl_df['New_Lows'].iloc[-1])}")
                n3.metric("NH-NL", f"{float(nhnl_df['NH_NL'].iloc[-1]):.0f}")
                n4.metric("NH-NL %", f"{float(nhnl_df['NH_NL_%'].iloc[-1]):+.2f}%")

                fig_nhnl = go.Figure()
                fig_nhnl.add_trace(go.Bar(x=nhnl_df.index, y=nhnl_df["NH_NL"], name="NH-NL"))
                fig_nhnl.add_trace(go.Scatter(x=nhnl_df.index, y=nhnl_df["NH_NL_EMA10"], name="EMA10", line=dict(width=2)))
                fig_nhnl.add_trace(go.Scatter(x=nhnl_df.index, y=nhnl_df["Zero_Line"], name="Zero", line=dict(width=1, dash="dot")))
                fig_nhnl.update_layout(height=360, title="BIST100 NH-NL (Elder yaklaşımı)", xaxis_title="Tarih", yaxis_title="Değer")
                st.plotly_chart(fig_nhnl, use_container_width=True)

    st.divider()
    st.subheader("3) Küresel VIX Endeksi")
    if st.button("VIX Grafiğini Çalıştır", key="run_vix_panel", use_container_width=True):
        st.session_state.vix_result = fetch_vix_series()

    vix_result = st.session_state.get("vix_result")
    if vix_result:
        if vix_result.get("error"):
            st.warning(vix_result["error"])
        else:
            vix_df = vix_result.get("df", pd.DataFrame()).copy()
            if not vix_df.empty:
                v1, v2 = st.columns(2)
                v1.metric("Son VIX", f"{float(vix_df['Close'].iloc[-1]):.2f}")
                v2.metric("VIX vs EMA13", f"{float(vix_df['Close'].iloc[-1] - vix_df['EMA13'].iloc[-1]):+.2f}")

                fig_vix = go.Figure()
                fig_vix.add_trace(go.Scatter(x=vix_df.index, y=vix_df["Close"], name="VIX", line=dict(width=2)))
                fig_vix.add_trace(go.Scatter(x=vix_df.index, y=vix_df["EMA13"], name="EMA13", line=dict(width=2, dash="dash")))
                fig_vix.update_layout(height=360, title="Küresel VIX Endeksi", xaxis_title="Tarih", yaxis_title="VIX")
                st.plotly_chart(fig_vix, use_container_width=True)

    st.divider()
    st.subheader("4) BIST100 Kuvvet Endeksi + 13 Günlük EMA (Elder)")
    if st.button("BIST100 Elder Panelini Çalıştır", key="run_bist_force_panel", use_container_width=True):
        st.session_state.bist_force_result = fetch_bist100_force_index_panel()

    bist_force_result = st.session_state.get("bist_force_result")
    if bist_force_result:
        if bist_force_result.get("error"):
            st.warning(bist_force_result["error"])
        else:
            elder_df = bist_force_result.get("df", pd.DataFrame()).copy()
            if not elder_df.empty:
                e1, e2, e3 = st.columns(3)
                e1.metric("XU100 Son Kapanış", f"{float(elder_df['Close'].iloc[-1]):.2f}")
                e2.metric("EMA13 Farkı", f"{float(elder_df['Close'].iloc[-1] - elder_df['EMA13'].iloc[-1]):+.2f}")
                e3.metric("Force Index", f"{float(elder_df['ForceIndex'].iloc[-1]):.0f}")

                fig_price = go.Figure()
                fig_price.add_trace(go.Candlestick(
                    x=elder_df.index,
                    open=elder_df["Open"],
                    high=elder_df["High"],
                    low=elder_df["Low"],
                    close=elder_df["Close"],
                    name="XU100"
                ))
                fig_price.add_trace(go.Scatter(x=elder_df.index, y=elder_df["EMA13"], name="EMA13", line=dict(width=2)))
                fig_price.update_layout(height=420, title="BIST100 Fiyat + 13 Günlük EMA", xaxis_rangeslider_visible=False)
                st.plotly_chart(fig_price, use_container_width=True)

                fig_force = go.Figure()
                fig_force.add_trace(go.Bar(x=elder_df.index, y=elder_df["ForceIndex"], name="Force Index"))
                fig_force.add_trace(go.Scatter(x=elder_df.index, y=elder_df["ForceIndexEMA13"], name="FI EMA13", line=dict(width=2)))
                fig_force.update_layout(height=320, title="BIST100 Kuvvet Endeksi (Force Index) + EMA13", xaxis_title="Tarih", yaxis_title="Force Index")
                st.plotly_chart(fig_force, use_container_width=True)
