from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components

from src.backtest import HISTORICAL_REGIMES, compute_metrics, compute_regime_metrics, run_backtest
from src.classifier import load_model
from src.data_loader import load_price_data
from src.detection import compute_features
from src.ledger import HashChainLedger
from src.portfolio import SimulatedBroker

ROOT = Path(__file__).resolve().parent
INK = "#07111f"
PANEL = "#0e2036"
PANEL_ALT = "#0a192b"
BORDER = "#213b59"
NAVY = "#146ef5"
TEAL = "#1ce6d4"
GOLD = "#ffb703"
RED = "#ff5b65"
MUTED = "#7f98b5"

st.set_page_config(page_title="Aura Ledger | TECHBEEZ", page_icon="A", layout="wide", initial_sidebar_state="expanded")

# Remove Streamlit's default light canvas when this file is used as a local
# launcher for the browser-native replay.
st.markdown(
    """<style>
    [data-testid="stAppViewContainer"], [data-testid="stMain"], .stApp {
        background: #06111f !important;
    }
    [data-testid="stHeader"] { background: #06111f !important; }
    .block-container { max-width: 1500px !important; padding: .55rem .7rem 0 !important; }
    iframe { border: 0 !important; background: #06111f !important; }
    </style>""",
    unsafe_allow_html=True,
)

# The browser-native replay is the primary judging experience.  It runs entirely
# in the client, so its Start/Reset/scenario controls do not trigger Streamlit
# reruns or page flashing.  Keep this wrapper for users who still launch the
# project with `streamlit run dashboard.py`.
standalone_demo = ROOT / "Aura_Ledger_Interactive_Demo.html"
if standalone_demo.exists():
    components.html(standalone_demo.read_text(encoding="utf-8"), height=1140, scrolling=True)
    st.stop()

st.markdown(
    f"""<style>
    .stApp {{ background: {INK}; color: #e8f3ff; font-family: Arial, sans-serif; }}
    .block-container {{ max-width: 1680px; padding: 0.85rem 1rem 2rem; }}
    [data-testid="stHeader"] {{ background: rgba(7,17,31,.88); }}
    section[data-testid="stSidebar"] {{ background: #091828; border-right: 1px solid {BORDER}; }}
    section[data-testid="stSidebar"] > div {{ padding-top: 1.25rem; }}
    h1,h2,h3,p,label {{ color: #e8f3ff !important; }}
    h2 {{ font-family: Consolas, monospace; font-size: 1rem !important; letter-spacing: .12em; text-transform: uppercase; }}
    [data-testid="stMetric"] {{ background: linear-gradient(135deg, #112741, #0b1d31); border: 1px solid {BORDER}; border-radius: 7px; padding: .7rem .85rem; box-shadow: inset 0 1px 0 rgba(255,255,255,.035); }}
    [data-testid="stMetricLabel"] {{ color: {MUTED} !important; font-family: Consolas, monospace; font-size: .66rem; text-transform: uppercase; letter-spacing: .09em; }}
    [data-testid="stMetricValue"] {{ color: {TEAL} !important; font-family: Consolas, monospace; font-size: 1.45rem; }}
    [data-testid="stMetricDelta"] {{ font-family: Consolas, monospace; font-size: .72rem; }}
    .stButton > button {{ border: 1px solid {GOLD}; background: rgba(255,183,3,.08); color: {GOLD}; border-radius: 4px; font-family: Consolas, monospace; font-size: .72rem; font-weight: 700; letter-spacing: .08em; text-transform: uppercase; }}
    .stButton > button:hover {{ border-color: {TEAL}; color: {TEAL}; background: rgba(28,230,212,.10); }}
    .stButton > button[kind="primary"] {{ background: {NAVY}; color: #fff; border-color: #3786ff; }}
    .stSlider [data-baseweb="slider"] div[role="slider"] {{ background: {GOLD}; }}
    [data-baseweb="tab-list"] {{ gap: .25rem; border-bottom-color: {BORDER}; }}
    [data-baseweb="tab"] {{ color: {MUTED}; font-family: Consolas, monospace; font-size: .72rem; letter-spacing: .06em; }}
    [data-baseweb="tab"][aria-selected="true"] {{ color: {TEAL}; border-bottom-color: {TEAL}; }}
    .brandbar {{ display:flex; align-items:center; gap:1rem; padding: .15rem 0 .8rem; border-bottom:1px solid {BORDER}; margin-bottom:.72rem; }}
    .brand {{ color:{GOLD}; font:700 1.38rem Consolas, monospace; letter-spacing:.12em; }}
    .monitor {{ color:{TEAL}; font:600 .68rem Consolas, monospace; letter-spacing:.1em; }}
    .monitor:before {{ content:'●'; padding-right:.38rem; }}
    .meta {{ color:{MUTED}; font:.64rem Consolas, monospace; letter-spacing:.05em; }}
    .session {{ margin-left:auto; color:{GOLD}; font:.66rem Consolas, monospace; letter-spacing:.07em; }}
    .panel-title {{ color:{GOLD}; font:700 .67rem Consolas, monospace; letter-spacing:.15em; text-transform:uppercase; border-bottom:1px solid {BORDER}; padding:.3rem .2rem .55rem; margin:0 0 .45rem; }}
    .system-box {{ background:linear-gradient(135deg, rgba(28,230,212,.12), rgba(20,110,245,.06)); border:1px solid rgba(28,230,212,.32); border-radius:5px; padding:.75rem; color:{TEAL}; font:.68rem Consolas, monospace; line-height:1.65; }}
    .system-box.alert {{ color:{GOLD}; border-color:rgba(255,183,3,.45); background:rgba(255,183,3,.07); }}
    .portfolio-card {{ background:linear-gradient(135deg, #102843, #0b1b2f); border:1px solid {BORDER}; border-radius:6px; padding:.8rem; margin-bottom:.55rem; }}
    .portfolio-label {{ color:{MUTED}; font:.6rem Consolas, monospace; letter-spacing:.09em; text-transform:uppercase; }}
    .portfolio-number {{ color:#fff; font:700 1.5rem Consolas, monospace; margin:.2rem 0; }}
    .row {{ display:flex; justify-content:space-between; align-items:center; border-top:1px solid rgba(127,152,181,.14); padding:.48rem 0; color:#dbeafe; font:.7rem Consolas, monospace; }}
    .asset-dot {{ color:{TEAL}; margin-right:.35rem; }} .asset-gold {{ color:{GOLD}; margin-right:.35rem; }} .asset-cash {{ color:{MUTED}; margin-right:.35rem; }}
    .value {{ color:{TEAL}; }} .warn {{ color:{GOLD}; }}
    .footer {{ color:{MUTED}; font:.66rem Consolas, monospace; text-align:center; border-top:1px solid {BORDER}; padding-top:1rem; margin-top:1.5rem; }}
    [data-testid="stDataFrame"] {{ border:1px solid {BORDER}; border-radius:6px; }}
    </style>""",
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner="Booting Aura Ledger replay engine...")
def build_simulation(data_version: int, model_version: int):
    prices = load_price_data(str(ROOT / "data" / "prices.csv"))
    model = load_model(str(ROOT / "models" / "crash_classifier.pkl"))
    ledger = HashChainLedger()
    results = run_backtest(prices, compute_features(prices), model, SimulatedBroker(), ledger)
    return results, ledger.to_dataframe(), ledger.verify_chain(), compute_metrics(results), compute_regime_metrics(results)


def chart_layout(title: str, height: int) -> dict:
    return {
        "title": {"text": title.upper(), "font": {"family": "JetBrains Mono", "size": 11, "color": GOLD}, "x": 0.02},
        "height": height,
        "paper_bgcolor": PANEL,
        "plot_bgcolor": PANEL,
        "font": {"family": "JetBrains Mono", "color": MUTED, "size": 10},
        "margin": {"l": 42, "r": 18, "t": 44, "b": 32},
        "legend": {"orientation": "h", "y": 1.18, "x": 1, "xanchor": "right", "font": {"size": 9}},
        "xaxis": {"gridcolor": "rgba(127,152,181,.12)", "zeroline": False, "showline": False},
        "yaxis": {"gridcolor": "rgba(127,152,181,.12)", "zeroline": False, "showline": False},
    }


def risk_gauge(probability: float) -> go.Figure:
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=probability * 100,
            number={"suffix": "", "font": {"family": "JetBrains Mono", "size": 36, "color": TEAL}},
            title={"text": "CRISIS SCORE", "font": {"family": "JetBrains Mono", "size": 11, "color": GOLD}},
            gauge={
                "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": MUTED, "tickfont": {"size": 8}},
                "bar": {"color": GOLD, "thickness": 0.22},
                "bgcolor": PANEL,
                "borderwidth": 0,
                "steps": [
                    {"range": [0, 30], "color": "rgba(28,230,212,.22)"},
                    {"range": [30, 65], "color": "rgba(255,183,3,.18)"},
                    {"range": [65, 100], "color": "rgba(255,91,101,.20)"},
                ],
                "threshold": {"line": {"color": RED, "width": 4}, "thickness": 0.78, "value": 65},
            },
        )
    )
    fig.update_layout(paper_bgcolor=PANEL, height=260, margin=dict(l=12, r=12, t=40, b=0))
    return fig


def portfolio_figure(frame: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=frame["date"], y=frame["strategy_value"], mode="lines", name="Aura Ledger", line=dict(color=GOLD, width=2.5), fill="tozeroy", fillcolor="rgba(255,183,3,.07)"))
    fig.add_trace(go.Scatter(x=frame["date"], y=frame["buyhold_value"], mode="lines", name="Unhedged", line=dict(color="#7087a5", width=1.4, dash="dot")))
    fig.update_layout(**chart_layout("Portfolio value - real time", 360))
    fig.update_yaxes(tickprefix="$", tickformat=",")
    return fig


def signal_figure(frame: pd.DataFrame, threshold: float) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=frame["date"], y=frame["crash_probability"], mode="lines", name="Crisis probability", line=dict(color=TEAL, width=2.4), fill="tozeroy", fillcolor="rgba(28,230,212,.10)"))
    fig.add_hline(y=threshold, line_dash="dash", line_color=GOLD, annotation_text="ACTION THRESHOLD", annotation_font=dict(color=GOLD, size=9))
    fig.update_layout(**chart_layout("Detection engine - risk overlay", 310))
    fig.update_yaxes(range=[0, 1], tickformat=".0%")
    return fig


def allocation_figure(frame: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    for column, label, color in [
        ("allocation_equity_pct", "Equity", NAVY),
        ("allocation_gold_pct", "Gold hedge", GOLD),
        ("allocation_cash_pct", "Cash", "#75869a"),
    ]:
        fig.add_trace(go.Scatter(x=frame["date"], y=frame[column], stackgroup="one", mode="lines", name=label, line=dict(width=0.8, color=color)))
    fig.update_layout(**chart_layout("Allocation rotation", 310))
    fig.update_yaxes(range=[0, 1], tickformat=".0%")
    return fig


def drawdown_figure(frame: pd.DataFrame) -> go.Figure:
    """Show the protection story directly: hedged loss versus unhedged loss."""
    strategy_drawdown = frame["strategy_value"].div(frame["strategy_value"].cummax()).sub(1)
    unhedged_drawdown = frame["buyhold_value"].div(frame["buyhold_value"].cummax()).sub(1)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=frame["date"], y=strategy_drawdown, mode="lines", name="Hedged / Aura", line=dict(color=TEAL, width=2.2), fill="tozeroy", fillcolor="rgba(28,230,212,.08)"))
    fig.add_trace(go.Scatter(x=frame["date"], y=unhedged_drawdown, mode="lines", name="Unhedged / equity", line=dict(color=RED, width=1.8, dash="dot")))
    fig.update_layout(**chart_layout("Hedged vs unhedged drawdown", 310))
    fig.update_yaxes(tickformat=".0%")
    return fig


def data_source_info() -> dict:
    source_path = ROOT / "data" / "data_source.json"
    if not source_path.exists():
        return {"kind": "synthetic_demo", "coverage": "2018-2022"}
    try:
        return json.loads(source_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {"kind": "local_data", "coverage": "local CSV"}


def smooth_replay_console(results: pd.DataFrame) -> None:
    """Client-side canvas replay: animation never triggers a Streamlit rerun."""
    records = [
        {
            "d": row.date.strftime("%Y-%m-%d"), "s": round(float(row.strategy_value), 2),
            "u": round(float(row.buyhold_value), 2), "p": round(float(row.crash_probability), 4),
            "e": round(float(row.allocation_equity_pct), 4), "g": round(float(row.allocation_gold_pct), 4),
            "c": round(float(row.allocation_cash_pct), 4), "a": row.action_taken,
        }
        for row in results.itertuples(index=False)
    ]
    scenario_windows = {}
    for name, (start, end) in HISTORICAL_REGIMES.items():
        start_index = int(results.index[results["date"] >= pd.Timestamp(start)][0])
        end_index = int(results.index[results["date"] >= pd.Timestamp(end)][0])
        scenario_windows[name] = {"start": start_index, "end": min(len(results) - 1, end_index + 120)}
    page = """<!doctype html><html><head><style>
    *{box-sizing:border-box} body{margin:0;background:#07111f;color:#e8f3ff;font-family:Arial,sans-serif}
    .shell{border:1px solid #213b59;border-radius:8px;overflow:hidden;background:linear-gradient(145deg,#0b1b2f,#07111f);padding:14px}
    .top{display:flex;align-items:center;gap:13px;border-bottom:1px solid #213b59;padding-bottom:11px;margin-bottom:12px}.brand{color:#ffb703;font:700 19px Consolas,monospace;letter-spacing:3px}.live{color:#1ce6d4;font:700 10px Consolas,monospace;letter-spacing:1px}.live:before{content:'●';padding-right:5px}.date{margin-left:auto;color:#ffb703;font:10px Consolas,monospace}
    .grid{display:grid;grid-template-columns:repeat(6,1fr);gap:8px}.card{background:#102843;border:1px solid #213b59;border-radius:5px;padding:9px}.label{color:#7f98b5;font:9px Consolas,monospace;letter-spacing:1px}.value{color:#1ce6d4;font:700 20px Consolas,monospace;margin-top:5px;white-space:nowrap}.sub{font:10px Consolas,monospace;margin-top:4px;color:#9ab1c9}
    .body{display:grid;grid-template-columns:1fr 2.4fr;gap:10px;margin-top:10px}.panel{background:#0e2036;border:1px solid #213b59;border-radius:6px;padding:10px}.title{color:#ffb703;font:700 10px Consolas,monospace;letter-spacing:1.4px;padding-bottom:8px;border-bottom:1px solid #213b59;margin-bottom:8px}.gauge{height:155px;display:flex;align-items:center;justify-content:center}.gauge svg{width:190px;height:140px}.score{font:700 32px Consolas,monospace;fill:#1ce6d4}.signal{font:10px Consolas,monospace;line-height:1.8;color:#b9cbe0}.phase{margin:6px 0;padding:7px;border:1px solid #1ce6d4;border-radius:4px;color:#1ce6d4;font:700 10px Consolas,monospace;letter-spacing:1px;text-align:center}.asset{margin:7px 0}.assetline{display:flex;justify-content:space-between;color:#b9cbe0;font:9px Consolas,monospace;margin-bottom:3px}.track{height:8px;background:#081624;border-radius:5px;overflow:hidden;border:1px solid #213b59}.fill{height:100%;width:0;transition:width .35s cubic-bezier(.22,1,.36,1)}.eq{background:#146ef5}.au{background:#ffb703}.ca{background:#75869a}.chart{height:250px;width:100%;display:block}.bottom{display:grid;grid-template-columns:1.2fr 1fr;gap:10px;margin-top:10px}.controls{display:flex;flex-wrap:wrap;gap:7px;margin-top:10px;align-items:center}.btn{background:#0b1b2f;border:1px solid #ffb703;color:#ffb703;border-radius:4px;padding:8px 11px;font:700 10px Consolas,monospace;letter-spacing:1px;cursor:pointer}.btn.primary{background:#146ef5;color:#fff;border-color:#3786ff}.btn.active{border-color:#1ce6d4;color:#1ce6d4}.range{accent-color:#ffb703;width:160px}.legend{font:10px Consolas,monospace;color:#7f98b5}.dot{display:inline-block;width:8px;height:3px;margin-right:4px;vertical-align:middle}.hedge{background:#ffb703}.unhedged{background:#ff5b65}.risk{background:#1ce6d4}.note{color:#7f98b5;font:9px Consolas,monospace;margin-top:9px}
    @media(max-width:900px){.grid{grid-template-columns:repeat(3,1fr)}.body,.bottom{grid-template-columns:1fr}.date{display:none}}
    </style></head><body><div class="shell"><div class="top"><div class="brand">AURA LEDGER</div><div class="live">SMOOTH REPLAY ENGINE</div><div class="date" id="date">REPLAY DATE</div></div>
    <div class="grid"><div class="card"><div class="label">HEDGED VALUE</div><div class="value" id="strategy">--</div><div class="sub" id="strategyDelta"></div></div><div class="card"><div class="label">UNHEDGED VALUE</div><div class="value" id="unhedged">--</div><div class="sub">Equity buy-and-hold</div></div><div class="card"><div class="label">CRISIS SCORE</div><div class="value" id="risk">--</div><div class="sub" id="status"></div></div><div class="card"><div class="label">EQUITY ALLOCATION</div><div class="value" id="equity">--</div><div class="sub">Risk asset exposure</div></div><div class="card"><div class="label">GOLD HEDGE</div><div class="value" id="gold">--</div><div class="sub">Defensive allocation</div></div><div class="card"><div class="label">LATEST DECISION</div><div class="value" id="action" style="font-size:15px">--</div><div class="sub">Hash-chain logged</div></div></div>
    <div class="controls"><button class="btn primary" id="play">START SMOOTH REPLAY</button><button class="btn" id="prev">PREVIOUS DAY</button><button class="btn" id="next">NEXT DAY</button><button class="btn" data-event="2008 Global Financial Crisis">2008</button><button class="btn" data-event="2018 Q4 selloff">2018</button><button class="btn active" data-event="2020 COVID crash">2020</button><button class="btn" data-event="2022 inflation drawdown">2022</button><span class="legend">SPEED</span><input class="range" id="speed" type="range" min="1" max="40" value="10"><input class="range" id="timeline" type="range" min="0" max="1" value="0"></div>
    <div class="body"><div class="panel"><div class="title">VSE ENGINE / CRISIS PROBABILITY</div><div class="gauge"><svg viewBox="0 0 200 130"><path d="M25 105 A78 78 0 0 1 175 105" fill="none" stroke="#213b59" stroke-width="15"/><path id="arc" d="M25 105 A78 78 0 0 1 175 105" fill="none" stroke="#1ce6d4" stroke-width="15" stroke-linecap="round"/><text id="score" x="100" y="98" text-anchor="middle" class="score">0</text><text x="100" y="116" text-anchor="middle" fill="#7f98b5" style="font:9px Consolas">CRISIS SCORE</text></svg></div><div class="phase" id="phase">NORMAL MARKET / EQUITY MODE</div><div class="signal" id="signal">SYSTEM NOMINAL</div><div class="asset"><div class="assetline"><span>EQUITY EXPOSURE</span><span id="eqPct">--</span></div><div class="track"><div class="fill eq" id="eqBar"></div></div></div><div class="asset"><div class="assetline"><span>GOLD HEDGE</span><span id="goldPct">--</span></div><div class="track"><div class="fill au" id="goldBar"></div></div></div><div class="asset"><div class="assetline"><span>CASH RESERVE</span><span id="cashPct">--</span></div><div class="track"><div class="fill ca" id="cashBar"></div></div></div></div><div class="panel"><div class="title">PORTFOLIO VALUE / HEDGED VS UNHEDGED <span class="legend"><span class="dot hedge"></span>HEDGED <span class="dot unhedged"></span>UNHEDGED</span></div><canvas class="chart" id="valueChart"></canvas></div></div>
    <div class="bottom"><div class="panel"><div class="title">DETECTION SIGNAL <span class="legend"><span class="dot risk"></span>CRISIS PROBABILITY</span></div><canvas class="chart" id="riskChart"></canvas></div><div class="panel"><div class="title">ALLOCATION ROTATION <span class="legend">BLUE EQUITY / GOLD HEDGE / GREY CASH</span></div><canvas class="chart" id="allocChart"></canvas></div></div>
    <div class="note">All animation is rendered in your browser from local data. No page refresh, no API calls, and no live broker orders.</div></div>
    <script>const DATA=__PAYLOAD__;const EVENTS=__EVENTS__;let start=0,end=DATA.length-1,i=0,playing=false,last=0,credit=0,defensiveSeen=false;const $=id=>document.getElementById(id);const money=n=>'$'+Math.round(n).toLocaleString();const pct=n=>(n*100).toFixed(0)+'%';
    function size(c){const r=c.getBoundingClientRect(),d=devicePixelRatio||1;c.width=r.width*d;c.height=r.height*d;return [c.width,c.height,d]}
    function lineChart(id,series,colors,minMax){const c=$(id);let [w,h,d]=size(c);const x=c.getContext('2d');x.scale(d,d);w/=d;h/=d;x.clearRect(0,0,w,h);const a=DATA.slice(start,i+1),vals=series.flatMap(s=>a.map(s));let lo=minMax?minMax[0]:Math.min(...vals),hi=minMax?minMax[1]:Math.max(...vals);if(hi===lo){hi+=1;lo-=1}const pad=(hi-lo)*.12;lo-=pad;hi+=pad;x.strokeStyle='rgba(127,152,181,.18)';x.lineWidth=1;for(let g=1;g<4;g++){let y=h*g/4;x.beginPath();x.moveTo(0,y);x.lineTo(w,y);x.stroke()}series.forEach((fn,n)=>{x.strokeStyle=colors[n];x.lineWidth=n?1.7:2.5;x.beginPath();a.forEach((row,k)=>{let xx=a.length<2?0:k*w/(a.length-1),yy=h-(fn(row)-lo)/(hi-lo)*h;k?x.lineTo(xx,yy):x.moveTo(xx,yy)});x.stroke()})}
    function allocChart(){const c=$('allocChart'),[w0,h0,d]=size(c),x=c.getContext('2d'),w=w0/d,h=h0/d,a=DATA.slice(start,i+1);x.scale(d,d);x.clearRect(0,0,w,h);if(!a.length)return;[['e','#146ef5'],['g','#ffb703'],['c','#75869a']].forEach(([key,col],layer)=>{x.fillStyle=col;x.beginPath();a.forEach((r,k)=>{let xx=a.length<2?0:k*w/(a.length-1),below=layer===0?0:layer===1?r.e:r.e+r.g,top=below+r[key];k?x.lineTo(xx,h-h*top):x.moveTo(xx,h-h*top)});for(let k=a.length-1;k>=0;k--){let r=a[k],xx=a.length<2?0:k*w/(a.length-1),below=layer===0?0:layer===1?r.e:r.e+r.g;x.lineTo(xx,h-h*below)}x.closePath();x.fill()})}
    function render(){const r=DATA[i],first=DATA[start];if(r.p>=.30)defensiveSeen=true;let phase=r.p>=.65?'CRASH DEFENSE / GOLD + CASH':r.p>=.30?'RISK RISING / SHIFTING CAPITAL':defensiveSeen&&r.e<.72?'RECOVERY / RE-ENTERING EQUITY':'NORMAL MARKET / EQUITY MODE';$('date').textContent='REPLAY DATE : '+r.d;$('strategy').textContent=money(r.s);$('strategyDelta').textContent=((r.s/first.s-1)*100).toFixed(1)+'% since scenario start';$('unhedged').textContent=money(r.u);$('risk').textContent=pct(r.p);$('equity').textContent=pct(r.e);$('gold').textContent=pct(r.g);$('action').textContent=r.a;$('status').textContent=phase;$('phase').textContent=phase;$('score').textContent=Math.round(r.p*100);$('eqPct').textContent=pct(r.e);$('goldPct').textContent=pct(r.g);$('cashPct').textContent=pct(r.c);$('eqBar').style.width=pct(r.e);$('goldBar').style.width=pct(r.g);$('cashBar').style.width=pct(r.c);const arc=$('arc'),len=arc.getTotalLength();arc.style.strokeDasharray=len;arc.style.strokeDashoffset=len*(1-r.p);arc.style.stroke=r.p>=.65?'#ff5b65':r.p>=.30?'#ffb703':'#1ce6d4';$('signal').innerHTML='PHASE : '+phase+'<br>CRISIS PROBABILITY : '+pct(r.p)+'<br>ACTION : '+r.a+'<br>DAY '+(i-start+1)+' OF '+(end-start+1);$('timeline').value=i;lineChart('valueChart',[r=>r.s,r=>r.u],['#ffb703','#ff5b65']);lineChart('riskChart',[r=>r.p],['#1ce6d4'],[0,1]);allocChart()}
    function scenario(name){const event=EVENTS[name];start=Math.max(0,event.start-120);end=event.end;i=Math.min(start+20,end);playing=false;credit=0;defensiveSeen=false;$('timeline').min=start;$('timeline').max=end;$('play').textContent='START SMOOTH REPLAY';document.querySelectorAll('[data-event]').forEach(b=>b.classList.toggle('active',b.dataset.event===name));render()}
    $('play').onclick=()=>{playing=!playing;$('play').textContent=playing?'PAUSE REPLAY':'START SMOOTH REPLAY'};$('prev').onclick=()=>{playing=false;i=Math.max(start,i-1);render()};$('next').onclick=()=>{playing=false;i=Math.min(end,i+1);render()};document.querySelectorAll('[data-event]').forEach(b=>b.onclick=()=>scenario(b.dataset.event));$('timeline').oninput=e=>{playing=false;i=+e.target.value;render()};function animate(t){if(playing){if(!last)last=t;credit+=(t-last)*Number($('speed').value)/1000;let n=Math.floor(credit);if(n){i=Math.min(end,i+n);credit-=n;render();if(i===end){playing=false;$('play').textContent='REPLAY COMPLETE'}}last=t}requestAnimationFrame(animate)}scenario('2020 COVID crash');requestAnimationFrame(animate);</script></body></html>"""
    page = page.replace("__PAYLOAD__", json.dumps(records, separators=(",", ":"))).replace("__EVENTS__", json.dumps(scenario_windows, separators=(",", ":")))
    components.html(page, height=910, scrolling=False)


def init_state() -> None:
    for key, value in {"day_index": 45, "confirmed_actions": set()}.items():
        if key not in st.session_state:
            st.session_state[key] = value


def portfolio_panel(current, previous) -> None:
    change = current.strategy_value / previous.strategy_value - 1 if previous.strategy_value else 0
    st.markdown('<p class="panel-title">Portfolio / protective basket</p>', unsafe_allow_html=True)
    st.markdown(
        f'''<div class="portfolio-card"><div class="portfolio-label">Total portfolio value</div>
        <div class="portfolio-number">${current.strategy_value:,.0f}</div>
        <div class="value">{change:+.2%} session change</div></div>
        <div class="row"><span><span class="asset-dot">●</span>EQUITY INDEX</span><span>{current.allocation_equity_pct:.0%}</span></div>
        <div class="row"><span><span class="asset-gold">◆</span>GOLD HEDGE</span><span>{current.allocation_gold_pct:.0%}</span></div>
        <div class="row"><span><span class="asset-cash">■</span>CASH RESERVE</span><span>{current.allocation_cash_pct:.0%}</span></div>
        <div class="row"><span>UNHEDGED VALUE</span><span>${current.buyhold_value:,.0f}</span></div>''',
        unsafe_allow_html=True,
    )


def live_monitor(results: pd.DataFrame, ledger_df: pd.DataFrame, data_info: dict, regime_df: pd.DataFrame) -> None:
    """Render a stable analysis canvas driven by user-controlled replay steps."""
    total_days = len(results)

    current = results.iloc[st.session_state.day_index]
    previous = results.iloc[max(st.session_state.day_index - 1, 0)]
    probability = float(current.crash_probability)
    threshold = st.session_state.risk_threshold
    mode = st.session_state.execution_mode
    if probability >= 0.65:
        status, status_class = "DEFENSIVE MODE", "alert"
    elif probability >= threshold:
        status, status_class = "ELEVATED RISK", "alert"
    else:
        status, status_class = "SYSTEM NOMINAL", ""
    source_name = "HISTORICAL MULTI-REGIME" if data_info.get("kind") == "historical_market_data" else "SYNTHETIC DEMO DATA"

    st.markdown(f'''<div class="brandbar"><div class="brand">AURA LEDGER</div><div class="monitor">MONITORING</div>
        <div class="meta">TECHBEEZ / {source_name} / CRASH PROTECTION PROTOCOL</div>
        <div class="session">REPLAY DATE : {current.date:%d %b %Y}</div></div>''', unsafe_allow_html=True)
    session_delta = current.strategy_value / previous.strategy_value - 1 if previous.strategy_value else 0
    risk_delta = probability - float(previous.crash_probability)
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("Equity index", f"{current.buyhold_value / 100:,.0f}", f"{session_delta:+.2%}")
    c2.metric("Gold hedge", f"{current.allocation_gold_pct:.0%}", f"{st.session_state.hedge_bias}% bias")
    c3.metric("Crisis score", f"{probability:.0%}", f"{risk_delta:+.1%}")
    c4.metric("Hedged value", f"${current.strategy_value:,.0f}", f"{session_delta:+.2%}")
    c5.metric("Unhedged value", f"${current.buyhold_value:,.0f}", f"{current.buyhold_value / results.iloc[0].buyhold_value - 1:+.1%}")
    c6.metric("Shield status", "ACTIVE" if probability >= threshold else "READY", status)

    visible = results.iloc[: st.session_state.day_index + 1]
    left, centre, right = st.columns([1.05, 2.8, 1.2], gap="small")
    with left:
        st.markdown('<p class="panel-title">VSE engine / probability</p>', unsafe_allow_html=True)
        st.plotly_chart(risk_gauge(probability), use_container_width=True, config={"displayModeBar": False})
        signals = current.signals if isinstance(current.signals, dict) else {}
        vol = signals.get("volatility_z") or 0
        ddv = signals.get("drawdown_velocity_z") or 0
        st.markdown(f'''<div class="system-box {status_class}"><b>{status}</b><br>VOLATILITY Z : {vol:.2f}<br>DRAWDOWN VEL : {ddv:.2f}<br>ACTION : {current.action_taken}<br>MODE : {mode.upper()}</div>''', unsafe_allow_html=True)
    with centre:
        st.plotly_chart(portfolio_figure(visible), use_container_width=True, config={"displayModeBar": False})
    with right:
        portfolio_panel(current, previous)
        protected = current.strategy_value - current.buyhold_value
        st.markdown('<p class="panel-title">Hedge outcome</p>', unsafe_allow_html=True)
        st.metric("Wealth preserved", f"${protected:,.0f}", f"{protected / current.buyhold_value:+.1%} vs unhedged")
        st.caption("Hedged = Aura allocation. Unhedged = equity buy-and-hold baseline.")

    risk_col, allocation_col, drawdown_col = st.columns([1.2, 1, 1.15], gap="small")
    with risk_col:
        st.plotly_chart(signal_figure(visible, threshold), use_container_width=True, config={"displayModeBar": False})
    with allocation_col:
        st.plotly_chart(allocation_figure(visible), use_container_width=True, config={"displayModeBar": False})
    with drawdown_col:
        st.plotly_chart(drawdown_figure(visible), use_container_width=True, config={"displayModeBar": False})

    is_action = current.action_taken != "HOLD"
    requires_confirmation = mode == "Manual" and is_action and st.session_state.day_index not in st.session_state.confirmed_actions
    if requires_confirmation:
        st.warning(f"Manual authorization required: {current.action_taken} proposed on {current.date:%d %b %Y}.")
        if st.button("Authorize proposed rebalance", type="primary"):
            st.session_state.confirmed_actions.add(st.session_state.day_index)
            st.session_state.day_index = min(st.session_state.day_index + 1, total_days - 1)
            st.rerun()

    st.markdown('<p class="panel-title">Tamper-evident decision ledger / latest blocks</p>', unsafe_allow_html=True)
    st.dataframe(ledger_df.tail(st.session_state.day_index + 1), use_container_width=True, hide_index=True, height=255, column_config={"crash_probability": st.column_config.NumberColumn("CRISIS PROBABILITY", format="%.1f")})
    st.markdown('<p class="panel-title">Historical stress-regime evidence / hedged versus unhedged</p>', unsafe_allow_html=True)
    st.dataframe(regime_df, use_container_width=True, hide_index=True, height=190, column_config={
        "regime": "STRESS EVENT", "period": "PERIOD",
        "hedged_return": st.column_config.NumberColumn("HEDGED RETURN", format="%.1%"),
        "unhedged_return": st.column_config.NumberColumn("UNHEDGED RETURN", format="%.1%"),
        "hedged_max_drawdown": st.column_config.NumberColumn("HEDGED MAX DD", format="%.1%"),
        "unhedged_max_drawdown": st.column_config.NumberColumn("UNHEDGED MAX DD", format="%.1%"),
        "drawdown_protection": st.column_config.NumberColumn("DD PROTECTION", format="%.1%"),
    })
    note = "Historical data downloaded once; dashboard operates offline." if data_info.get("kind") == "historical_market_data" else "Synthetic demo scenario - run fetch_data.py before making historical performance claims."
    st.markdown(f'<p class="footer">{note}</p>', unsafe_allow_html=True)


def main() -> None:
    try:
        results, ledger_df, chain_valid, metrics, regime_df = build_simulation(
            (ROOT / "data" / "prices.csv").stat().st_mtime_ns,
            (ROOT / "models" / "crash_classifier.pkl").stat().st_mtime_ns,
        )
    except FileNotFoundError as exc:
        st.error(f"Demo setup incomplete: {exc}. Run python generate_demo_data.py, then python train_model.py.")
        st.stop()
    except Exception as exc:
        st.error(f"Aura Ledger could not load its local model or data. Run python train_model.py. Details: {exc}")
        st.stop()
    init_state()
    data_info = data_source_info()
    with st.sidebar:
        st.markdown("## Control deck")
        st.caption("OFFLINE DEMO / TECHBEEZ / NO AUTO-REFRESH")
        previous_day, next_day = st.columns(2)
        with previous_day:
            if st.button("Previous", use_container_width=True):
                st.session_state.day_index = max(0, st.session_state.day_index - 1)
        with next_day:
            if st.button("Next day", type="primary", use_container_width=True):
                st.session_state.day_index = min(len(results) - 1, st.session_state.day_index + 1)
        future_actions = results.index[(results.index > st.session_state.day_index) & (results["action_taken"] != "HOLD")]
        if st.button("Jump to next decision", use_container_width=True) and len(future_actions):
            st.session_state.day_index = int(future_actions[0])
        st.session_state.timeline_day = int(st.session_state.day_index)
        timeline = st.slider("Replay timeline", 0, len(results) - 1, key="timeline_day", help="Drag to inspect any trading day without auto-play or blinking.")
        if timeline != st.session_state.day_index:
            st.session_state.day_index = timeline
        st.radio("Execution protocol", ["Auto", "Manual"], horizontal=True, key="execution_mode")
        st.markdown("---\n##### Scenario navigation")
        scenario = st.selectbox("Historical stress event", ["Live replay"] + list(HISTORICAL_REGIMES), key="scenario_event")
        if st.button("Load selected event", use_container_width=True) and scenario != "Live replay":
            event_start = pd.Timestamp(HISTORICAL_REGIMES[scenario][0])
            st.session_state.day_index = int(results.index[results["date"] >= event_start][0])
        if st.button("Reset replay", use_container_width=True):
            st.session_state.day_index, st.session_state.confirmed_actions = 45, set()
        st.markdown("---\n##### Sensitivity parameters")
        st.slider("Risk action threshold", 0.15, 0.75, 0.30, 0.05, key="risk_threshold")
        st.slider("Gold hedge bias", 0, 100, 55, 5, key="hedge_bias")
        st.markdown("---\n##### Ledger integrity")
        st.success("HASH CHAIN VERIFIED" if chain_valid else "CHAIN VERIFICATION FAILED")
        st.caption(f"Strategy drawdown: {metrics['strategy']['max_drawdown']:.1%}")
        st.info("DATA: historical" if data_info.get("kind") == "historical_market_data" else "DATA: synthetic demo")
    st.markdown("### Live replay console")
    st.caption("Client-side animation: press Start Smooth Replay to advance charts and metrics without a Streamlit page refresh.")
    smooth_replay_console(results)
    st.markdown("### Detailed audit workspace")
    live_monitor(results, ledger_df, data_info, regime_df)


if __name__ == "__main__":
    main()
