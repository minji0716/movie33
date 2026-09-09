import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error

# ------------------------------------------------------------------
# 기본 설정
# ------------------------------------------------------------------
st.set_page_config(page_title="영화 흥행 예측기", layout="wide")
st.title("🎬 영화 흥행 예측기")
st.caption("KOBIS 박스오피스 일별 데이터와 영화 정보를 결합해 총 관객 수를 다중 회귀로 예측합니다.")

DAILY_URL = "https://raw.githubusercontent.com/greatsong/modudata/main/data/kobis_daily.csv"
MOVIES_URL = "https://raw.githubusercontent.com/greatsong/modudata/main/data/kobis_movies.csv"

FEATURE_OPTIONS = {
    "first_scrn": "개봉(첫 관측일) 스크린수",
    "first_show": "개봉(첫 관측일) 상영횟수",
    "peak": "성수기 개봉 여부 (1월·7월·12월=1)",
    "first_week_audi": "개봉 첫 주 관객수",
    "days_in_top10": "박스오피스 10위권 유지일수",
    "avg_screen_daily": "박스오피스 기간 평균 스크린수",
    "avg_show_daily": "박스오피스 기간 평균 상영횟수",
    "avg_daily_audi": "박스오피스 기간 평균 일일 관객수",
    "max_daily_audi": "박스오피스 기간 최고 일일 관객수",
    "days_appeared_daily": "일별 데이터에 등장한 일수(TOP10 등장일)",
}
DEFAULT_CHECKED = {
    "first_scrn", "first_show", "first_week_audi", "days_in_top10", "avg_daily_audi",
}

LOW_PRED_THRESHOLD = 1000   # 이보다 예측치가 작으면 바닥에 붙여 표시
Y_AXIS_FLOOR = 10           # 로그축 최솟값
STICK_VALUE = 30            # 바닥에 붙일 때 사용할 표시 값


# ------------------------------------------------------------------
# 데이터 로드 & 전처리
# ------------------------------------------------------------------
@st.cache_data
def load_data():
    daily = pd.read_csv(DAILY_URL, encoding="utf-8")
    movies = pd.read_csv(MOVIES_URL, encoding="utf-8")
    return daily, movies


with st.spinner("데이터를 불러오는 중입니다..."):
    daily, movies = load_data()

# 날짜 파싱 (기준 기간 계산용)
daily["날짜_dt"] = pd.to_datetime(daily["날짜"].astype(str), format="%Y%m%d")
period_start = daily["날짜_dt"].min().strftime("%Y-%m-%d")
period_end = daily["날짜_dt"].max().strftime("%Y-%m-%d")

# 일별 데이터를 영화코드 기준으로 집계 (박스오피스 표 → 영화별 요약 특징)
agg = (
    daily.groupby("영화코드")
    .agg(
        avg_screen_daily=("스크린수", "mean"),
        avg_show_daily=("상영횟수", "mean"),
        avg_daily_audi=("일관객", "mean"),
        max_daily_audi=("일관객", "max"),
        days_appeared_daily=("날짜", "count"),
    )
    .reset_index()
    .rename(columns={"영화코드": "movieCd"})
)

# 영화 정보 표와 병합 (movieCd 기준, 영화 정보 표의 모든 영화 사용)
movies = movies.copy()
movies["movieCd"] = movies["movieCd"].astype(agg["movieCd"].dtype)
merged = movies.merge(agg, on="movieCd", how="left")

# 박스오피스 집계값이 없는 영화(일별 TOP10 데이터에 한 번도 없었던 경우)는 0으로 채움
for col in ["avg_screen_daily", "avg_show_daily", "avg_daily_audi", "max_daily_audi", "days_appeared_daily"]:
    merged[col] = merged[col].fillna(0)

# 숫자형으로 강제 변환 (결측/이상값 방어)
numeric_cols = list(FEATURE_OPTIONS.keys()) + ["total_audi"]
for col in numeric_cols:
    merged[col] = pd.to_numeric(merged[col], errors="coerce").fillna(0)

# ------------------------------------------------------------------
# 영화코드 순 정렬 후, 10편마다 앞의 3편을 시험용으로 분리
# ------------------------------------------------------------------
merged = merged.sort_values("movieCd").reset_index(drop=True)
positions = np.arange(len(merged))
is_test = (positions % 10) < 3
merged["split"] = np.where(is_test, "test", "train")

train_df = merged[merged["split"] == "train"]
test_df = merged[merged["split"] == "test"]

# ------------------------------------------------------------------
# 변수 선택 (체크박스)
# ------------------------------------------------------------------
st.subheader("① 예측에 사용할 변수 선택")
cols = st.columns(2)
selected_features = []
for i, (key, label) in enumerate(FEATURE_OPTIONS.items()):
    col = cols[i % 2]
    checked = col.checkbox(label, value=(key in DEFAULT_CHECKED), key=f"feat_{key}")
    if checked:
        selected_features.append(key)

if not selected_features:
    st.warning("최소 한 개 이상의 변수를 선택해주세요.")
    st.stop()

# ------------------------------------------------------------------
# 모델 학습 & 평가
# ------------------------------------------------------------------
X_train = train_df[selected_features]
y_train = train_df["total_audi"]
X_test = test_df[selected_features]
y_test = test_df["total_audi"]

model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
rmse = float(np.sqrt(np.mean((y_test.values - y_pred) ** 2)))

st.subheader("② 학습 및 평가 요약")
c1, c2, c3 = st.columns(3)
c1.metric("학습에 쓴 영화 편수", f"{len(train_df):,} 편")
c2.metric("점수를 잰(시험용) 영화 편수", f"{len(test_df):,} 편")
c3.metric("기준 기간 (박스오피스 데이터)", f"{period_start} ~ {period_end}")

c4, c5 = st.columns(2)
c4.metric("예측 점수 (R², 시험용 영화 기준)", f"{r2:.3f}")
c5.metric("평균 오차 (MAE / RMSE)", f"{mae:,.0f}명 / {rmse:,.0f}명")

with st.expander("회귀 계수 자세히 보기"):
    coef_df = pd.DataFrame({
        "변수": [FEATURE_OPTIONS[f] for f in selected_features],
        "회귀 계수": model.coef_,
    })
    st.dataframe(coef_df, use_container_width=True)
    st.write(f"절편(intercept): {model.intercept_:,.1f}")

# ------------------------------------------------------------------
# 산점도 (Plotly, 로그-로그 축, y=x 기준선, 예측 1,000명 미만은 바닥에 표시)
# ------------------------------------------------------------------
st.subheader("③ 실제 총 관객 수 vs 예측 총 관객 수")

actual = y_test.values.astype(float)
pred = y_pred.astype(float)

# 로그축을 위해 0 이하 값은 최소값(1)으로 보정
actual_plot = np.where(actual <= 0, 1, actual)

low_pred_mask = pred < LOW_PRED_THRESHOLD
num_low_pred = int(low_pred_mask.sum())

pred_plot = np.where(low_pred_mask, STICK_VALUE, pred)
pred_plot = np.where(pred_plot <= 0, STICK_VALUE, pred_plot)

hover_movie = test_df["movieNm"].values

fig = go.Figure()

# 정상 예측 포인트
fig.add_trace(go.Scatter(
    x=actual_plot[~low_pred_mask],
    y=pred_plot[~low_pred_mask],
    mode="markers",
    name="예측 (정상 범위)",
    marker=dict(size=9, color="#2563eb", opacity=0.75, line=dict(width=0.5, color="white")),
    text=hover_movie[~low_pred_mask],
    customdata=pred[~low_pred_mask],
    hovertemplate="영화명: %{text}<br>실제 총 관객수: %{x:,.0f}명<br>예측 총 관객수: %{customdata:,.0f}명<extra></extra>",
))

# 예측이 1,000명 미만인 포인트 (바닥에 붙여 표시)
if num_low_pred > 0:
    fig.add_trace(go.Scatter(
        x=actual_plot[low_pred_mask],
        y=pred_plot[low_pred_mask],
        mode="markers",
        name=f"예측 1,000명 미만 ({num_low_pred}편, 바닥에 표시)",
        marker=dict(size=9, symbol="triangle-down", color="#dc2626", opacity=0.85, line=dict(width=0.5, color="white")),
        text=hover_movie[low_pred_mask],
        customdata=pred[low_pred_mask],
        hovertemplate="영화명: %{text}<br>실제 총 관객수: %{x:,.0f}명<br>예측 총 관객수(실제값): %{customdata:,.0f}명<extra></extra>",
    ))

# y = x 기준선
axis_max = max(actual_plot.max(), pred_plot.max()) * 1.5
axis_min = Y_AXIS_FLOOR
fig.add_trace(go.Scatter(
    x=[axis_min, axis_max],
    y=[axis_min, axis_max],
    mode="lines",
    name="예측=실제 기준선",
    line=dict(color="gray", dash="dash"),
))

fig.update_layout(
    xaxis=dict(title="실제 총 관객 수 (명, 로그)", type="log", range=[np.log10(axis_min), np.log10(axis_max)]),
    yaxis=dict(title="예측 총 관객 수 (명, 로그)", type="log", range=[np.log10(axis_min), np.log10(axis_max)]),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    height=600,
    margin=dict(t=60),
)

st.plotly_chart(fig, use_container_width=True)

st.info(f"예측이 {LOW_PRED_THRESHOLD:,}명보다 작게 나온 영화는 **{num_low_pred}편**이며, 그래프 하단에 붙여 표시했습니다 (▽ 표시).")

with st.expander("시험용 영화별 실제·예측 상세 보기"):
    detail_df = test_df[["movieCd", "movieNm", "total_audi"]].copy()
    detail_df["예측 총 관객수"] = np.round(y_pred).astype(int)
    detail_df = detail_df.rename(columns={"movieCd": "영화코드", "movieNm": "영화명", "total_audi": "실제 총 관객수"})
    st.dataframe(detail_df.reset_index(drop=True), use_container_width=True)
