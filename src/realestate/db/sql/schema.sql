-- 공개 화면용 집계표 (realestate_public). scripts/publish.py 가 쓰기 계정으로 실행한다.
-- MVP 1: R-ONE 지수만. 이후 MVP 에서 agg_region_month, region_score 등을 추가한다 (PLAN.md 5-3).

CREATE TABLE IF NOT EXISTS rone_region (
    region_key   VARCHAR(100) NOT NULL PRIMARY KEY,
    name         VARCHAR(40)  NOT NULL,
    sido         VARCHAR(20)  NOT NULL,
    parent_key   VARCHAR(100) NULL,
    depth        TINYINT      NOT NULL,
    region_type  VARCHAR(10)  NOT NULL,
    display_name VARCHAR(60)  NOT NULL,
    has_real     BOOLEAN      NOT NULL,
    first_ym     DATE         NOT NULL,
    last_ym      DATE         NOT NULL
);

CREATE TABLE IF NOT EXISTS rone_index (
    source       VARCHAR(10)  NOT NULL,
    region_key   VARCHAR(100) NOT NULL,
    ym           DATE         NOT NULL,
    index_value  DOUBLE       NOT NULL,
    mom_pct      DOUBLE       NULL,
    PRIMARY KEY (source, region_key, ym)
);

CREATE TABLE IF NOT EXISTS publish_meta (
    table_name   VARCHAR(60)  NOT NULL PRIMARY KEY,
    row_count    INT          NOT NULL,
    data_last_ym DATE         NULL,
    published_at DATETIME     NOT NULL
);

-- MVP 3: 실거래 집계·지역 점수·백테스트 성적표 (scripts/build_public.py → publish.py --tables ...). 개별 거래 행 없음.
CREATE TABLE IF NOT EXISTS region_name (
    sgg_cd       CHAR(5)      NOT NULL PRIMARY KEY,
    sido_cd      CHAR(2)      NOT NULL,
    sido_nm      VARCHAR(20)  NOT NULL,
    sgg_nm       VARCHAR(40)  NOT NULL,
    zone         VARCHAR(10)  NOT NULL
);

CREATE TABLE IF NOT EXISTS agg_region_month (
    sgg_cd       CHAR(5)      NOT NULL,
    ym           DATE         NOT NULL,
    trade_cnt    INT          NOT NULL,
    cancel_cnt   INT          NOT NULL,
    median_price DOUBLE       NULL,
    p25_ppm2     DOUBLE       NULL,
    median_ppm2  DOUBLE       NULL,
    p75_ppm2     DOUBLE       NULL,
    PRIMARY KEY (sgg_cd, ym)
);

CREATE TABLE IF NOT EXISTS region_score (
    asof_ym          DATE         NOT NULL,
    sgg_cd           CHAR(5)      NOT NULL,
    grade            VARCHAR(20)  NOT NULL,
    score            DOUBLE       NULL,
    p_a              DOUBLE       NULL,
    p_b              DOUBLE       NULL,
    p_c              DOUBLE       NULL,
    p_d              DOUBLE       NULL,
    drawdown_pct     DOUBLE       NULL,
    underval_pct     DOUBLE       NULL,
    trend_3m_pct     DOUBLE       NULL,
    volume_ratio     DOUBLE       NULL,
    trade_12m        INT          NOT NULL,
    cancel_ratio     DOUBLE       NULL,
    r1               BOOLEAN      NOT NULL,
    r2               BOOLEAN      NOT NULL,
    r3               BOOLEAN      NOT NULL,
    rule_flags       VARCHAR(20)  NOT NULL,
    sido_level_index BOOLEAN      NOT NULL,
    model_version    VARCHAR(80)  NOT NULL,
    PRIMARY KEY (asof_ym, sgg_cd)
);

CREATE TABLE IF NOT EXISTS bt_ic (
    period       VARCHAR(10)  NOT NULL,
    horizon_m    TINYINT      NOT NULL,
    zone         VARCHAR(10)  NOT NULL,
    value_key    VARCHAR(10)  NOT NULL,
    value_name   VARCHAR(40)  NOT NULL,
    ic_mean      DOUBLE       NOT NULL,
    ci_lo        DOUBLE       NULL,
    ci_hi        DOUBLE       NULL,
    pos_share    DOUBLE       NOT NULL,
    months       INT          NOT NULL,
    PRIMARY KEY (period, horizon_m, zone, value_key)
);

CREATE TABLE IF NOT EXISTS bt_grade (
    period       VARCHAR(10)  NOT NULL,
    horizon_m    TINYINT      NOT NULL,
    grade        VARCHAR(20)  NOT NULL,
    n_all        INT          NOT NULL,
    n            INT          NOT NULL,
    x_mean       DOUBLE       NULL,
    x_median     DOUBLE       NULL,
    hit          DOUBLE       NULL,
    mdd_mean     DOUBLE       NULL,
    PRIMARY KEY (period, horizon_m, grade)
);

CREATE TABLE IF NOT EXISTS bt_baseline (
    horizon_m    TINYINT      NOT NULL,
    name         VARCHAR(40)  NOT NULL,
    spread       DOUBLE       NOT NULL,
    lo           DOUBLE       NULL,
    hi           DOUBLE       NULL,
    PRIMARY KEY (horizon_m, name)
);

CREATE TABLE IF NOT EXISTS bt_meta (
    k            VARCHAR(40)  NOT NULL PRIMARY KEY,
    v            VARCHAR(200) NOT NULL
);

CREATE TABLE IF NOT EXISTS complex_price (
    row_id       INT          NOT NULL PRIMARY KEY,
    sgg_cd       CHAR(5)      NOT NULL,
    umd_nm       VARCHAR(20)  NOT NULL,
    apt_nm       VARCHAR(80)  NOT NULL,
    build_year   SMALLINT     NULL,
    trade_cnt    INT          NOT NULL,
    median_ppm2  DOUBLE       NULL,
    model_ppm2   DOUBLE       NULL,
    gap_pct      DOUBLE       NULL,
    window_from  DATE         NOT NULL,
    window_to    DATE         NOT NULL,
    KEY idx_sgg (sgg_cd)
);
