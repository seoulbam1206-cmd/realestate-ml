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
