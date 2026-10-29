-- TiDB Cloud SQL Editor 에서 "관리자(root)"로 한 번만 실행 (내가 직접).
-- 1) <접두사> 를 클러스터의 사용자 접두사로 바꾸세요 (예: 'abc123'). 콘솔 Connect 화면의 user 앞부분입니다.
-- 2) <비밀번호1>, <비밀번호2> 는 콘솔에서 직접 입력하고, 이 파일에는 저장하지 마세요.

CREATE DATABASE IF NOT EXISTS realestate_public;

-- 쓰기 계정: 로컬 .env 에만 저장, scripts/publish.py 전용
CREATE USER '<접두사>.writer'@'%' IDENTIFIED BY '<비밀번호1>';
GRANT SELECT, INSERT, UPDATE, DELETE, CREATE, DROP, INDEX, ALTER ON realestate_public.* TO '<접두사>.writer'@'%';

-- 읽기 전용 계정: Streamlit Secrets 에만 저장
CREATE USER '<접두사>.web_ro'@'%' IDENTIFIED BY '<비밀번호2>';
GRANT SELECT ON realestate_public.* TO '<접두사>.web_ro'@'%';

-- 확인
SHOW GRANTS FOR '<접두사>.web_ro'@'%';
