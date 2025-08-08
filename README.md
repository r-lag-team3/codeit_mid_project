# codeit_mid_project
3팀의 중급 프로젝트: 문서 기반 RAG(Retrieval-Augmented Generation) 구현 및 성능 평가

## 제출(보고서, 협업일지)
- 보고서PDF: https://github.com/r-lag-team3/codeit_mid_project/raw/main/report/3TEAM_REPORT.pdf
- 협업일지:  https://github.com/r-lag-team3/co-log/blob/883c5c39f8f5ba98feb50a9ee550f19cfef3db78/README.md

## 프로젝트 개요
    - 다양한 PDF/HWP 문서로부터 정보를 추출, 전처리, 임베딩, 검색, LLM 기반 질의응답까지의 전체 RAG 파이프라인 구현
    - 리트리버 성능 평가 및 실험 자동화

## 기능 개요
    - 문서 전처리 및 청크 생성
    - FAISS 기반 임베딩 및 벡터 검색
    - RAG 체인 및 LLM 연동
    - GPT 기반 평가 및 정답 스팬 비교
    - 실험 결과 자동 저장 및 관리

## version
- 1.0: 기본 기능 구현
    - 청크 Document 생성
    - 리트리버를 통한 문서 검색
    - 검색된 문서를 바탕으로 Open AI LLM 모델에 프롬프트 입력 및 답변 생성

- 2.0: 성능 평가 추가
    - GPT를 이용한 평가 기능 추가
    - 정답 스팬을 생성 기능 추가
    - 정답 스팬의 참고 문서와 리트리버가 찾은 참고 문서를 비교하여 리트리버의 성능 평가

- 3.0: 성능 개선
    - 전처리 작업 추가 진행
    - 문서에 없는 내용으로 답변생성 방지 프롬프트 추가
    - 배치사이즈로 분리하여 임베딩(임베딩 모델 토큰 초과 방지)
    - 이전 대화내용 기억 기능 추가
    - 제목 키워드 기반 필터 기능 추가
    - Embedding, generator모델 Hugging Face와 Open AI 모두 사용

## 디렉토리 구조
```md
codeit_mid_project/
│
├── data/                     # 원본 문서 및 전처리된 데이터 (git: x)
│   ├── raw_data/             # 원본 PDF, HWP 파일 등 (git: x)
│   ├── pdf_data/             # HWP 파일을 PDF로 변환(실제로 사용할 데이터) (git: x)
│   └── processed/            # 전처리된 텍스트, chunk 등 (git: o)
│       └── span_list.json    # 정답 스팬(span_list.json) 위치
│
├── embeddings/               # 벡터 임베딩 저장소 (FAISS 등)
│
├── experiment/               # 실험 결과 폴더
│   ├──test_experiment1/
│   ├──test_experiment2/
│   └──test_experiment3/      # 실험 내용(결과, 사용한 LLM, 하이퍼파라미터 설정 등)
│            ...
│
│
├── rag_chain/                # RAG 파이프라인 구성 모듈
│
├── utils/                    # 각종 도구 스크립트 
│
├── test.ipynb                # 실험용 주피터 노트북
│
├── README.md
└── .gitignore
```

## 파일 설명
    - chunking: 전처리된 텍스트, 청크, 정답 스팬(span_list.json) 등
    - faiss: FAISS 임베딩 및 검색
    - rag_chain: RAG 체인 구현
    - API 키와 같은 민감 정보는 별도 환경변수 또는 config 파일로 관리

## 사용 방법
    - 데이터 준비: data/pdf_data/에 PDF 파일 저장
    - 전처리 및 청크 생성: utils/preprocess.py 등 활용
    - 임베딩 생성: embeddings/faiss.py 실행
    - RAG 파이프라인 실행: rag_chain/ 내 모듈 사용
    - 평가 및 실험: utils/evaluate.py, experiment/ 폴더 참고

## 실험 및 평가
    - 실험 결과(experiment/) 제공
    - 평가 코드로 리트리버 성능, LLM 응답 품질 등 비교 가능
