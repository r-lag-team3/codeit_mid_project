# codeit_mid_project
3팀's 중급 프로젝트

## version
- 1.0: 기본 기능 구현
    - 청크 Document 생성
    - 리트리버를 통한 문서 검색
    - 검색된 문서를 바탕으로 LLM 모델에 프롬프트 입력 및 답변 생성

- 2.0: 성능 평가 추가
    - GPT를 이용한 평가 기능 추가
    - 정답 스팬을 생성 기능 추가
    - 정답 스팬의 참고 문서와 리트리버가 찾은 참고 문서를 비교하여 리트리버의 성능 평가

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
├── scripts/                  # 일회성 스크립트 (데이터 변환, 초기화 등)
│
├── utils/                    # 각종 도구 스크립트 
│
├── test.ipynb                # 실험용 주피터 노트북
│
├── main.ipynb                # 코드 통합(보고서용 주피터 노트북)
│
├── README.md
└── .gitignore
```