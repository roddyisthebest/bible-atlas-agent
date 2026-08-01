# Chat API 스펙 (iOS 연동)

## 1. 개요

`POST /invoke`, `POST /stream` 은 이제 **멀티턴 채팅**을 지원한다.
클라이언트가 대화 상태(`summary`, `messages`)를 들고 다니고, 서버는 매 응답에
"다음 요청에 그대로 실어보낼 상태" 를 완성해서 돌려준다. 클라는 응답의
`summary` / `messages` 로 로컬 상태를 **통째로 덮어쓰기만** 하면 된다.

- 서버는 stateless. 어떤 DB/세션도 두지 않는다.
- 클라는 요약이 언제 일어났는지 알 필요 없다.

## 2. 요청 스키마

```json
POST /stream        (또는 /invoke)
Content-Type: application/json
X-API-Key: <API_KEY>

{
  "query": "그럼 거기서 다음엔 어디로 갔어?",
  "summary": null,
  "messages": [
    { "role": "user",      "content": "출애굽 경로 알려줘" },
    { "role": "assistant", "content": "라암셋에서 출발해…" }
  ]
}
```

| 필드       | 타입           | 필수 | 설명                                             |
| ---------- | -------------- | ---- | ------------------------------------------------ |
| `query`    | string (min 1) | ✅   | 이번 턴의 새 질문                                |
| `summary`  | string \| null | ❌   | 이전에 서버가 준 요약본. 없으면 `null` 또는 생략 |
| `messages` | array          | ❌   | 이전 대화 이력. 없으면 `[]` 또는 생략            |

`messages[]` 원소:

- `role`: `"user"` \| `"assistant"` (system 은 클라가 못 넣음)
- `content`: 문자열

## 3. 응답 스키마

### `/invoke` — JSON 응답

```json
{
  "answer": "…",
  "place_id_map": { "베들레헴": ["a112427"] },
  "recommended_questions": [],
  "summary": null,
  "messages": [
    { "role": "user", "content": "..." },
    { "role": "assistant", "content": "..." }
  ]
}
```

### `/stream` — SSE `done` 이벤트 payload

동일한 필드 세트를 `done` 이벤트의 `data` JSON 으로 내려준다.
`node`, `error` 이벤트 스펙은 기존과 동일.

| 필드                    | 타입                     | 설명                                                               |
| ----------------------- | ------------------------ | ------------------------------------------------------------------ |
| `answer`                | string                   | 사용자에게 보여줄 최종 답변                                        |
| `place_id_map`          | dict\<string, string[]\> | 지명 → place_id 목록. 첫 글자 `a`=ancient, `m`=modern              |
| `recommended_questions` | string[]                 | `non_bible_reject` 경로에서만 채워짐 (최대 3)                      |
| `summary`               | string \| null           | **다음 요청에 실어보낼 요약본**. 아직 없으면 `null`                |
| `messages`              | array                    | **다음 요청에 실어보낼 이력**. 요약 발생 시 `[]` 로 리셋될 수 있음 |

## 4. 클라이언트 규약 (딱 3가지)

1. **첫 요청**: `summary` 생략(또는 `null`), `messages: []`
2. **매 요청**: 로컬에 저장된 `summary` 와 `messages` 를 있는 그대로 실어보냄
3. **매 응답**: 응답의 `summary` / `messages` 로 **로컬 상태를 통째로 덮어씀**
   - 절대 append 하지 않는다. 이번 턴의 `(user, assistant)` 는 서버가 이미
     `messages` 에 포함시켜 돌려준다.

새 대화 시작: 로컬 `summary = null`, `messages = []` 로 초기화. 별도 API 없음.

## 5. 서버 동작 (참고용)

```
[요청 수신]
  ctx = []
  if summary:  ctx += [SystemMessage("이전 대화 요약:\n" + summary)]
  ctx += messages 를 LangChain 메시지로 변환
  ctx += [HumanMessage(query)]

[graph.stream(ctx) → answer, place_id_map, recommended_questions]

[상태 업데이트]
  new_messages = messages + [(user, query), (assistant, answer)]
  new_summary  = summary
  if len(new_messages) > THRESHOLD:
      new_summary  = summarize(summary, new_messages)   # bounded
      new_messages = []                                  # 리셋

[응답]
  { answer, place_id_map, recommended_questions,
    summary: new_summary, messages: new_messages }
```

### 상수

- `THRESHOLD = 10` (messages 원소 개수 = user+assistant 합계)
- `SUMMARY_MAX_TOKENS = 400` (요약 LLM `max_tokens`)
- `SUMMARY_MAX_CHARS = 1600` (최종 hard truncate)

### 요약 폭주 방지 (3중)

1. 요약 LLM 호출 시 `max_tokens=400` 강제
2. 프롬프트에 "400 토큰(한글 약 800자) 이내" 명시
3. 응답 문자열을 `[:1600]` 로 hard truncate

재요약(`summary + new_messages → new_summary`)도 같은 상한을 지키므로
요약이 무한히 커지지 않는다.

## 6. 라우터/에이전트 영향

- **router** 는 `query` 만 본다. 매 턴 독립 라우팅. (사용자가 도중에 주제
  전환해도 라우터가 새로 판단 → 의도한 동작.)
- **place_agent** 는 `messages` 전체(있으면 summary 시스템 메시지 포함) 를
  보고 지시대명사·문맥을 해석한다.
- **bible_general_agent / non_bible_reject** 는 기존과 동일하게 `query` 만 사용.

## 7. 예시 상태 전이 (THRESHOLD=10 기준)

| 턴  | 요청에 실은 것                           | 응답으로 저장할 것                       |
| --- | ---------------------------------------- | ---------------------------------------- |
| 1   | `{query, messages:[]}`                   | `{summary:null, messages:[u1,a1]}`       |
| 2   | `{query, messages:[u1,a1]}`              | `{summary:null, messages:[u1,a1,u2,a2]}` |
| …   | …                                        | …                                        |
| 5   | `{query, messages:[u1..a4]}` (10개 초과) | `{summary:"…요약…", messages:[]}`        |
| 6   | `{query, summary, messages:[]}`          | `{summary, messages:[u6,a6]}`            |
| …   | 다시 쌓임                                | 임계값 도달 시 재요약 후 리셋            |

## 8. 하위호환

- `summary` / `messages` 는 요청·응답 모두 **선택 필드**.
- 기존 iOS 클라가 이 필드들을 안 보내도 서버는 단일턴으로 동작하며,
  응답의 새 필드를 무시해도 무방하다.
