# bible-atlas-agent

LangChain과 LangGraph를 사용한 간단한 데모 프로젝트입니다.

Repository: https://github.com/roddyisthebest/bible-atlas-agent

## 준비

`.env.example`을 복사해 `.env`를 만들고 `OPENAI_API_KEY` 값을 채워주세요.

```bash
cp .env.example .env
```

## 실행

```bash
uv sync
uv run python demo.py
```

## Graph 흐름

```text
START
→ generate_answer
→ END
```
