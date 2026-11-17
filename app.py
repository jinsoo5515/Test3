import os

import streamlit as st
from dotenv import load_dotenv
from openai import (
    APIConnectionError,
    APITimeoutError,
    AuthenticationError,
    BadRequestError,
    NotFoundError,
    OpenAI,
    PermissionDeniedError,
    RateLimitError,
)

load_dotenv()

LANGUAGES = {"영어": "English", "일본어": "Japanese", "베트남어": "Vietnamese"}
MAX_CHARS = 5000
DEFAULT_MODEL = "gpt-6.1-sol"

SYSTEM_PROMPT = (
    "You are a professional translator. Translate the user's text into {lang}. "
    "Preserve meaning, tone, line breaks, and formatting. "
    "Output only the translated text, with no explanations."
)


class TranslationError(Exception):
    """사용자에게 그대로 보여줄 수 있는 한국어 오류 메시지를 담는다."""


def get_setting(name: str, default: str | None = None) -> str | None:
    """환경변수(.env) 우선, 없으면 st.secrets(배포 환경)에서 읽는다."""
    value = os.getenv(name)
    if value:
        return value
    try:
        return st.secrets.get(name, default)
    except Exception:
        return default


def translate(text: str, target_language: str) -> str:
    api_key = get_setting("OPENAI_API_KEY")
    if not api_key:
        raise TranslationError("API 키가 설정되지 않았습니다. .env 또는 Secrets에 OPENAI_API_KEY를 등록해 주세요.")
    model = get_setting("OPENAI_MODEL", DEFAULT_MODEL) or DEFAULT_MODEL

    client = OpenAI(api_key=api_key, timeout=60)
    params = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT.format(lang=target_language)},
            {"role": "user", "content": text},
        ],
    }

    try:
        try:
            response = client.chat.completions.create(temperature=0.2, **params)
        except BadRequestError as e:
            # 모델이 temperature를 지원하지 않는 경우 해당 파라미터 없이 재시도
            if "temperature" in str(e).lower():
                response = client.chat.completions.create(**params)
            else:
                raise
    except AuthenticationError:
        raise TranslationError("API 키가 올바르지 않습니다. 키를 확인해 주세요.") from None
    except PermissionDeniedError:
        raise TranslationError("이 API 키에는 해당 모델을 사용할 권한이 없습니다.") from None
    except NotFoundError:
        raise TranslationError(f"모델 '{model}'을(를) 찾을 수 없습니다. OPENAI_MODEL 설정을 확인해 주세요.") from None
    except RateLimitError:
        raise TranslationError("요청 한도를 초과했거나 사용 가능한 크레딧이 부족합니다. 잠시 후 다시 시도해 주세요.") from None
    except (APIConnectionError, APITimeoutError):
        raise TranslationError("서버에 연결할 수 없습니다. 네트워크 상태를 확인하고 다시 시도해 주세요.") from None
    except BadRequestError:
        raise TranslationError("요청이 올바르지 않습니다. 입력 내용이나 모델 설정을 확인해 주세요.") from None
    except Exception:
        raise TranslationError("번역 중 알 수 없는 오류가 발생했습니다. 잠시 후 다시 시도해 주세요.") from None

    result = (response.choices[0].message.content or "").strip()
    if not result:
        raise TranslationError("번역 결과가 비어 있습니다. 다시 시도해 주세요.")
    return result


BUTTON_LABELS = {"영어": "영어 (EN)", "일본어": "일본어 (JA)", "베트남어": "베트남어 (VI)"}

st.set_page_config(page_title="번역기", page_icon="🌐", layout="centered")
st.markdown(
    """
    <style>
    footer, #MainMenu {visibility: hidden;}
    .block-container {padding-top: 2.5rem; max-width: 720px;}
    div[data-testid="stHorizontalBlock"] {flex-wrap: wrap; gap: 0.5rem;}
    div[data-testid="stHorizontalBlock"] > div {min-width: 140px;}
    div[data-testid="stCode"] pre, div[data-testid="stCode"] code {
        font-family: inherit !important; font-size: 1rem; white-space: pre-wrap;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("🌐 번역기")
st.caption("텍스트를 입력하고 번역할 언어 버튼을 눌러 주세요.")

text = st.text_area(
    "번역할 텍스트",
    height=180,
    placeholder="번역할 텍스트를 입력하세요. (입력 후 버튼을 누르거나 Ctrl+Enter)",
    label_visibility="collapsed",
)
over = len(text) > MAX_CHARS
st.caption(f"{'⚠️ ' if over else ''}{len(text):,} / {MAX_CHARS:,}자")

cols = st.columns(len(LANGUAGES))
clicked = None
for col, label in zip(cols, LANGUAGES):
    if col.button(BUTTON_LABELS[label], use_container_width=True, type="primary"):
        clicked = label

if clicked:
    if not text.strip():
        st.session_state.pop("result", None)
        st.warning("번역할 텍스트를 입력해 주세요.")
    elif len(text) > MAX_CHARS:
        st.session_state.pop("result", None)
        st.error(f"입력은 {MAX_CHARS:,}자까지 가능합니다. (현재 {len(text):,}자)")
    else:
        try:
            with st.spinner("번역 중..."):
                translated = translate(text, LANGUAGES[clicked])
            st.session_state["result"] = {"language": clicked, "text": translated}
        except TranslationError as e:
            st.session_state.pop("result", None)
            st.error(str(e))

result = st.session_state.get("result")
if result:
    st.divider()
    st.subheader(f"번역 결과 - {result['language']}")
    st.code(result["text"], language=None, wrap_lines=True)
