"""
Reusable components for a search UI.
"""
from dataclasses import dataclass
from time import perf_counter
from typing import Callable
from typing import List
from typing import Optional

import streamlit as st

RESULT_HEADER_LEVEL = "#####"
prev_query = ""
prev_results = []


@dataclass(frozen=True)
class Document:
    title: str
    text: str


@dataclass(frozen=True)
class Result:
    document: Document
    score: float


def _collapsed_text(is_collapsed: bool) -> str:
    return "expand" if is_collapsed else "collapse"


def _format_result_text(text: str):
    tag = """
    <style>
    code {
        white-space : pre-wrap !important;
        word-break: break-word;
    }
    </style>
    """
    return f"{tag}\n```\n{text}\n```"


def _is_collapsed(key: str):
    # NOTE: We default to collapsed.
    return st.session_state.get(f"{key}_collapsed", True)


def _reset_collapsed_state():
    for key in st.session_state.keys():
        if str(key).endswith("collapsed"):
            st.session_state[key] = True


def _toggle_collapsed(key: str):
    st.session_state[f"{key}_collapsed"] = not _is_collapsed(key)


def render_result(key: str, result: Result, max_desc_len: int = 160):
    # Bound max length because we truncate to "text...".
    assert max_desc_len > 4

    # Create a container for the result.
    score_container, result_container = st.columns([1, 10])

    # Write the score.
    score_container.markdown(f"{RESULT_HEADER_LEVEL} Score")
    score_container.markdown(f"{result.score:.4f}")

    # Write the title.
    result_container.markdown(f"{RESULT_HEADER_LEVEL} :orange[{result.document.title}]")

    # Create a sub-container so that we can have expanding as a feature that
    # updates the text above an expand button.
    summary_container = result_container.container()

    # If the text is short, we always render the full text.
    if len(result.document.text) <= max_desc_len:
        text = result.document.text
    # If the text is long, we render an excerpt with an expand button.
    else:
        result_container.button(
            _collapsed_text(_is_collapsed(key)),
            key=key,
            on_click=lambda: _toggle_collapsed(key),
        )

        # Conditionally render either an excerpt or the full text.
        if _is_collapsed(key):
            text = result.document.text[: max_desc_len - 3] + "..."
        else:
            text = result.document.text

    # Fill in the summary text.
    # summary_container.code(text, language="text")
    summary_container.markdown(_format_result_text(text), unsafe_allow_html=True)

    # Add some spacing.
    result_container.markdown("\n")

    return result_container


def search_app(
    search_fn: Callable[[str], List[Result]],
    init_fn: Optional[Callable] = None,
    title: str = "Semantic Search",
    show_time: bool = False,
):
    global prev_query, prev_results

    # Initialize app.
    st.set_page_config(title, layout="wide")
    if init_fn is not None:
        init_fn()

    # Query UI.
    st.markdown(f"# {title}")
    st.markdown("## Query")
    query = st.text_input(
        label="Type your query below. Hit enter to search.",
        max_chars=160,
        type="default",
        on_change=_reset_collapsed_state,
    )
    st.divider()

    # Results UI.
    if query != "":
        if query == prev_query:
            results = prev_results
            search_time = None
        else:
            start = perf_counter()
            results = search_fn(query)
            search_time = perf_counter() - start
            prev_query = query
            prev_results = results

        st.markdown("## Results")
        if search_time is not None:
            time_msg = f"Searched in {search_time:.4f}s"
            if show_time:
                st.text(time_msg)
            else:
                print(time_msg)
        for i, result in enumerate(results):
            key = f"result_{i}"
            render_result(key, result)
