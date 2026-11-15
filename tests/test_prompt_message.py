"""_prompt_message: prompt blocks -> the turn's human message (v0.1.20).

Pure unit tests (no Temporal): multimodal prompts become langchain content
blocks; plain prompts stay byte-identical to the pre-0.1.20 string path so
old histories replay unchanged.
"""

from __future__ import annotations

from acp_temporal_bridge.session import ImageIn, PromptIn, _prompt_message


def test_plain_prompt_is_plain_string() -> None:
    msg = _prompt_message(PromptIn(text="hello"))
    assert msg.content == "hello"


def test_image_prompt_builds_content_blocks() -> None:
    msg = _prompt_message(
        PromptIn(
            text="what is this?",
            images=[ImageIn(data="QUJD", mime="image/png")],
        )
    )
    assert isinstance(msg.content, list)
    assert msg.content[0] == {"type": "text", "text": "what is this?"}
    assert msg.content[1] == {
        "type": "image_url",
        "image_url": {"url": "data:image/png;base64,QUJD"},
    }


def test_image_prompt_without_text_keeps_images_only() -> None:
    msg = _prompt_message(
        PromptIn(text="", images=[ImageIn(data="QQ==", mime="image/webp")])
    )
    assert msg.content == [
        {"type": "image_url", "image_url": {"url": "data:image/webp;base64,QQ=="}}
    ]


def test_attached_files_named_when_no_images() -> None:
    msg = _prompt_message(
        PromptIn(text="see attached", attached_files=["big.pdf (application/pdf)"])
    )
    assert "see attached" in msg.content
    assert "big.pdf (application/pdf)" in msg.content


def test_images_plus_attached_files() -> None:
    msg = _prompt_message(
        PromptIn(
            text="look",
            images=[ImageIn(data="QQ==", mime="image/png")],
            attached_files=["notes.txt"],
        )
    )
    kinds = [b["type"] for b in msg.content]
    assert kinds == ["text", "image_url", "text"]
    assert "notes.txt" in msg.content[-1]["text"]
