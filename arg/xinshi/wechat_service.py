from __future__ import annotations

import json
import logging
import time

from arg.xinshi.chat_service import handle_chat_request
from arg.xinshi.logutil import preview_text
from arg.xinshi.schemas import ChatRequest
from arg.xinshi.wechat_crypto import (
    decrypt_json_payload,
    encrypt_json_payload,
    verify_msg_signature,
)

log = logging.getLogger(__name__)


class WeChatPayloadError(Exception):
    def __init__(self, detail: str, status_code: int = 400):
        super().__init__(detail)
        self.detail = detail
        self.status_code = status_code


class WeChatSignatureError(WeChatPayloadError):
    def __init__(self):
        super().__init__("微信 msg_signature 校验失败", 403)


def _json_for_log(payload: dict | str | None) -> str:
    return json.dumps(payload, ensure_ascii=False, default=str, separators=(",", ":"))


def _extract_encrypt(payload: dict) -> str:
    if not isinstance(payload, dict):
        raise WeChatPayloadError("请求体 JSON 必须是对象")
    encrypt = payload.get("Encrypt")
    if not isinstance(encrypt, str) or not encrypt:
        raise WeChatPayloadError("请求体缺少 Encrypt")
    return encrypt


def build_encrypted_chat_reply(
    payload: dict,
    *,
    msg_signature: str,
    timestamp: str,
    nonce: str,
    openid: str | None,
    signature_present: bool,
) -> dict | None:
    encrypt = _extract_encrypt(payload)
    log.info("wechat request encrypted payload=%s", _json_for_log(payload))
    if not verify_msg_signature(msg_signature, timestamp, nonce, encrypt):
        log.warning(
            "invalid WeChat msg_signature timestamp=%s nonce=%s openid=%s signature_ignored=%s",
            timestamp,
            nonce,
            openid,
            signature_present,
        )
        raise WeChatSignatureError()

    msg = decrypt_json_payload(encrypt)
    log.info("wechat request decrypted payload=%s", _json_for_log(msg))
    msg_type = str(msg.get("MsgType") or "").strip()
    content = str(msg.get("Content") or "").strip()
    if msg_type and msg_type != "text":
        log.info("ignore unsupported WeChat msg_type=%s openid=%s", msg_type, openid)
        log.info("wechat response plaintext payload=%s", _json_for_log("success"))
        log.info("wechat response encrypted payload=%s", _json_for_log(None))
        return None
    if not content:
        log.info("ignore WeChat message without Content msg_type=%s openid=%s", msg_type, openid)
        log.info("wechat response plaintext payload=%s", _json_for_log("success"))
        log.info("wechat response encrypted payload=%s", _json_for_log(None))
        return None

    log.info(
        "wechat encrypted JSON message received from=%s to=%s openid=%s content_len=%d preview=%r",
        msg.get("FromUserName"),
        msg.get("ToUserName"),
        openid,
        len(content),
        preview_text(content, 80),
    )
    chat_resp = handle_chat_request(ChatRequest(message=content))
    now = int(time.time())
    reply_message = {
        "ToUserName": msg.get("FromUserName") or openid or "",
        "FromUserName": msg.get("ToUserName") or "",
        "CreateTime": now,
        "MsgType": "text",
        "Content": chat_resp.answer,
    }
    log.info("wechat response plaintext payload=%s", _json_for_log(reply_message))
    encrypted_reply = encrypt_json_payload(reply_message, str(now), nonce)
    log.info("wechat response encrypted payload=%s", _json_for_log(encrypted_reply))
    return encrypted_reply
