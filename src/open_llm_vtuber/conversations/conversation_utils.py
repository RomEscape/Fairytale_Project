import asyncio
import re
from typing import Optional, Union, Any, List, Dict
import numpy as np
import json
from loguru import logger

from ..message_handler import message_handler
from .types import WebSocketSend, BroadcastContext
from .tts_manager import TTSTaskManager
from ..agent.output_types import SentenceOutput, AudioOutput
from ..agent.input_types import BatchInput, TextData, ImageData, TextSource, ImageSource
from ..asr.asr_interface import ASRInterface
from ..live2d_model import Live2dModel
from ..tts.tts_interface import TTSInterface
from ..utils.stream_audio import prepare_audio_payload


# Convert class methods to standalone functions
def create_batch_input(
    input_text: str,
    images: Optional[List[Dict[str, Any]]],
    from_name: str,
    metadata: Optional[Dict[str, Any]] = None,
) -> BatchInput:
    """Create batch input for agent processing"""
    return BatchInput(
        texts=[
            TextData(source=TextSource.INPUT, content=input_text, from_name=from_name)
        ],
        images=[
            ImageData(
                source=ImageSource(img["source"]),
                data=img["data"],
                mime_type=img["mime_type"],
            )
            for img in (images or [])
        ]
        if images
        else None,
        metadata=metadata,
    )


async def process_agent_output(
    output: Union[AudioOutput, SentenceOutput],
    character_config: Any,
    live2d_model: Live2dModel,
    tts_engine: TTSInterface,
    websocket_send: WebSocketSend,
    tts_manager: TTSTaskManager,
    translate_engine: Optional[Any] = None,
) -> str:
    """Process agent output with character information and optional translation"""
    output.display_text.name = character_config.character_name
    output.display_text.avatar = character_config.avatar

    full_response = ""
    try:
        if isinstance(output, SentenceOutput):
            full_response = await handle_sentence_output(
                output,
                live2d_model,
                tts_engine,
                websocket_send,
                tts_manager,
                translate_engine,
            )
        elif isinstance(output, AudioOutput):
            full_response = await handle_audio_output(output, websocket_send)
        else:
            logger.warning(f"Unknown output type: {type(output)}")
    except Exception as e:
        logger.error(f"Error processing agent output: {e}")
        await websocket_send(
            json.dumps(
                {"type": "error", "message": f"Error processing response: {str(e)}"}
            )
        )

    return full_response


def _fix_spacing(text: str) -> str:
    """
    Fix spacing after punctuation marks in Korean text.
    Adds space after punctuation marks (.,!?) if not already present.
    Also handles Korean honorific endings followed by Korean characters.
    """
    if not isinstance(text, str) or not text:
        return text
    
    # 쉼표, 마침표, 물음표, 느낌표 뒤에 공백이 없으면 추가
    # 단, 숫자 뒤의 마침표(예: 1.2)나 연속된 구두점(예: ...)은 제외
    # 한글 자음/모음 뒤에 구두점이 오는 경우도 처리
    # 존댓말 어미 뒤에 한글이 바로 오는 경우도 처리
    fixed_text = text
    # Korean honorific endings that should have space after them
    honorific_endings = ["어요", "아요", "해요", "예요", "세요", "습니다", "네요", "죠", "까요", "나요", "가요", "지요", "었어요", "았어요", "했습니다"]
    
    for _ in range(50):  # 여러 번 반복하여 모든 경우를 처리
        # 쉼표 뒤에 공백이 없는 경우 (가장 먼저 처리)
        new_text = re.sub(r'([,])([^\s\n])', r'\1 \2', fixed_text)
        # 구두점(마침표, 물음표, 느낌표) 뒤에 공백이 없는 경우
        new_text = re.sub(r'([.!?])([^\s\n])', r'\1 \2', new_text)
        # 연속된 구두점 처리 (예: ... 뒤에 공백이 없는 경우)
        new_text = re.sub(r'([.,!?]{2,})([^\s\n])', r'\1 \2', new_text)
        # 한글 뒤에 구두점이 붙어있고 그 뒤에 공백이 없는 경우
        new_text = re.sub(r'([가-힣])([.,!?])([^\s\n])', r'\1\2 \3', new_text)
        # 존댓말 어미 뒤에 한글이 바로 오는 경우 (반복 방지를 위해)
        # "있어요이" -> "있어요 이" 같은 경우 처리
        for ending in honorific_endings:
            # 존댓말 어미 뒤에 한글이 바로 오는 경우
            pattern = re.escape(ending) + r'([가-힣])'
            replacement = ending + r' \1'
            new_text = re.sub(pattern, replacement, new_text)
        # 숫자 뒤의 마침표는 제외 (예: 1.2는 1. 2로 바뀌지 않도록)
        new_text = re.sub(r'(\d)\. (\d)', r'\1.\2', new_text)
        
        if new_text == fixed_text:
            break
        fixed_text = new_text
    return fixed_text


async def handle_sentence_output(
    output: SentenceOutput,
    live2d_model: Live2dModel,
    tts_engine: TTSInterface,
    websocket_send: WebSocketSend,
    tts_manager: TTSTaskManager,
    translate_engine: Optional[Any] = None,
) -> str:
    """Handle sentence output type with optional translation support"""
    full_response = ""
    accumulated_display_text = ""  # 누적된 display_text를 추적
    
    async for display_text, tts_text, actions in output:
        logger.debug(f"Processing output: '''{tts_text}'''...")

        if translate_engine:
            if len(re.sub(r'[\s.,!?，。！？\'"』」）】\s]+', "", tts_text)):
                tts_text = translate_engine.translate(tts_text)
            logger.info(f"Text after translation: '''{tts_text}'''...")
        else:
            logger.debug("No translation engine available. Skipping translation.")

        # 누적된 텍스트에 현재 chunk 추가
        accumulated_display_text += display_text.text
        
        # 반복 텍스트 제거 (ollama_llm.py에서 처리했지만 추가 안전장치)
        # 여러 패턴을 체크하여 다양한 반복을 감지
        if len(accumulated_display_text) > 10:
            text_len = len(accumulated_display_text)
            
            # Check 1: 전체 반복 (여러 split ratio 체크)
            for ratio in [0.4, 0.45, 0.5, 0.55, 0.6]:
                split_point = int(text_len * ratio)
                if split_point >= 10:
                    first_part = accumulated_display_text[:split_point]
                    second_part = accumulated_display_text[split_point:]
                    if second_part.startswith(first_part):
                        remaining = second_part[len(first_part):].strip()
                        if len(remaining) < len(first_part) * 0.1:
                            accumulated_display_text = first_part.strip()
                            logger.debug(f"Removed duplicate text in conversation_utils (ratio {ratio:.2f})")
                            break
            
            # Check 2: 끝부분 반복 감지
            if len(accumulated_display_text) == text_len:  # 아직 제거되지 않았으면
                check_len = min(text_len // 3, 50)
                if check_len >= 10:
                    last_part = accumulated_display_text[-check_len:]
                    search_text = accumulated_display_text[:text_len // 2]
                    if last_part in search_text:
                        first_idx = search_text.find(last_part)
                        if first_idx >= 0 and first_idx <= len(search_text) * 0.3:
                            accumulated_display_text = accumulated_display_text[:first_idx + len(last_part)].strip()
                            logger.debug("Removed duplicate phrase at end in conversation_utils")
        
        # Stop sequence 제거 (ollama_llm.py에서 처리했지만 추가 안전장치)
        cleaned_text = accumulated_display_text
        stop_sequences = ["<|eot|>", "<|eot_id|>", "\n사용자:", "\n사용자 말하기:"]
        for stop_seq in stop_sequences:
            if stop_seq in cleaned_text:
                cleaned_text = cleaned_text.split(stop_seq)[0]
        # 부분적으로 포함된 stop sequence도 제거
        if "<|eot" in cleaned_text:
            cleaned_text = cleaned_text.split("<|eot")[0]
        # 단일 문자 stop sequence도 제거 (예: "<" 만 있는 경우)
        if cleaned_text.endswith("<"):
            cleaned_text = cleaned_text[:-1].strip()
        
        # 누적된 전체 텍스트에 띄어쓰기 수정 적용 (강력하게)
        fixed_accumulated = _fix_spacing(cleaned_text)
        # 추가로 구두점 뒤 띄어쓰기 확인 및 수정
        fixed_accumulated = re.sub(r'([.!?])([가-힣])', r'\1 \2', fixed_accumulated)
        fixed_accumulated = re.sub(r'([,])([가-힣])', r'\1 \2', fixed_accumulated)
        
        # 띄어쓰기 수정 후 다시 stop sequence 체크
        if "<|eot" in fixed_accumulated:
            fixed_accumulated = fixed_accumulated.split("<|eot")[0].strip()
        if fixed_accumulated.endswith("<"):
            fixed_accumulated = fixed_accumulated[:-1].strip()
        
        # 이전에 전송한 텍스트 이후의 새로운 부분만 추출
        # (이전에 전송한 부분은 이미 수정되었으므로, 새로운 부분만 수정)
        prev_length = len(full_response)
        new_chunk = fixed_accumulated[prev_length:]
        
        # display_text.text를 수정된 새로운 chunk로 업데이트
        if new_chunk:
            display_text.text = new_chunk
        # new_chunk가 비어있으면 원본 유지 (이미 전송된 경우)
        
        full_response = fixed_accumulated  # 전체 누적 텍스트 업데이트
        
        await tts_manager.speak(
            tts_text=tts_text,
            display_text=display_text,
            actions=actions,
            live2d_model=live2d_model,
            tts_engine=tts_engine,
            websocket_send=websocket_send,
        )
    return full_response


async def handle_audio_output(
    output: AudioOutput,
    websocket_send: WebSocketSend,
) -> str:
    """Process and send AudioOutput directly to the client"""
    full_response = ""
    async for audio_path, display_text, transcript, actions in output:
        full_response += transcript
        audio_payload = prepare_audio_payload(
            audio_path=audio_path,
            display_text=display_text,
            actions=actions.to_dict() if actions else None,
        )
        await websocket_send(json.dumps(audio_payload))
    return full_response


async def send_conversation_start_signals(websocket_send: WebSocketSend) -> None:
    """Send initial conversation signals"""
    await websocket_send(
        json.dumps(
            {
                "type": "control",
                "text": "conversation-chain-start",
            }
        )
    )
    await websocket_send(json.dumps({"type": "full-text", "text": "Thinking..."}))


async def process_user_input(
    user_input: Union[str, np.ndarray],
    asr_engine: ASRInterface,
    websocket_send: WebSocketSend,
) -> str:
    """Process user input, converting audio to text if needed"""
    if isinstance(user_input, np.ndarray):
        logger.info("Transcribing audio input...")
        input_text = await asr_engine.async_transcribe_np(user_input)
        await websocket_send(
            json.dumps({"type": "user-input-transcription", "text": input_text})
        )
        return input_text
    return user_input


async def finalize_conversation_turn(
    tts_manager: TTSTaskManager,
    websocket_send: WebSocketSend,
    client_uid: str,
    broadcast_ctx: Optional[BroadcastContext] = None,
) -> None:
    """Finalize a conversation turn"""
    if tts_manager.task_list:
        await asyncio.gather(*tts_manager.task_list)
        await websocket_send(json.dumps({"type": "backend-synth-complete"}))

        response = await message_handler.wait_for_response(
            client_uid, "frontend-playback-complete"
        )

        if not response:
            logger.warning(f"No playback completion response from {client_uid}")
            return

    await websocket_send(json.dumps({"type": "force-new-message"}))

    if broadcast_ctx and broadcast_ctx.broadcast_func:
        await broadcast_ctx.broadcast_func(
            broadcast_ctx.group_members,
            {"type": "force-new-message"},
            broadcast_ctx.current_client_uid,
        )

    await send_conversation_end_signal(websocket_send, broadcast_ctx)


async def send_conversation_end_signal(
    websocket_send: WebSocketSend,
    broadcast_ctx: Optional[BroadcastContext],
    session_emoji: str = "",
) -> None:
    """Send conversation chain end signal"""
    chain_end_msg = {
        "type": "control",
        "text": "conversation-chain-end",
    }

    await websocket_send(json.dumps(chain_end_msg))

    if broadcast_ctx and broadcast_ctx.broadcast_func and broadcast_ctx.group_members:
        await broadcast_ctx.broadcast_func(
            broadcast_ctx.group_members,
            chain_end_msg,
        )

    logger.info(f"Conversation Chain completed!")


def cleanup_conversation(tts_manager: TTSTaskManager, session_emoji: str) -> None:
    """Clean up conversation resources"""
    tts_manager.clear()
    logger.debug(f"Clearing up conversation.")


EMOJI_LIST = [
    "🐶",
    "🐱",
    "🐭",
    "🐹",
    "🐰",
    "🦊",
    "🐻",
    "🐼",
    "🐨",
    "🐯",
    "🦁",
    "🐮",
    "🐷",
    "🐸",
    "🐵",
    "🐔",
    "🐧",
    "🐦",
    "🐤",
    "🐣",
    "🐥",
    "🦆",
    "🦅",
    "🦉",
    "🦇",
    "🐺",
    "🐗",
    "🐴",
    "🦄",
    "🐝",
    "🌵",
    "🎄",
    "🌲",
    "🌳",
    "🌴",
    "🌱",
    "🌿",
    "☘️",
    "🍀",
    "🍂",
    "🍁",
    "🍄",
    "🌾",
    "💐",
    "🌹",
    "🌸",
    "🌛",
    "🌍",
    "⭐️",
    "🔥",
    "🌈",
    "🌩",
    "⛄️",
    "🎃",
    "🎄",
    "🎉",
    "🎏",
    "🎗",
    "🀄️",
    "🎭",
    "🎨",
    "🧵",
    "🪡",
    "🧶",
    "🥽",
    "🥼",
    "🦺",
    "👔",
    "👕",
    "👜",
    "👑",
]
