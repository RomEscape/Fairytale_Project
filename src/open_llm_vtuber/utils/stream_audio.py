import base64
import re
from pydub import AudioSegment
from pydub.utils import make_chunks
from ..agent.output_types import Actions
from ..agent.output_types import DisplayText


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
    max_iterations = 50  # 최대 반복 횟수
    # Korean honorific endings that should have space after them
    honorific_endings = ["어요", "아요", "해요", "예요", "세요", "습니다", "네요", "죠", "까요", "나요", "가요", "지요", "었어요", "았어요", "했습니다"]
    
    for iteration in range(max_iterations):
        prev_text = fixed_text
        # 쉼표 뒤에 공백이 없는 경우 (가장 먼저 처리)
        fixed_text = re.sub(r'([,])([^\s\n])', r'\1 \2', fixed_text)
        # 구두점(마침표, 물음표, 느낌표) 뒤에 공백이 없는 경우
        fixed_text = re.sub(r'([.!?])([^\s\n])', r'\1 \2', fixed_text)
        # 연속된 구두점 처리 (예: ... 뒤에 공백이 없는 경우)
        fixed_text = re.sub(r'([.,!?]{2,})([^\s\n])', r'\1 \2', fixed_text)
        # 한글 뒤에 구두점이 붙어있고 그 뒤에 공백이 없는 경우
        fixed_text = re.sub(r'([가-힣])([.,!?])([^\s\n])', r'\1\2 \3', fixed_text)
        # 존댓말 어미 뒤에 한글이 바로 오는 경우 (반복 방지를 위해)
        # "있어요이" -> "있어요 이" 같은 경우 처리
        for ending in honorific_endings:
            # 존댓말 어미 뒤에 한글이 바로 오는 경우
            pattern = re.escape(ending) + r'([가-힣])'
            replacement = ending + r' \1'
            fixed_text = re.sub(pattern, replacement, fixed_text)
        # 숫자 뒤의 마침표는 제외 (예: 1.2는 1. 2로 바뀌지 않도록)
        fixed_text = re.sub(r'(\d)\. (\d)', r'\1.\2', fixed_text)
        
        # 더 이상 변경이 없으면 종료
        if fixed_text == prev_text:
            break
    return fixed_text


def _get_volume_by_chunks(audio: AudioSegment, chunk_length_ms: int) -> list:
    """
    Calculate the normalized volume (RMS) for each chunk of the audio.

    Parameters:
        audio (AudioSegment): The audio segment to process.
        chunk_length_ms (int): The length of each audio chunk in milliseconds.

    Returns:
        list: Normalized volumes for each chunk.
    """
    chunks = make_chunks(audio, chunk_length_ms)
    volumes = [chunk.rms for chunk in chunks]
    max_volume = max(volumes)
    if max_volume == 0:
        raise ValueError("Audio is empty or all zero.")
    return [volume / max_volume for volume in volumes]


def prepare_audio_payload(
    audio_path: str | None,
    chunk_length_ms: int = 20,
    display_text: DisplayText = None,
    actions: Actions = None,
    forwarded: bool = False,
) -> dict[str, any]:
    """
    Prepares the audio payload for sending to a broadcast endpoint.
    If audio_path is None, returns a payload with audio=None for silent display.

    Parameters:
        audio_path (str | None): The path to the audio file to be processed, or None for silent display
        chunk_length_ms (int): The length of each audio chunk in milliseconds
        display_text (DisplayText, optional): Text to be displayed with the audio
        actions (Actions, optional): Actions associated with the audio

    Returns:
        dict: The audio payload to be sent
    """
    if isinstance(display_text, DisplayText):
        # Apply spacing fix before converting to dict
        display_text.text = _fix_spacing(display_text.text)
        display_text = display_text.to_dict()
    elif isinstance(display_text, dict):
        # Apply spacing fix if it's already a dict
        if 'text' in display_text:
            display_text['text'] = _fix_spacing(display_text['text'])

    if not audio_path:
        # Return payload for silent display
        return {
            "type": "audio",
            "audio": None,
            "volumes": [],
            "slice_length": chunk_length_ms,
            "display_text": display_text,
            "actions": actions.to_dict() if actions else None,
            "forwarded": forwarded,
        }

    try:
        audio = AudioSegment.from_file(audio_path)
        audio_bytes = audio.export(format="wav").read()
    except Exception as e:
        raise ValueError(
            f"Error loading or converting generated audio file to wav file '{audio_path}': {e}"
        )
    audio_base64 = base64.b64encode(audio_bytes).decode("utf-8")
    volumes = _get_volume_by_chunks(audio, chunk_length_ms)

    payload = {
        "type": "audio",
        "audio": audio_base64,
        "volumes": volumes,
        "slice_length": chunk_length_ms,
        "display_text": display_text,
        "actions": actions.to_dict() if actions else None,
        "forwarded": forwarded,
    }

    return payload


# Example usage:
# payload, duration = prepare_audio_payload("path/to/audio.mp3", display_text="Hello", expression_list=[0,1,2])
