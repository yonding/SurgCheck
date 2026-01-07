# convert_to_jsonl.py
import json
import sys
from pathlib import Path

# 출력에 펼쳐 넣을(복사할) 메타 키 목록
META_KEYS = {
    "main_tag",
    "sub_tag",
    "sub_sub_tag",
    "ori_img_tag",
}

def load_input(path: Path):
    text = path.read_text(encoding="utf-8").strip()
    # 입력이 JSON Lines(.jsonl)이면 라인별 파싱, 아니면 배열형 JSON으로 가정
    if text.startswith("["):
        return json.loads(text)
    else:
        data = []
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            data.append(json.loads(line))
        return data

def ensure_image_prefix(q: str) -> str:
    prefix = "<image>\n"
    if q.lstrip().startswith("<image>"):
        return q  # 이미 붙어 있으면 그대로 둠
    return prefix + q

def extract_meta_flat(item: dict) -> dict:
    """
    item의 최상위 또는 item['meta'] 내부에 있는 META_KEYS를 찾아
    {key: value}로 평탄화하여 반환.
    - 충돌 시: item 최상위 값이 meta 내부 값보다 우선.
    - 값이 None이거나 빈 문자열이어도 그대로 보존.
    """
    flat = {}

    # 1) meta 딕셔너리에서 우선 수집
    meta = item.get("meta", {})
    if isinstance(meta, dict):
        for k in META_KEYS:
            if k in meta:
                flat[k] = meta[k]

    # 2) item 최상위가 있으면 덮어씀(우선순위 높음)
    for k in META_KEYS:
        if k in item:
            flat[k] = item[k]

    return flat

def convert_item(item):
    image = item.get("image_path")
    convs = item.get("conversations", [])
    # (human, gpt) 페어로 분할
    pairs = []
    i = 0
    while i < len(convs):
        if convs[i].get("from") != "human":
            i += 1
            continue
        q = convs[i].get("value", "")
        a = ""
        if i + 1 < len(convs) and convs[i + 1].get("from") == "gpt":
            a = convs[i + 1].get("value", "")
            i += 2
        else:
            # 짝이 없는 경우도 한 줄로 내보내되 답변은 빈 문자열
            i += 1
        pairs.append((q, a))

    # 메타 평탄화 추출
    meta_flat = extract_meta_flat(item)

    # 원하는 형식으로 변환
    outputs = []
    for q, a in pairs:
        q_prefixed = ensure_image_prefix(q)
        obj = {
            "image": image,
            "conversations": [
                {"from": "human", "value": q_prefixed},
                {"from": "gpt",   "value": a}
            ]
        }
        # 평탄화 메타를 최상위에 병합
        obj.update(meta_flat)
        outputs.append(obj)
    return outputs

def main(inp: str, outp: str):
    in_path = Path(inp)
    out_path = Path(outp)
    data = load_input(in_path)

    with out_path.open("w", encoding="utf-8") as f:
        for item in data:
            for obj in convert_item(item):
                f.write(json.dumps(obj, ensure_ascii=False) + "\n")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("사용법: python convert_to_jsonl.py input.json output.jsonl")
        sys.exit(1)
    main(sys.argv[1], sys.argv[2])
