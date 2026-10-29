#!/usr/bin/env python3
"""
scanner-animal.py — 基于 YOLOv5 的动物识别脚本（支持野外相机陷阱 / 夜间红外图）

传入一张图片，识别其中的动物，在图上画框，并返回：
  * 动物总数、各类别数量
  * 每只动物的 x, y（框左上角）、width、height（单位：像素）

两类开源模型（都基于 ultralytics/yolov5，首次运行自动下载，无需自己训练）：
  * megadetector（默认）：MegaDetector v5，专为野外相机陷阱图片训练，白天、夜间红外都能用，
    小目标和林间遮挡表现远好于通用模型。只区分 animal / person / vehicle 三类，本脚本只保留 animal。
  * yolov5n/s/m/l/x：官方 COCO 预训练模型，能区分具体种类
    （bird, cat, dog, horse, sheep, cow, elephant, bear, zebra, giraffe），适合手机拍的普通照片。

安装依赖：
    pip install torch torchvision opencv-python pandas requests pyyaml tqdm matplotlib seaborn scipy

用法：
    python scanner-animal.py test.jpg                          # 默认 MegaDetector v5a
    python scanner-animal.py test.jpg --json                   # 输出 JSON，方便程序调用
    python scanner-animal.py test.jpg --conf 0.1               # 夜间漏检时降低阈值
    python scanner-animal.py test.jpg --model yolov5m          # 改用 COCO 模型，可区分猫/狗等
    python scanner-animal.py test.jpg --model megadetector-b   # MegaDetector v5b

MegaDetector 权重（约 270MB）会下载到 ~/.cache/scanner-animal/，只下载一次。

作为模块使用（文件名带连字符，需用 importlib 导入）：
    import importlib.util
    spec = importlib.util.spec_from_file_location("scanner_animal", "scanner-animal.py")
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    result = m.detect_animals("test.jpg")
    print(result["count"], result["animals"])
"""

import argparse
import json
import os
import shutil
import sys
import urllib.request
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

# COCO 数据集中的动物类别（用于 yolov5n/s/m/l/x）
ANIMAL_CLASSES = (
    "bird", "cat", "dog", "horse", "sheep",
    "cow", "elephant", "bear", "zebra", "giraffe",
)
# MegaDetector 的动物类别名
MD_ANIMAL = ("animal",)
MD_NAMES = {0: "animal", 1: "person", 2: "vehicle"}  # MegaDetector 类别 1/2/3（模型内为 0/1/2）

# 类别中文名，仅用于终端输出
CN_NAMES = {
    "bird": "鸟", "cat": "猫", "dog": "狗", "horse": "马", "sheep": "羊",
    "cow": "牛", "elephant": "大象", "bear": "熊", "zebra": "斑马", "giraffe": "长颈鹿",
    "animal": "动物",
}

# MegaDetector v5 权重（同一文件在两个 GitHub 仓库都有发布，依次尝试）
MD_MODELS = {
    "megadetector":   "md_v5a.0.0.pt",
    "megadetector-b": "md_v5b.0.0.pt",
}
MD_URLS = (
    "https://github.com/agentmorris/MegaDetector/releases/download/v5.0/{name}",
    "https://github.com/microsoft/CameraTraps/releases/download/v5.0/{name}",
)
MD_MIN_BYTES = 200 * 1024 * 1024  # 完整文件约 280MB，小于该值视为下载不完整
MD_IMGSZ = 1280                   # MegaDetector v5 以 1280 训练
MD_CONF = 0.2                     # MegaDetector 常用阈值
COCO_IMGSZ, COCO_CONF = 640, 0.25
CACHE_DIR = Path(os.environ.get("SCANNER_ANIMAL_CACHE", Path.home() / ".cache" / "scanner-animal"))


# --------------------------------------------------------------------------- #
# 权重下载
# --------------------------------------------------------------------------- #
def download_file(urls, dest, min_bytes=1):
    """依次尝试 urls，下载到 dest（先写 .part，完整后再改名）。已存在且够大则直接返回。"""
    dest = Path(dest)
    if dest.is_file() and dest.stat().st_size >= min_bytes:
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + ".part")
    errors = []
    for url in urls:
        try:
            print(f"下载模型权重：{url}", file=sys.stderr)
            with urllib.request.urlopen(url, timeout=60) as resp, open(part, "wb") as f:
                total = int(resp.headers.get("Content-Length") or 0)
                done, last = 0, -1
                while True:
                    chunk = resp.read(1 << 20)
                    if not chunk:
                        break
                    f.write(chunk)
                    done += len(chunk)
                    if total and done * 100 // total != last and (done * 100 // total) % 10 == 0:
                        last = done * 100 // total
                        print(f"  {last}%  ({done >> 20}/{total >> 20} MB)", file=sys.stderr)
            if part.stat().st_size < min_bytes:
                raise IOError(f"文件过小（{part.stat().st_size} 字节），下载不完整")
            shutil.move(str(part), str(dest))
            return dest
        except Exception as exc:  # 网络错误 / 404 / 不完整，换下一个源
            errors.append(f"{url} -> {exc}")
            if part.exists():
                part.unlink()
    raise RuntimeError("权重下载失败：\n  " + "\n  ".join(errors))


# --------------------------------------------------------------------------- #
# 模型加载
# --------------------------------------------------------------------------- #
def load_model(model="megadetector", weights=None, repo=None, device=None, conf=None, iou=0.45):
    """
    加载 YOLOv5 系列模型。

    model:   megadetector / megadetector-b（默认，自动下载权重）
             或 yolov5n / yolov5s / yolov5m / yolov5l / yolov5x（官方 COCO 模型，自动下载）
    weights: 本地 .pt 权重路径；给了就优先使用它（按自定义权重处理）
    repo:    本地 yolov5 仓库目录（离线使用）；不给则从 GitHub 拉取 ultralytics/yolov5
    device:  'cpu' / 'cuda' / '0' 等，默认自动选择
    conf:    置信度阈值；不给则 MegaDetector 用 0.2，COCO 模型用 0.25
    返回的模型带有 scanner_imgsz 属性（推荐推理尺寸），detect_animals 会自动使用。
    """
    try:
        import torch  # 延迟导入，方便在没装 torch 的环境里测试其余逻辑
    except ImportError:
        sys.exit("未安装 PyTorch。请先执行：pip install torch torchvision")

    is_md = weights is None and model in MD_MODELS
    if weights:
        weights_path = Path(weights)
    elif is_md:
        name = MD_MODELS[model]
        try:
            weights_path = download_file([u.format(name=name) for u in MD_URLS],
                                         CACHE_DIR / name, MD_MIN_BYTES)
        except RuntimeError as exc:
            sys.exit(f"{exc}\n可手动下载 {name} 后用 --weights 指定本地路径。")
    else:
        weights_path = None

    source = "local" if repo else "github"
    repo_or_dir = str(repo) if repo else "ultralytics/yolov5"

    orig_load = torch.load
    if weights_path is not None:
        # 旧版 YOLOv5 权重是完整 pickle，PyTorch>=2.6 默认 weights_only=True 会拒绝加载。
        # MegaDetector 官方代码同样使用 weights_only=False；仅对这次加载生效，加载后还原。
        def _legacy_load(*a, **k):
            k.setdefault("weights_only", False)
            return orig_load(*a, **k)
        torch.load = _legacy_load

    try:
        if weights_path is not None:
            net = torch.hub.load(repo_or_dir, "custom", path=str(weights_path), source=source)
        else:
            net = torch.hub.load(repo_or_dir, model, pretrained=True, source=source)
    except Exception as exc:  # 网络、依赖缺失等
        sys.exit(
            "加载 YOLOv5 失败：%s\n"
            "排查建议：\n"
            "  1) 检查网络能否访问 github.com（首次运行需下载 yolov5 代码）；\n"
            "  2) 补装依赖：pip install pandas requests pyyaml tqdm matplotlib seaborn scipy；\n"
            "  3) 离线使用：先 git clone https://github.com/ultralytics/yolov5，"
            "再用 --repo 指向该目录，并用 --weights 指定 .pt 文件。" % exc
        )
    finally:
        torch.load = orig_load

    net.conf = conf if conf is not None else (MD_CONF if is_md else COCO_CONF)  # 置信度阈值
    net.iou = iou                                                              # NMS IoU 阈值
    net.scanner_imgsz = MD_IMGSZ if is_md else COCO_IMGSZ
    if is_md:
        net.names = dict(MD_NAMES)  # 按官方文档固定类别顺序，不依赖权重里保存的名称
    if device:
        device = str(device)
        net.to(f"cuda:{device}" if device.isdigit() else device)  # "0" -> "cuda:0"
    return net


# --------------------------------------------------------------------------- #
# 图片读写（兼容中文路径）
# --------------------------------------------------------------------------- #
def read_image(path):
    data = np.fromfile(str(path), dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"无法读取图片（文件不存在或格式不支持）：{path}")
    return img


def write_image(path, img):
    ext = Path(path).suffix or ".jpg"
    ok, buf = cv2.imencode(ext, img)
    if not ok:
        raise ValueError(f"无法保存图片：{path}")
    buf.tofile(str(path))


# --------------------------------------------------------------------------- #
# 检测
# --------------------------------------------------------------------------- #
def detect_animals(image_path, model=None, output_path=None, classes=None,
                   imgsz=None, draw=True, **load_kwargs):
    """
    识别图片中的动物。

    classes: 要保留的类别名；默认自动选择（MegaDetector 保留 animal，COCO 模型保留 10 种动物）
    imgsz:   推理尺寸；默认用模型推荐值（MegaDetector 1280，COCO 模型 640）

    返回 dict：
        {
          "image": 原图路径,
          "image_width": ..., "image_height": ...,
          "count": 动物总数,
          "counts": {"dog": 2, "cat": 1},
          "animals": [
              {"label": "dog", "confidence": 0.91,
               "x": 120, "y": 80, "width": 200, "height": 260},   # x,y 为框左上角
              ...
          ],
          "output_image": 画好框的图片路径（draw=False 时为 None）
        }
    """
    bgr = read_image(image_path)
    h, w = bgr.shape[:2]

    if model is None:
        model = load_model(**load_kwargs)

    names = model.names
    if classes is None:
        has_md = "animal" in (names.values() if isinstance(names, dict) else names)
        classes = MD_ANIMAL if has_md else ANIMAL_CLASSES
    wanted = set(classes)
    size = imgsz or getattr(model, "scanner_imgsz", COCO_IMGSZ)

    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    results = model(rgb, size=size)

    animals = []
    for x1, y1, x2, y2, conf, cls in results.xyxy[0].tolist():
        label = names[int(cls)]
        if label not in wanted:
            continue
        # 限制在图片范围内，转成 左上角 x,y + 宽高
        x1, y1 = max(0.0, x1), max(0.0, y1)
        x2, y2 = min(float(w), x2), min(float(h), y2)
        animals.append({
            "label": label,
            "confidence": round(float(conf), 4),
            "x": int(round(x1)),
            "y": int(round(y1)),
            "width": int(round(x2 - x1)),
            "height": int(round(y2 - y1)),
        })

    # 按置信度从高到低排序
    animals.sort(key=lambda a: a["confidence"], reverse=True)
    counts = dict(Counter(a["label"] for a in animals))

    out_file = None
    if draw:
        out_file = Path(output_path) if output_path else _default_output(image_path)
        write_image(out_file, draw_boxes(bgr, animals))

    return {
        "image": str(image_path),
        "image_width": w,
        "image_height": h,
        "count": len(animals),
        "counts": counts,
        "animals": animals,
        "output_image": str(out_file) if out_file else None,
    }


def _default_output(image_path):
    p = Path(image_path)
    return p.with_name(f"{p.stem}_result.jpg")


def draw_boxes(bgr, animals):
    """在图片副本上画出检测框和标签，不修改原图。"""
    img = bgr.copy()
    h, w = img.shape[:2]
    thickness = max(2, round(0.003 * (h + w) / 2))
    font_scale = max(0.5, 0.0006 * (h + w) / 2)
    palette = [(56, 56, 255), (151, 157, 255), (31, 112, 255), (29, 178, 255), (49, 210, 207),
               (10, 249, 72), (23, 204, 146), (134, 219, 61), (52, 147, 26), (187, 212, 0)]

    for i, a in enumerate(animals, 1):
        color = palette[ANIMAL_CLASSES.index(a["label"]) % len(palette)] if a["label"] in ANIMAL_CLASSES else (0, 255, 0)
        p1 = (a["x"], a["y"])
        p2 = (a["x"] + a["width"], a["y"] + a["height"])
        cv2.rectangle(img, p1, p2, color, thickness, cv2.LINE_AA)

        text = f'#{i} {a["label"]} {a["confidence"]:.2f}'
        (tw, th), base = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)
        top = p1[1] - th - base - 4 >= 0
        y0 = p1[1] - th - base - 4 if top else p1[1]
        cv2.rectangle(img, (p1[0], y0), (p1[0] + tw + 6, y0 + th + base + 4), color, -1, cv2.LINE_AA)
        cv2.putText(img, text, (p1[0] + 3, y0 + th + 1), cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale, (255, 255, 255), 1, cv2.LINE_AA)
    return img


# --------------------------------------------------------------------------- #
# 命令行
# --------------------------------------------------------------------------- #
def parse_args(argv=None):
    ap = argparse.ArgumentParser(description="基于 YOLOv5 识别图片中的动物并框选")
    ap.add_argument("image", help="输入图片路径")
    ap.add_argument("-o", "--output", help="结果图片路径，默认 <原图名>_result.jpg")
    ap.add_argument("--model", default="megadetector",
                    choices=["megadetector", "megadetector-b", "yolov5n", "yolov5s", "yolov5m", "yolov5l", "yolov5x"],
                    help="megadetector（默认，适合野外相机/夜间红外，只分'动物'）；"
                         "yolov5* 为 COCO 模型，可区分猫/狗等具体种类")
    ap.add_argument("--weights", help="自己的 .pt 权重文件（优先于 --model）")
    ap.add_argument("--repo", help="本地 yolov5 仓库目录（离线使用）")
    ap.add_argument("--conf", type=float, help="置信度阈值；默认 MegaDetector 0.2，COCO 模型 0.25。夜间漏检可降到 0.1")
    ap.add_argument("--iou", type=float, default=0.45, help="NMS IoU 阈值，默认 0.45")
    ap.add_argument("--imgsz", type=int, help="推理尺寸；默认 MegaDetector 1280，COCO 模型 640")
    ap.add_argument("--device", help="设备：cpu / cuda / 0，默认自动")
    ap.add_argument("--classes", help="只识别这些类别，逗号分隔，如 cat,dog；默认自动（所有动物）")
    ap.add_argument("--no-draw", action="store_true", help="只输出数据，不保存画框图片")
    ap.add_argument("--json", action="store_true", help="以 JSON 输出结果")
    return ap.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if not Path(args.image).is_file():
        sys.exit(f"找不到图片：{args.image}")

    classes = tuple(c.strip() for c in args.classes.split(",") if c.strip()) if args.classes else None

    model = load_model(model=args.model, weights=args.weights, repo=args.repo,
                       device=args.device, conf=args.conf, iou=args.iou)
    result = detect_animals(args.image, model=model, output_path=args.output,
                            classes=classes, imgsz=args.imgsz, draw=not args.no_draw)

    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0

    print(f"图片：{result['image']}（{result['image_width']}x{result['image_height']}）")
    print(f"共识别到 {result['count']} 只动物")
    for name, n in result["counts"].items():
        print(f"  {name}（{CN_NAMES.get(name, name)}）: {n}")
    if result["animals"]:
        print("\n序号  类别        置信度    x      y      width  height")
        for i, a in enumerate(result["animals"], 1):
            print(f"{i:<5} {a['label']:<11} {a['confidence']:<9.2f} "
                  f"{a['x']:<6} {a['y']:<6} {a['width']:<6} {a['height']}")
    if result["output_image"]:
        print(f"\n画框结果已保存：{result['output_image']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
