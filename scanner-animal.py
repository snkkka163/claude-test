#!/usr/bin/env python3
"""
scanner-animal.py — 基于 YOLOv5 的动物识别脚本

传入一张图片，识别其中的动物，在图上画框，并返回：
  * 动物总数、各类别数量
  * 每只动物的 x, y（框左上角）、width、height（单位：像素）

模型：使用开源项目 ultralytics/yolov5 官方发布的 COCO 预训练权重（默认 yolov5s），
首次运行会自动从 GitHub 下载代码和权重，无需自己训练。
COCO 中属于动物的类别：bird, cat, dog, horse, sheep, cow, elephant, bear, zebra, giraffe

安装依赖：
    pip install torch torchvision opencv-python pandas requests pyyaml tqdm matplotlib seaborn scipy

用法：
    python scanner-animal.py test.jpg
    python scanner-animal.py test.jpg --json                  # 输出 JSON，方便程序调用
    python scanner-animal.py test.jpg --model yolov5m --conf 0.4 --output out.jpg

作为模块使用（文件名带连字符，需用 importlib 导入）：
    import importlib.util
    spec = importlib.util.spec_from_file_location("scanner_animal", "scanner-animal.py")
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    result = m.detect_animals("test.jpg")
    print(result["count"], result["animals"])
"""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

# COCO 数据集中的动物类别
ANIMAL_CLASSES = (
    "bird", "cat", "dog", "horse", "sheep",
    "cow", "elephant", "bear", "zebra", "giraffe",
)

# 类别中文名，仅用于终端输出
CN_NAMES = {
    "bird": "鸟", "cat": "猫", "dog": "狗", "horse": "马", "sheep": "羊",
    "cow": "牛", "elephant": "大象", "bear": "熊", "zebra": "斑马", "giraffe": "长颈鹿",
}


# --------------------------------------------------------------------------- #
# 模型加载
# --------------------------------------------------------------------------- #
def load_model(model="yolov5s", weights=None, repo=None, device=None, conf=0.25, iou=0.45):
    """
    加载 YOLOv5 模型。

    model:   官方预训练模型名 yolov5n / yolov5s / yolov5m / yolov5l / yolov5x（会自动下载）
    weights: 本地 .pt 权重路径；给了就优先使用它，而不是官方预训练模型
    repo:    本地 yolov5 仓库目录（离线使用）；不给则从 GitHub 拉取 ultralytics/yolov5
    device:  'cpu' / 'cuda' / '0' 等，默认自动选择
    """
    try:
        import torch  # 延迟导入，方便在没装 torch 的环境里测试其余逻辑
    except ImportError:
        sys.exit("未安装 PyTorch。请先执行：pip install torch torchvision")

    source = "local" if repo else "github"
    repo_or_dir = str(repo) if repo else "ultralytics/yolov5"

    try:
        if weights:
            net = torch.hub.load(repo_or_dir, "custom", path=str(weights), source=source)
        else:
            net = torch.hub.load(repo_or_dir, model, pretrained=True, source=source)
    except Exception as exc:  # 网络、依赖缺失等
        sys.exit(
            "加载 YOLOv5 失败：%s\n"
            "排查建议：\n"
            "  1) 检查网络能否访问 github.com（首次运行需下载代码和权重）；\n"
            "  2) 补装依赖：pip install pandas requests pyyaml tqdm matplotlib seaborn scipy；\n"
            "  3) 离线使用：先 git clone https://github.com/ultralytics/yolov5，"
            "再用 --repo 指向该目录，并用 --weights 指定 .pt 文件。" % exc
        )

    net.conf = conf  # 置信度阈值
    net.iou = iou    # NMS IoU 阈值
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
def detect_animals(image_path, model=None, output_path=None, classes=ANIMAL_CLASSES,
                   imgsz=640, draw=True, **load_kwargs):
    """
    识别图片中的动物。

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

    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    results = model(rgb, size=imgsz)
    names = model.names
    wanted = set(classes)

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
    ap.add_argument("--model", default="yolov5s",
                    choices=["yolov5n", "yolov5s", "yolov5m", "yolov5l", "yolov5x"],
                    help="官方预训练模型，越往右越准越慢，默认 yolov5s")
    ap.add_argument("--weights", help="自己的 .pt 权重文件（优先于 --model）")
    ap.add_argument("--repo", help="本地 yolov5 仓库目录（离线使用）")
    ap.add_argument("--conf", type=float, default=0.25, help="置信度阈值，默认 0.25")
    ap.add_argument("--iou", type=float, default=0.45, help="NMS IoU 阈值，默认 0.45")
    ap.add_argument("--imgsz", type=int, default=640, help="推理尺寸，默认 640")
    ap.add_argument("--device", help="设备：cpu / cuda / 0，默认自动")
    ap.add_argument("--classes", help="只识别这些类别，逗号分隔，如 cat,dog；默认所有动物")
    ap.add_argument("--no-draw", action="store_true", help="只输出数据，不保存画框图片")
    ap.add_argument("--json", action="store_true", help="以 JSON 输出结果")
    return ap.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if not Path(args.image).is_file():
        sys.exit(f"找不到图片：{args.image}")

    classes = tuple(c.strip() for c in args.classes.split(",") if c.strip()) if args.classes else ANIMAL_CLASSES

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
