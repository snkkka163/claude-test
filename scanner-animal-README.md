# scanner-animal

传入一张图片，识别动物并画框，返回动物数量和每只的 `x, y, width, height`（`x, y` 为框左上角，单位像素）。

## 安装

```bash
git clone https://github.com/snkkka163/claude-test.git
cd claude-test
pip install torch torchvision opencv-python pandas requests pyyaml tqdm matplotlib seaborn scipy
```

## 使用

```bash
# 默认：MegaDetector（适合野外相机 / 夜间红外），首次运行会下载约 270MB 权重
python scanner-animal.py 图片.jpg

# 输出 JSON
python scanner-animal.py 图片.jpg --json

# 夜间漏检时降低置信度阈值
python scanner-animal.py 图片.jpg --conf 0.1

# 指定结果图片路径（默认 <原图名>_result.jpg）
python scanner-animal.py 图片.jpg -o out.jpg

# 只要数据，不保存画框图片
python scanner-animal.py 图片.jpg --no-draw
```

## 换模型

```bash
# MegaDetector v5b
python scanner-animal.py 图片.jpg --model megadetector-b

# COCO 模型，可区分 猫/狗/鸟/马/羊/牛/大象/熊/斑马/长颈鹿
python scanner-animal.py 图片.jpg --model yolov5s
python scanner-animal.py 图片.jpg --model yolov5m --conf 0.4

# 只识别指定类别（COCO 模型）
python scanner-animal.py 图片.jpg --model yolov5s --classes cat,dog

# 使用本地权重 / 离线运行
python scanner-animal.py 图片.jpg --weights md_v5a.0.0.pt --repo ./yolov5
```

## 输出示例（数值仅为演示格式）

```
共识别到 1 只动物
  animal（动物）: 1

序号  类别        置信度    x      y      width  height
1     animal      0.85      912    565    85     84
```

`--json` 输出：

```json
{
  "count": 1,
  "counts": {"animal": 1},
  "animals": [
    {"label": "animal", "confidence": 0.85, "x": 912, "y": 565, "width": 85, "height": 84}
  ],
  "output_image": "图片_result.jpg"
}
```

## 参数

| 参数 | 说明 |
| --- | --- |
| `--model` | `megadetector`（默认）、`megadetector-b`、`yolov5n/s/m/l/x` |
| `--weights` | 本地 `.pt` 权重，优先于 `--model` |
| `--repo` | 本地 yolov5 仓库目录（离线用） |
| `--conf` | 置信度阈值，默认 MegaDetector 0.2 / COCO 0.25 |
| `--iou` | NMS IoU 阈值，默认 0.45 |
| `--imgsz` | 推理尺寸，默认 MegaDetector 1280 / COCO 640 |
| `--device` | `cpu` / `cuda` / `0` |
| `--classes` | 只保留指定类别，逗号分隔 |
| `-o, --output` | 结果图片路径 |
| `--no-draw` | 不保存画框图片 |
| `--json` | 以 JSON 输出 |

## 说明

- MegaDetector 只分 `animal / person / vehicle`，脚本只保留 `animal`，因此不区分具体动物种类。
- 权重缓存在 `~/.cache/scanner-animal/`，可用环境变量 `SCANNER_ANIMAL_CACHE` 修改。
