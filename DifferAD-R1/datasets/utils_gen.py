import torch.utils.data as data
import json
from PIL import Image
import numpy as np
# === NumPy 2.x 兼容 shim，必须放在 import imgaug 之前 ===
if not hasattr(np, "sctypes"):
    np.sctypes = {
        "int":    [np.int8, np.int16, np.int32, np.int64],
        "uint":   [np.uint8, np.uint16, np.uint32, np.uint64],
        "float":  [np.float16, np.float32, np.float64],
        "complex":[np.complex64, np.complex128],
        "others": [np.bool_, np.bytes_, np.str_, np.object_, np.datetime64, np.timedelta64],
    }
import torch
import os
from perlin import rand_perlin_2d_np
import cv2
import glob
import torchvision.transforms as transforms
import imgaug.augmenters as iaa
import glob
import time
import base64
import requests
import math

def single_region_from_perlin(perlin_thr, min_areapx: int = 64, top_k: int = 1):
    """
    从 Perlin 阈值图中选择前 top_k 个连通区域（按面积由大到小）。
    - top_k=1 : 与原函数一致（单区域）
    - top_k>1 : 返回多区域
    - top_k<=0: 保留所有区域
    返回形状保持 H*W*1，float32（0/1）
    """
    m = (np.squeeze(perlin_thr, axis=2) > 0).astype(np.uint8)  # H*W
    if m.sum() == 0:
        return perlin_thr.astype(np.float32)

    num, labels, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    if num <= 1:
        return perlin_thr.astype(np.float32)

    # 各区域面积
    areas = stats[1:, cv2.CC_STAT_AREA]
    idxs = [i+1 for i,a in enumerate(areas) if a >= min_areapx]
    if not idxs:
        idxs = list(range(1, num))  # 全部保留（避免全被阈掉）

    # 面积排序
    idxs.sort(key=lambda i: stats[i, cv2.CC_STAT_AREA], reverse=True)
    if top_k is not None and top_k > 0:
        idxs = idxs[:top_k]  # 取前k

    m_keep = np.isin(labels, np.array(idxs, dtype=labels.dtype)).astype(np.uint8)
    return m_keep[..., None].astype(np.float32)


def list_images(root_dir, patterns=("*.png","*.jpg","*.jpeg","*.bmp"), recursive=False):
    files = []
    for p in patterns:
        if recursive:
            files.extend(glob.glob(os.path.join(root_dir, "**", p), recursive=True))
        else:
            files.extend(glob.glob(os.path.join(root_dir, p)))
    return sorted(files)

def find_mask_for_anomaly(anom_path: str, gt_anom_dir: str):
    """
    依据 anomaly 图文件名，在 ground_truth/anomaly 下寻找对应 mask。
    兼容：same_stem.{png|bmp|jpg|jpeg} 或 same_stem + '_mask'.{...}
    找不到则返回 None。
    """
    stem = os.path.splitext(os.path.basename(anom_path))[0]
    # 优先同名
    for ext in (".png",".bmp",".jpg",".jpeg"):
        cand = os.path.join(gt_anom_dir, stem + ext)
        if os.path.isfile(cand):
            return cand
    # 其次 *_mask
    for ext in (".png",".bmp",".jpg",".jpeg"):
        cand = os.path.join(gt_anom_dir, stem + "_mask" + ext)
        if os.path.isfile(cand):
            return cand
    # 再尝试递归匹配（有些数据会有子目录）
    all_masks = list_images(gt_anom_dir, recursive=True)
    for m in all_masks:
        base = os.path.splitext(os.path.basename(m))[0]
        if base == stem or base == stem + "_mask":
            return m
    return None

def encode_image(image_path):
    resolution = 512
    with Image.open(image_path) as img:
        width, height = img.size
        if max(width, height) > resolution:  # 判断是否需要调整尺寸
            if width > height:
                new_width = resolution
                new_height = int((new_width / width) * height)
            else:
                new_height = resolution
                new_width = int((new_height / height) * width)
            img_resized = img.resize((new_width, new_height))
        else:
            img_resized = img
        import io
        buffer = io.BytesIO()
        img_resized.save(buffer, format="PNG")
        buffer.seek(0)
        return base64.b64encode(buffer.read()).decode('utf-8')

def request(content, max_retries=3, backoff=0.5):
    url = 'https://chat.intern-ai.org.cn/api/v1/chat/completions'
    api_key = os.getenv("SILICON_API_KEY", "")
    if not api_key:
        raise ValueError("请先设置环境变量 SILICON_API_KEY=你的token")

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    data = {
        "model": "internvl3-latest",
        "messages": [{
            "role": "user",
            "content": content
        }]
    }
    for attempt in range(max_retries):
        try:
            response = requests.post(url, headers=headers, data=json.dumps(data), timeout=30)
            resp_json = response.json()

            # 符合 OpenAI 风格的输出格式
            if "choices" in resp_json:
                return resp_json["choices"][0]["message"]["content"]

            print(f"API Error Response: {resp_json}")
        except requests.exceptions.RequestException as e:
            print(f"Request failed ({attempt+1}/{max_retries}): {e}")

        time.sleep(backoff * (2 ** attempt))  # 指数退避

    raise RuntimeError("API request failed after maximum retries")

def think_process_gen(good_path, systh_path, systh_mask_path, object_name):
    content = []
    content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encode_image(systh_path)}"}})
    content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encode_image(good_path)}"}})
    # content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encode_image(systh_mask_path)}"}})
    content.append({
                "type": "text",
                "text": (
                    # f'I am providing three images. The first two images feature the object labeled "{object_name}". '
                    # f'In the first image, the object is in good condition. The second image displays a defect, '
                    # f'while the third image presents a mask highlighting the defect identified in the second image. '
                    # f'Your task is 1.describe the surface characteristics of the second image in detail (color/texture/smoothness/whether there is wear, etc.), '
                    # f'2.Identify and describe the anomalous regions in the image, '
                    # f'3.Determine the anomaly type and approximate location of the anomalous regions to better achieve anomaly localization.'

                    # f'I am providing two images of the object labeled "{object_name}".'
                    # f'The first image shows the object in good condition.'
                    # f'The second image contains a defect.'
                    # f'Your task is:1. Describe the surface characteristics of the second image in detail (color, texture, smoothness, signs of wear, etc.). '
                    # f'2. Identify and describe the anomalous regions in the image. '
                    # f'Determine the anomaly type and approximate location of the anomalous regions to better achieve accurate anomaly localization. '
                    # f'Keep your response concise, about 100 words.'
                    # )})

                    f'I am providing two images of the object. Please compare the two given images and determine whether there are any differences.'
                    f'Provide a detailed description of the target image, and if anomalies or differences exist, focus on explaining the '
                    f'different regions and identify their approximate locations.'
                    f'Your task is:1. Describe the surface characteristics of the anomaly image in detail (color, texture, smoothness, signs of wear, etc.). '
                    f'2. Identify and describe the anomalous regions in the image. '
                    f'Determine the anomaly type and approximate location of the anomalous regions to better achieve accurate anomaly localization. '
                    f'Keep your response concise, about 100 words.'
                    )})
    desc = request(content).replace('\n', ' ').strip()
    return desc

# This is the resize function of Qwen2.5-VL
def smart_resize(
    height: int, width: int, factor: int = 28, min_pixels: int = 56 * 56, max_pixels: int = 14 * 14 * 4 * 1280
):
    """Rescales the image so that the following conditions are met:
    1. Both dimensions (height and width) are divisible by 'factor'.
    2. The total number of pixels is within the range ['min_pixels', 'max_pixels'].
    3. The aspect ratio of the image is maintained as closely as possible.
    """
    if height < factor or width < factor:
        raise ValueError(f"height:{height} or width:{width} must be larger than factor:{factor}")
    elif max(height, width) / min(height, width) > 200:
        raise ValueError(
            f"absolute aspect ratio must be smaller than 200, got {max(height, width) / min(height, width)}"
        )
    h_bar = round(height / factor) * factor
    w_bar = round(width / factor) * factor
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = math.floor(height / beta / factor) * factor
        w_bar = math.floor(width / beta / factor) * factor
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = math.ceil(height * beta / factor) * factor
        w_bar = math.ceil(width * beta / factor) * factor
    return h_bar, w_bar


def convert_to_qwen25vl_format(bbox, orig_height, orig_width, factor=28, min_pixels=56*56, max_pixels=14*14*4*1280):
    new_height, new_width = smart_resize(orig_height, orig_width, factor, min_pixels, max_pixels)
    scale_w = new_width / orig_width
    scale_h = new_height / orig_height
    
    x1, y1, x2, y2 = bbox
    x1_new = round(x1 * scale_w)
    y1_new = round(y1 * scale_h)
    x2_new = round(x2 * scale_w)
    y2_new = round(y2 * scale_h)
    
    x1_new = max(0, min(x1_new, new_width - 1))
    y1_new = max(0, min(y1_new, new_height - 1))
    x2_new = max(0, min(x2_new, new_width - 1))
    y2_new = max(0, min(y2_new, new_height - 1))
    
    return [x1_new, y1_new, x2_new, y2_new]

# ============== Qwen2-VL（0..1000）归一化（新增，按你的需求） ==============
def convert_to_qwen2vl_format(bbox, h, w):
    x1, y1, x2, y2 = bbox
    x1_new = round(x1 / w * 1000)
    y1_new = round(y1 / h * 1000)
    x2_new = round(x2 / w * 1000)
    y2_new = round(y2 / h * 1000)
    x1_new = max(0, min(x1_new, 1000))
    y1_new = max(0, min(y1_new, 1000))
    x2_new = max(0, min(x2_new, 1000))
    y2_new = max(0, min(y2_new, 1000))
    return [x1_new, y1_new, x2_new, y2_new]


# ====== Poisson 粘贴辅助 ======
# ====== Poisson 粘贴辅助（支持多连通域）======
def largest_cc_mask(
    bin_mask: np.ndarray,
    top_k: int = 1,            # 1=与原逻辑一致；>1=保留前k大连通域；<=0=保留全部
    min_areapx: int = 0,       # 连通域最小面积过滤
    connectivity: int = 8      # 4 或 8 连通
) -> np.ndarray:
    """
    返回 uint8 的 {0,255} 二值图。默认与原函数行为一致（仅最大连通域）。
    """
    # 标准化为 uint8 二值
    if bin_mask.dtype != np.uint8:
        bin_mask = (bin_mask > 0).astype(np.uint8) * 255
    else:
        _, bin_mask = cv2.threshold(bin_mask, 127, 255, cv2.THRESH_BINARY)

    # 没有前景直接返回
    if (bin_mask > 0).sum() == 0:
        return bin_mask

    num, labels, stats, _ = cv2.connectedComponentsWithStats(
        (bin_mask > 0).astype(np.uint8), connectivity=connectivity
    )
    if num <= 1:
        return bin_mask  # 只有背景

    # 统计每个连通域面积（跳过背景 id=0）
    areas = stats[1:, cv2.CC_STAT_AREA]
    ids = [i+1 for i, a in enumerate(areas) if a >= max(0, int(min_areapx))]
    if not ids:
        # 如果全被 min_areapx 过滤掉，则退化为原始“最大连通域”
        max_id = 1 + int(np.argmax(areas))  # 还原到 labels 下标
        out = np.where(labels == max_id, 255, 0).astype(np.uint8)
        return out

    # 按面积从大到小排序
    ids.sort(key=lambda i: stats[i, cv2.CC_STAT_AREA], reverse=True)
    if top_k is not None and top_k > 0:
        ids = ids[:top_k]  # 只保留前k大

    out = (np.isin(labels, np.array(ids, dtype=labels.dtype)).astype(np.uint8) * 255)
    return out


def random_scale_to_fit(h, w, box_h, box_w, scale_min=0.6, scale_max=1.4):
    """随机缩放，确保粘贴后能放进目标图像。返回缩放比例 s 与可放置中心范围。"""
    s = float(torch.empty(1).uniform_(scale_min, scale_max).numpy()[0])
    Hs, Ws = int(round(box_h * s)), int(round(box_w * s))
    s_h = (h - 2) / max(box_h, 1)
    s_w = (w - 2) / max(box_w, 1)
    s_cap = max(0.1, min(s, s_h, s_w))
    if s_cap != s:
        s = s_cap
        Hs, Ws = int(round(box_h * s)), int(round(box_w * s))
    cx_min, cx_max = Ws // 2 + 1, w - (Ws // 2) - 1
    cy_min, cy_max = Hs // 2 + 1, h - (Hs // 2) - 1
    if cx_min >= cx_max or cy_min >= cy_max:
        cx_min = cx_max = w // 2
        cy_min = cy_max = h // 2
    return s, (cx_min, cx_max, cy_min, cy_max)

def paste_mask(mask_canvas, small_mask, center_xy):
    """把 small_mask（uint8,{0,255}）按 center_xy 放到 mask_canvas 上。"""
    h, w = mask_canvas.shape[:2]
    hm, wm = small_mask.shape[:2]
    cx, cy = center_xy
    x1, y1 = cx - wm // 2, cy - hm // 2
    x2, y2 = x1 + wm, y1 + hm
    x1c, y1c = max(0, x1), max(0, y1)
    x2c, y2c = min(w, x2), min(h, y2)
    if x1c >= x2c or y1c >= y2c:
        return mask_canvas
    sx1, sy1 = x1c - x1, y1c - y1
    sx2, sy2 = sx1 + (x2c - x1c), sy1 + (y2c - y1c)
    roi = small_mask[sy1:sy2, sx1:sx2]
    if roi.dtype != np.uint8:
        roi = roi.astype(np.uint8)
    _, roi = cv2.threshold(roi, 127, 255, cv2.THRESH_BINARY)

    # # 关键：总是反转，得到“异常=255、背景=0”
    # roi = cv2.bitwise_not(roi)
    canvas_roi = mask_canvas[y1c:y2c, x1c:x2c]
    canvas_roi = np.where(roi > 0, 255, canvas_roi)
    mask_canvas[y1c:y2c, x1c:x2c] = canvas_roi
    return mask_canvas

def list_image_mask_pairs(root_dir):
    """
    在 root_dir 下递归搜集所有 test/anomaly/* 图像，
    并将其映射到对应的 ground_truth/anomaly/ 掩码（同名或 *_mask）。
    """
    anom_imgs = glob.glob(os.path.join(root_dir, "*", "test", "anomaly", "*.*"), recursive=True)
    pairs = []
    for img in anom_imgs:
        # 将 .../test/anomaly/... 映射为 .../ground_truth/anomaly/...
        # 保持后续子路径不变
        if os.sep + "test" + os.sep + "anomaly" + os.sep in img:
            gt_dir = img.split(os.sep + "test" + os.sep + "anomaly" + os.sep)[0] + os.sep + "ground_truth" + os.sep + "anomaly"
            m = find_mask_for_anomaly(img, gt_dir)
            if m is not None:
                pairs.append((img, m))
    return pairs
