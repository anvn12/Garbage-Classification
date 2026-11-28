"""
Hệ thống Xử lý Ảnh & Phân loại Rác thải Đa lớp (Gradio App - XLAV Lab 03)
- Nguồn đầu vào dùng chung (Image, Video, Webcam) cho cả 2 module.
- Module 1: Xử lý ảnh truyền thống từ MyCV.py (màu sắc, hình học, làm mịn).
- Module 2: Phân loại 5 loại rác thải bằng mạng MyCNN (best_mycnn.pth).
"""

import os
import sys
import time
from pathlib import Path
from typing import Tuple, Dict, Optional

import cv2
import numpy as np
from PIL import Image

import torch
import torchvision.transforms as transforms
import gradio as gr

# Thêm thư mục hiện tại vào sys.path để import được MyCV và MyClassifier dù chạy từ đâu
CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.append(str(CURRENT_DIR))

from MyCV import MyImageEditor
from MyClassifier import MyCNN

# Thư mục lưu ảnh/video xuất ra khi bấm Snapshot & Export
EXPORTS_DIR = CURRENT_DIR / "exports"
EXPORTS_DIR.mkdir(parents=True, exist_ok=True)


# setup

# Thứ tự 5 lớp chuẩn từ ImageFolder khi train trong notebook
CLASS_NAMES = ["cardboard", "glass", "metal", "paper", "plastic"]
CLASS_LABELS = ["Cardboard", "Glass", "Metal", "Paper", "Plastic"]

# Tự động chọn GPU nếu có, không thì chạy CPU
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Pipeline tiền xử lý ảnh đầu vào giống hệt val_transform trong xlav-lab3.ipynb
IMAGE_SIZE = 64
val_transform = transforms.Compose(
    [
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ]
)

# Khởi tạo model và nạp trọng số tốt nhất đã lưu
MODEL_PATH = CURRENT_DIR / "weights/best_mycnn.pth"
model = MyCNN(num_classes=len(CLASS_NAMES)).to(DEVICE)
model.load_state_dict(torch.load(str(MODEL_PATH), map_location=DEVICE))
model.eval()


# --- TIỆN ÍCH CHUNG CHO VIDEO ---


def extract_media_path(value) -> Optional[str]:
    """Lấy đường dẫn file từ giá trị gr.Video của nhiều phiên bản Gradio."""
    if value is None:
        return None

    # Một số component có thể trả về list/tuple hoặc FileData-like object.
    while isinstance(value, (list, tuple)):
        value = value[0] if value else None
        if value is None:
            return None

    if isinstance(value, dict):
        for key in ("path", "name", "video", "url"):
            candidate = value.get(key)
            if candidate:
                value = candidate
                break

    # Hỗ trợ đối tượng FileData có thuộc tính .path.
    if not isinstance(value, (str, Path)) and hasattr(value, "path"):
        value = value.path

    if value is None:
        return None

    path = str(value).strip()
    return path or None


def create_video_writer(
    out_path: str,
    fps: float,
    size: Tuple[int, int],
) -> Tuple[Optional[cv2.VideoWriter], Optional[str]]:
    """
    Tạo cv2.VideoWriter ưu tiên codec H.264 (avc1) vì trình duyệt/Gradio chỉ
    phát được H.264 (mp4v - MPEG-4 Part 2 sẽ hiển thị khung trống).
    Tự động fallback sang mp4v nếu H.264 không khả dụng.
    """
    width, height = size

    # H.264 (yuv420p) yêu cầu kích thước chẵn -> chỉ thử avc1 khi thỏa điều kiện
    codecs = ["avc1", "mp4v"] if (width % 2 == 0 and height % 2 == 0) else ["mp4v"]

    for codec in codecs:
        writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*codec), fps, size)
        if writer.isOpened():
            return writer, codec
        writer.release()

    return None, None


# --- MODULE 1 - các hàm xử lý ảnh cơ bản (gọi từ MyCV.py) ---


def apply_color_transformation(
    img: np.ndarray,
    method: str,
    hist_eq: bool = False,
    grayscale: bool = False,
    alpha: float = 1.0,
    beta: float = 0.0,
    c_log: float = 25.0,
) -> np.ndarray:
    """Áp dụng biến đổi màu sắc lên ảnh."""
    if img is None:
        return img

    # Gọi từng phép biến đổi tương ứng từ MyCV.py theo lựa chọn trong dropdown
    if method == "Linear Mapping":
        img = MyImageEditor.linear_mapping(img, alpha=alpha, beta=beta)
    elif method == "Logarithmic Mapping":
        img = MyImageEditor.logarithmic_mapping(img, _c=c_log)
    elif method == "Exponential Mapping":
        img = MyImageEditor.exponential_mapping(img)

    # Histogram Equalization là checkbox riêng, có thể bật độc lập hoặc kết hợp với method ở trên
    if hist_eq:
        img = MyImageEditor.histogram_equalization(img)

    # Grayscale là checkbox riêng; giữ 3 kênh (R=G=B) để các bước sau và hiển thị không lỗi
    if grayscale:
        if len(img.shape) == 2:
            gray = img
        else:
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        img = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)

    return img


def apply_geometric_transformation(
    img: np.ndarray,
    angle: float = 0.0,
    scale: float = 1.0,
    tx: float = 0.0,
    ty: float = 0.0,
    flip_mode: str = "None",
) -> np.ndarray:
    """Áp dụng biến đổi hình học (xoay, thu phóng, tịnh tiến, lật)."""
    if img is None:
        return None

    h, w = img.shape[:2]

    # Chỉ tính ma trận và warp khi người dùng có thay đổi góc, tỉ lệ hoặc độ dời
    if angle != 0 or scale != 1.0 or tx != 0 or ty != 0:
        center = (w / 2.0, h / 2.0)
        M = MyImageEditor.get_affine_matrix(
            center, angle=angle, scale=scale, tx=tx, ty=ty
        )
        img = MyImageEditor.warpAffine(img, M, dsize=(w, h))

    # Lật ảnh nếu người dùng chọn chế độ lật
    if flip_mode != "None":
        img = MyImageEditor.flip(img, flip_mode=flip_mode)

    return img


def apply_image_smoothing(
    img: np.ndarray,
    smooth_type: str = "None",
    ksize: int = 3,
    sigma: float = 1.0,
) -> np.ndarray:
    """Áp dụng bộ lọc làm mịn ảnh (Average, Gaussian, Median) từ MyImageEditor."""
    # Bỏ qua nếu không chọn bộ lọc hoặc kernel <= 1
    if img is None or smooth_type == "None" or smooth_type is None or int(ksize) <= 1:
        return img

    if smooth_type == "Average Blur":
        return MyImageEditor.blur(img, ksize=int(ksize))
    elif smooth_type == "Gaussian Blur":
        return MyImageEditor.GaussianBlur(img, ksize=int(ksize), sigma=float(sigma))
    elif smooth_type == "Median Blur":
        return MyImageEditor.medianBlur(img, ksize=int(ksize))

    return img


def process_module1_pipeline(
    img: np.ndarray,
    color_method: str,
    hist_eq: bool,
    grayscale: bool,
    alpha: float,
    beta: float,
    c_log: float,
    angle: float,
    scale: float,
    tx: float,
    ty: float,
    flip_mode: str,
    smooth_type: str,
    ksize: int,
    sigma: float,
) -> Optional[np.ndarray]:
    """Chạy nối tiếp 3 bước xử lý cho 1 khung hình: Màu -> Hình học -> Làm mịn."""
    if img is None:
        return None

    # Thực hiện lần lượt từng bước xử lý
    res = apply_color_transformation(
        img, color_method, hist_eq, grayscale, alpha, beta, c_log
    )
    res = apply_geometric_transformation(res, angle, scale, tx, ty, flip_mode)
    res = apply_image_smoothing(res, smooth_type, ksize, sigma)
    return res


def process_module1_video(
    video_path: str,
    color_method: str,
    hist_eq: bool,
    grayscale: bool,
    alpha: float,
    beta: float,
    c_log: float,
    angle: float,
    scale: float,
    tx: float,
    ty: float,
    flip_mode: str,
    smooth_type: str,
    ksize: int,
    sigma: float,
) -> Tuple[Optional[str], Optional[str], Optional[str], str]:
    """Xử lý tất cả frame của video bằng pipeline Module 1 và xuất MP4."""
    video_path = extract_media_path(video_path)
    if not video_path or not os.path.exists(video_path):
        return None, None, None, f"Invalid video path: {video_path}"

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return (
            None,
            None,
            None,
            "OpenCV cannot open the video. Check codec/FFmpeg support.",
        )

    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 0:
        fps = 25.0

    ret, first_frame = cap.read()
    if not ret or first_frame is None:
        cap.release()
        return None, None, None, "Cannot read the first frame from the video."

    height, width = first_frame.shape[:2]
    out_path = str(EXPORTS_DIR / f"module1_video_{time.time_ns()}.mp4")
    writer, codec = create_video_writer(out_path, fps, (width, height))
    if writer is None:
        cap.release()
        return (
            None,
            None,
            None,
            "Cannot create video writer. No usable codec available.",
        )

    frame_count = 0
    t_start = time.perf_counter()
    frame_bgr = first_frame

    try:
        # Đọc đến EOF, không giới hạn 300 frame để không cắt cụt video dài.
        while frame_bgr is not None:
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            processed_rgb = process_module1_pipeline(
                frame_rgb,
                color_method,
                hist_eq,
                grayscale,
                alpha,
                beta,
                c_log,
                angle,
                scale,
                tx,
                ty,
                flip_mode,
                smooth_type,
                ksize,
                sigma,
            )
            if processed_rgb is None:
                processed_rgb = frame_rgb

            writer.write(cv2.cvtColor(processed_rgb, cv2.COLOR_RGB2BGR))
            frame_count += 1

            ret, next_frame = cap.read()
            frame_bgr = next_frame if ret else None
    finally:
        cap.release()
        writer.release()

    elapsed = time.perf_counter() - t_start
    if frame_count == 0:
        return (
            None,
            None,
            None,
            "No frames were processed. The video may be empty or unreadable.",
        )

    status = (
        f"Processed all {frame_count} frames in {elapsed:.2f}s "
        f"(codec: {codec}). Exported: {Path(out_path).name}"
    )
    return video_path, out_path, out_path, status


# --- MODULE 2 - dự đoán 5 class (dùng MyCNN) ---


@torch.no_grad()
def predict_single_image(
    img_input: np.ndarray,
) -> Tuple[str, float, float, Dict[str, float], Optional[np.ndarray]]:
    """
    Dự đoán phân loại một ảnh bằng model MyCNN:
      - Chuẩn hóa ảnh theo val_transform
      - Tính thời gian suy luận (ms)
      - Trả về Top-1 label, độ tin cậy, độ trễ, dict xác suất (kèm 1 số thập phân) và ảnh gắn nhãn
    """
    if img_input is None:
        return "No input image", 0.0, 0.0, {}, None

    start_time = time.perf_counter()

    # Đổi sang PIL RGB, áp dụng transform và thêm chiều batch [1, 3, 64, 64]
    pil_img = Image.fromarray(img_input.astype(np.uint8)).convert("RGB")
    tensor = val_transform(pil_img).unsqueeze(0).to(DEVICE)

    # Đưa qua mạng CNN và dùng Softmax để lấy phân phối xác suất
    outputs = model(tensor)
    probs = torch.softmax(outputs, dim=1).squeeze(0).cpu().numpy()

    proc_time_ms = round((time.perf_counter() - start_time) * 1000.0, 2)

    # Sắp xếp các lớp theo xác suất giảm dần
    sorted_indices = np.argsort(probs)[::-1]
    top1_idx = int(sorted_indices[0])
    top1_label = CLASS_LABELS[top1_idx]
    top1_conf = round(float(probs[top1_idx]) * 100.0, 1)

    # Ghi kèm phần trăm 1 chữ số thập phân (VD: 46.6%) vào tên nhãn vì gr.Label mặc định làm tròn số nguyên
    sorted_probs = {
        f"{CLASS_LABELS[i]} ({float(probs[i]) * 100.0:.1f}%)": float(probs[i])
        for i in sorted_indices
    }

    # Vẽ dải đen ở cạnh trên ảnh và in nhãn Top-1 + độ tin cậy lên
    annotated_img = img_input.copy()
    h, w = annotated_img.shape[:2]
    banner_h = max(36, int(h * 0.09))
    cv2.rectangle(annotated_img, (0, 0), (w, banner_h), (20, 20, 20), -1)
    caption = f"{top1_label}: {top1_conf:.1f}% ({proc_time_ms:.1f} ms)"
    font_scale = max(0.55, min(1.1, w / 620.0))
    cv2.putText(
        annotated_img,
        caption,
        (12, int(banner_h * 0.7)),
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        (0, 255, 127),
        2,
        cv2.LINE_AA,
    )

    return top1_label, top1_conf, proc_time_ms, sorted_probs, annotated_img


@torch.no_grad()
def process_video_ai(
    video_path: str,
    batch_size: Optional[int] = None,
) -> Tuple[str, float, float, Dict[str, float], Optional[str], str]:
    """
    Dự đoán tất cả frame của video bằng MyCNN.

    - Mỗi frame được phân loại độc lập.
    - Video-level prediction được tính bằng trung bình xác suất (softmax) của
      tất cả frame, không phải majority vote.
    - Video đầu ra hiển thị nhãn dự đoán riêng của từng frame.

    Returns:
        top1_label, mean_confidence_percent, avg_prediction_ms_per_frame,
        mean_probability_dict, annotated_video_path, summary
    """
    video_path = extract_media_path(video_path)
    if not video_path or not os.path.exists(video_path):
        return "No video input", 0.0, 0.0, {}, None, f"Invalid video path: {video_path}"

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return (
            "Prediction error",
            0.0,
            0.0,
            {},
            None,
            (
                "OpenCV cannot open the video. Check that the file is a supported video "
                "and that OpenCV/FFmpeg can decode it."
            ),
        )

    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 0:
        fps = 25.0

    ret, first_frame = cap.read()
    if not ret or first_frame is None:
        cap.release()
        return (
            "Prediction error",
            0.0,
            0.0,
            {},
            None,
            "Cannot read the first frame from the video.",
        )

    height, width = first_frame.shape[:2]
    out_path = str(EXPORTS_DIR / f"ai_video_prediction_{time.time_ns()}.mp4")
    writer, codec = create_video_writer(out_path, fps, (width, height))
    if writer is None:
        cap.release()
        return (
            "Prediction error",
            0.0,
            0.0,
            {},
            None,
            ("Cannot create the output video. No usable MP4 codec is available."),
        )

    # Batch inference giúp giảm overhead so với gọi model riêng từng frame.
    # Tất cả frame vẫn được dự đoán; batch_size chỉ là số frame xử lý cùng lúc.
    if batch_size is None:
        batch_size = 16 if DEVICE.type == "cpu" else 32
    batch_size = max(1, int(batch_size))

    frames_rgb = [cv2.cvtColor(first_frame, cv2.COLOR_BGR2RGB)]
    frame_count = 0
    sum_probs = np.zeros(len(CLASS_NAMES), dtype=np.float64)
    prediction_time_ms_total = 0.0
    t_start = time.perf_counter()

    def flush_batch():
        """Dự đoán một batch, ghi nhãn từng frame và cộng xác suất vào tổng."""
        nonlocal frame_count, sum_probs, prediction_time_ms_total, frames_rgb
        if not frames_rgb:
            return

        batch_start = time.perf_counter()
        pil_images = [Image.fromarray(frame).convert("RGB") for frame in frames_rgb]
        batch_tensor = torch.stack([val_transform(image) for image in pil_images]).to(
            DEVICE
        )
        logits = model(batch_tensor)
        batch_probs = torch.softmax(logits, dim=1).cpu().numpy()
        prediction_time_ms_total += (time.perf_counter() - batch_start) * 1000.0

        # Cộng xác suất từng frame để tính mean probability cuối video.
        sum_probs += batch_probs.sum(axis=0)

        for frame_rgb, probs in zip(frames_rgb, batch_probs):
            top_idx = int(np.argmax(probs))
            frame_label = CLASS_LABELS[top_idx]
            frame_conf = float(probs[top_idx]) * 100.0

            # Overlay trên video thể hiện dự đoán riêng của frame hiện tại.
            annotated = frame_rgb.copy()
            h, w = annotated.shape[:2]
            banner_h = max(36, int(h * 0.09))
            cv2.rectangle(annotated, (0, 0), (w, banner_h), (20, 20, 20), -1)
            caption = f"Frame {frame_count + 1} | {frame_label}: {frame_conf:.1f}%"
            font_scale = max(0.55, min(1.1, w / 620.0))
            cv2.putText(
                annotated,
                caption,
                (12, int(banner_h * 0.7)),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                (0, 255, 127),
                2,
                cv2.LINE_AA,
            )
            writer.write(cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR))
            frame_count += 1

        frames_rgb = []

    try:
        # Đọc đến EOF. Không dùng CAP_PROP_FRAME_COUNT để giới hạn số frame,
        # vì metadata frame count của một số video không chính xác.
        while True:
            if len(frames_rgb) >= batch_size:
                flush_batch()

            ret, frame_bgr = cap.read()
            if not ret or frame_bgr is None:
                break
            frames_rgb.append(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))

        flush_batch()
    except Exception:
        # Đảm bảo giải phóng tài nguyên trước khi chuyển lỗi lên Gradio.
        cap.release()
        writer.release()
        raise
    finally:
        cap.release()
        writer.release()

    elapsed = time.perf_counter() - t_start
    if frame_count == 0:
        return (
            "Prediction error",
            0.0,
            0.0,
            {},
            None,
            ("No frames were processed. The video may be empty or unreadable."),
        )

    # Video-level result: mean của vector xác suất 5 lớp trên toàn bộ frame.
    mean_probs = (sum_probs / frame_count).astype(np.float64)
    sorted_indices = np.argsort(mean_probs)[::-1]
    top1_idx = int(sorted_indices[0])
    top1_label = CLASS_LABELS[top1_idx]
    top1_conf = round(float(mean_probs[top1_idx]) * 100.0, 1)

    mean_probability_dict = {
        f"{CLASS_LABELS[i]} ({float(mean_probs[i]) * 100.0:.1f}%)": float(mean_probs[i])
        for i in sorted_indices
    }

    avg_prediction_ms = round(prediction_time_ms_total / frame_count, 2)
    avg_fps = frame_count / max(elapsed, 0.001)
    probability_text = ", ".join(
        f"{CLASS_LABELS[i]} {float(mean_probs[i]) * 100.0:.1f}%" for i in sorted_indices
    )
    summary = (
        f"Video-level prediction (mean probability): {top1_label} ({top1_conf:.1f}%). "
        f"Processed all {frame_count} frames in {elapsed:.2f}s "
        f"(throughput: {avg_fps:.1f} FPS; average prediction time: "
        f"{avg_prediction_ms:.2f} ms/frame; codec: {codec}). "
        f"Mean probabilities: {probability_text}. "
        f"Saved: {Path(out_path).name}"
    )

    return (
        top1_label,
        top1_conf,
        avg_prediction_ms,
        mean_probability_dict,
        out_path,
        summary,
    )


# HÀM LƯU & XUẤT KẾT QUẢ (SNAPSHOT & EXPORT)


def save_and_export_image(
    image: np.ndarray, prefix: str = "snapshot"
) -> Tuple[Optional[str], str]:
    """Lưu ảnh kết quả hiện tại thành file PNG trong thư mục exports/ để tải về."""
    if image is None:
        return None, "No result image available to export."

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    file_name = f"{prefix}_{timestamp}.png"
    file_path = EXPORTS_DIR / file_name

    # Chuyển RGB -> BGR nếu là ảnh 3 kênh màu trước khi ghi bằng OpenCV
    if len(image.shape) == 3 and image.shape[2] == 3:
        cv2.imwrite(
            str(file_path), cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_RGB2BGR)
        )
    else:
        cv2.imwrite(str(file_path), image.astype(np.uint8))

    return str(file_path), f"Saved: {file_name}"


# gradio UI
def build_app() -> gr.Blocks:
    """Dựng giao diện Gradio hợp nhất: Input dùng chung -> Xử lý ảnh -> Dự đoán AI cả ảnh gốc & ảnh đã xử lý."""
    with gr.Blocks(title="Image Processing & Waste Classification") as demo:
        gr.Markdown("# Image Processing & Multi-Class Waste Classification")

        # ---------------------------------------------------------------------
        # 1. NGUỒN ĐẦU VÀO DÙNG CHUNG (IMAGE / VIDEO / WEBCAM)
        # ---------------------------------------------------------------------
        with gr.Group():
            input_mode = gr.Radio(
                choices=["Image", "Video", "Webcam"],
                value="Image",
                label="Input Source",
            )
            shared_image = gr.Image(
                label="Upload Image (.jpg, .png)", type="numpy", visible=True
            )
            shared_video = gr.Video(label="Upload Video (.mp4)", visible=False)
            shared_webcam = gr.Image(
                label="Webcam Stream",
                sources=["webcam"],
                streaming=True,
                type="numpy",
                visible=False,
            )

        # ---------------------------------------------------------------------
        # 2. KHU VỰC XỬ LÝ ẢNH (MODULE 1) & SO SÁNH ORIGINAL VS PROCESSED
        # ---------------------------------------------------------------------
        with gr.Row():
            # Cột trái: Các thanh điều khiển xử lý ảnh cơ bản
            with gr.Column(scale=4):
                with gr.Accordion("Color Transformation", open=True):
                    m1_color_method = gr.Dropdown(
                        choices=[
                            "None",
                            "Linear Mapping",
                            "Logarithmic Mapping",
                            "Exponential Mapping",
                        ],
                        value="None",
                        label="Color Method",
                    )
                    m1_alpha = gr.Slider(
                        0.1, 3.0, value=1.0, step=0.1, label="Alpha (Contrast)"
                    )
                    m1_beta = gr.Slider(
                        -100.0, 100.0, value=0.0, step=5.0, label="Beta (Brightness)"
                    )
                    m1_clog = gr.Slider(
                        1.0,
                        50.0,
                        value=25.0,
                        step=1.0,
                        label="Constant c (Logarithmic)",
                    )
                    m1_hist_eq = gr.Checkbox(
                        label="Histogram Equalization", value=False
                    )
                    m1_grayscale = gr.Checkbox(label="Grayscale", value=False)

                with gr.Accordion("Geometric Transformation", open=True):
                    m1_angle = gr.Slider(
                        -180, 180, value=0, step=5, label="Rotation Angle (deg)"
                    )
                    m1_scale = gr.Slider(0.2, 2.5, value=1.0, step=0.05, label="Scale")
                    m1_tx = gr.Slider(
                        -150, 150, value=0, step=5, label="Translation X (px)"
                    )
                    m1_ty = gr.Slider(
                        -150, 150, value=0, step=5, label="Translation Y (px)"
                    )
                    m1_flip = gr.Radio(
                        choices=["None", "Horizontal", "Vertical", "Both"],
                        value="None",
                        label="Flip",
                    )

                with gr.Accordion("Image Smoothing", open=True):
                    m1_smooth_type = gr.Dropdown(
                        choices=[
                            "None",
                            "Average Blur",
                            "Gaussian Blur",
                            "Median Blur",
                        ],
                        value="None",
                        label="Smoothing Filter",
                    )
                    m1_ksize = gr.Slider(1, 25, value=3, step=2, label="Kernel Size")
                    m1_sigma = gr.Slider(
                        0.1, 10.0, value=1.0, step=0.2, label="Sigma (Gaussian)"
                    )

                with gr.Row():
                    m1_btn_apply = gr.Button("Apply Processing", variant="secondary")
                    m1_btn_reset = gr.Button("Reset")

            # Cột phải: Hiển thị song song Original vs Processed và khu vực dự đoán AI ngay bên dưới
            with gr.Column(scale=6):
                with gr.Row(visible=True) as m1_img_row:
                    m1_view_orig_img = gr.Image(
                        label="Original Image", interactive=False
                    )
                    m1_view_proc_img = gr.Image(
                        label="Processed Image", interactive=False
                    )

                with gr.Row(visible=False) as m1_vid_row:
                    m1_view_orig_vid = gr.Video(
                        label="Original Video", interactive=False
                    )
                    m1_view_proc_vid = gr.Video(
                        label="Processed Video", interactive=False
                    )

                # Nút Predict đặt ngay dưới 2 khung Original và Processed để dự đoán cả 2 phiên bản
                btn_predict_both = gr.Button(
                    "Predict Original & Processed", variant="primary"
                )

                # -------------------------------------------------------------
                # KẾT QUẢ DỰ ĐOÁN AI (MODULE 2 - ĐẶT NGAY DƯỚI CỘT 2)
                # -------------------------------------------------------------
                gr.Markdown("### AI Prediction Comparison (Original vs. Processed)")
                gr.Markdown(
                    "For video: Top-1 and probabilities are averaged across all decoded frames; the annotated video shows each frame's own prediction."
                )
                with gr.Row():
                    # Cột kết quả cho bản Original
                    with gr.Column():
                        gr.Markdown("#### Original Input Prediction")
                        with gr.Row():
                            orig_top1 = gr.Textbox(
                                label="Top-1 Label", interactive=False
                            )
                            orig_conf = gr.Number(
                                label="Confidence (%)", interactive=False
                            )
                            orig_time = gr.Number(label="Time (ms)", interactive=False)
                        orig_label_chart = gr.Label(
                            num_top_classes=5, label="Probability Distribution"
                        )
                        with gr.Column(visible=True) as orig_img_col:
                            orig_ann_img = gr.Image(
                                label="Annotated Original", interactive=False
                            )
                        with gr.Column(visible=False) as orig_vid_col:
                            orig_ann_vid = gr.Video(
                                label="Annotated Original Video", interactive=False
                            )
                            orig_vid_stats = gr.Textbox(
                                label="Video Summary", interactive=False
                            )
                        orig_btn_export = gr.Button("Snapshot & Export (Original)")
                        orig_export_file = gr.File(
                            label="Download Original Prediction", height=90
                        )

                    # Cột kết quả cho bản Processed
                    with gr.Column():
                        gr.Markdown("#### Processed Input Prediction")
                        with gr.Row():
                            proc_top1 = gr.Textbox(
                                label="Top-1 Label", interactive=False
                            )
                            proc_conf = gr.Number(
                                label="Confidence (%)", interactive=False
                            )
                            proc_time = gr.Number(label="Time (ms)", interactive=False)
                        proc_label_chart = gr.Label(
                            num_top_classes=5, label="Probability Distribution"
                        )
                        with gr.Column(visible=True) as proc_img_col:
                            proc_ann_img = gr.Image(
                                label="Annotated Processed", interactive=False
                            )
                        with gr.Column(visible=False) as proc_vid_col:
                            proc_ann_vid = gr.Video(
                                label="Annotated Processed Video", interactive=False
                            )
                            proc_vid_stats = gr.Textbox(
                                label="Video Summary", interactive=False
                            )
                        proc_btn_export = gr.Button("Snapshot & Export (Processed)")
                        proc_export_file = gr.File(
                            label="Download Processed Prediction", height=90
                        )

        # ---------------------------------------------------------------------
        # 4. KẾT NỐI SỰ KIỆN CHO GIAO DIỆN HỢP NHẤT
        # ---------------------------------------------------------------------
        def toggle_shared_input(mode):
            """Ẩn/hiện các khung nhập và hiển thị theo chế độ Image, Video hoặc Webcam."""
            is_img = mode == "Image"
            is_vid = mode == "Video"
            is_cam = mode == "Webcam"
            return (
                gr.update(visible=is_img),
                gr.update(visible=is_vid),
                gr.update(visible=is_cam),
                gr.update(visible=is_img or is_cam),
                gr.update(visible=is_vid),
                gr.update(visible=is_img or is_cam),
                gr.update(visible=is_vid),
                gr.update(visible=is_img or is_cam),
                gr.update(visible=is_vid),
            )

        input_mode.change(
            fn=toggle_shared_input,
            inputs=[input_mode],
            outputs=[
                shared_image,
                shared_video,
                shared_webcam,
                m1_img_row,
                m1_vid_row,
                orig_img_col,
                orig_vid_col,
                proc_img_col,
                proc_vid_col,
            ],
        )

        m1_params = [
            m1_color_method,
            m1_hist_eq,
            m1_grayscale,
            m1_alpha,
            m1_beta,
            m1_clog,
            m1_angle,
            m1_scale,
            m1_tx,
            m1_ty,
            m1_flip,
            m1_smooth_type,
            m1_ksize,
            m1_sigma,
        ]

        def run_m1_apply(mode, img, vid, cam, *params):
            """Chạy xử lý Module 1 cho nguồn đầu vào đang chọn."""
            if mode == "Video":
                orig_v, proc_v, _, _ = process_module1_video(vid, *params)
                return None, None, orig_v, proc_v

            active_img = img if mode == "Image" else cam
            if active_img is None:
                return None, None, None, None
            return active_img, process_module1_pipeline(active_img, *params), None, None

        m1_btn_apply.click(
            fn=run_m1_apply,
            inputs=[input_mode, shared_image, shared_video, shared_webcam, *m1_params],
            outputs=[
                m1_view_orig_img,
                m1_view_proc_img,
                m1_view_orig_vid,
                m1_view_proc_vid,
            ],
        )

        def _predict_pair(orig_img, proc_img):
            """Hàm phụ chạy dự đoán AI cho cả ảnh gốc và ảnh đã xử lý cùng lúc."""
            o_t1, o_c, o_ms, o_p, o_ann = predict_single_image(orig_img)
            p_t1, p_c, p_ms, p_p, p_ann = predict_single_image(proc_img)
            return o_t1, o_c, o_ms, o_p, o_ann, p_t1, p_c, p_ms, p_p, p_ann

        pred_img_outputs = [
            orig_top1,
            orig_conf,
            orig_time,
            orig_label_chart,
            orig_ann_img,
            proc_top1,
            proc_conf,
            proc_time,
            proc_label_chart,
            proc_ann_img,
        ]

        def run_m1_interactive(mode, img, *params):
            """Tự động cập nhật ảnh Original và Processed, đồng thời dự đoán AI khi đổi ảnh hoặc kéo slider."""
            if mode != "Image" or img is None:
                # Trả về gr.update() cho tất cả outputs khi không ở chế độ Image hoặc chưa có ảnh
                return (gr.update(),) * (2 + len(pred_img_outputs))
            proc = process_module1_pipeline(img, *params)
            return (img, proc, *_predict_pair(img, proc))

        for ctrl in [shared_image, *m1_params]:
            ctrl.change(
                fn=run_m1_interactive,
                inputs=[input_mode, shared_image, *m1_params],
                outputs=[m1_view_orig_img, m1_view_proc_img, *pred_img_outputs],
            )

        def run_m1_stream(cam, *params):
            """Xử lý và dự đoán AI thời gian thực từng khung hình từ Webcam."""
            if cam is None:
                # Trả về None cho tất cả outputs khi chưa có khung hình
                return (None,) * (2 + len(pred_img_outputs))
            proc = process_module1_pipeline(cam, *params)
            return (cam, proc, *_predict_pair(cam, proc))

        shared_webcam.stream(
            fn=run_m1_stream,
            inputs=[shared_webcam, *m1_params],
            outputs=[m1_view_orig_img, m1_view_proc_img, *pred_img_outputs],
        )

        def reset_m1_controls():
            """Đặt lại toàn bộ thông số xử lý ảnh về mặc định."""
            return (
                "None",
                False,
                False,
                1.0,
                0.0,
                25.0,
                0,
                1.0,
                0,
                0,
                "None",
                "None",
                3,
                1.0,
            )

        m1_btn_reset.click(fn=reset_m1_controls, outputs=m1_params)

        def run_predict_both(mode, orig_img, proc_img, orig_vid, proc_vid, shared_vid):
            """Dự đoán cả hai đầu vào; với video trả về xác suất trung bình toàn frame."""
            if mode == "Video":
                original_source = extract_media_path(orig_vid) or extract_media_path(
                    shared_vid
                )
                if not original_source:
                    empty = (
                        "No video input",
                        0.0,
                        0.0,
                        {},
                        None,
                        "Please upload a video first.",
                    )
                    o_result = empty
                    p_result = empty
                else:
                    try:
                        o_result = process_video_ai(original_source)
                    except Exception as exc:
                        o_result = (
                            "Prediction error",
                            0.0,
                            0.0,
                            {},
                            None,
                            f"Original video prediction failed: {type(exc).__name__}: {exc}",
                        )

                    processed_source = extract_media_path(proc_vid)
                    if processed_source:
                        try:
                            p_result = process_video_ai(processed_source)
                        except Exception as exc:
                            p_result = (
                                "Prediction error",
                                0.0,
                                0.0,
                                {},
                                None,
                                f"Processed video prediction failed: {type(exc).__name__}: {exc}",
                            )
                    else:
                        # Tương tự nhánh Image: nếu chưa Apply Processing thì dùng bản gốc.
                        # Tái sử dụng kết quả để tránh chạy CNN hai lần trên cùng một video.
                        p_result = (
                            o_result[0],
                            o_result[1],
                            o_result[2],
                            o_result[3],
                            o_result[4],
                            "No processed video was generated; the original video result is reused. "
                            + o_result[5],
                        )

                o_t1, o_c, o_ms, o_p, o_vid, o_sum = o_result
                p_t1, p_c, p_ms, p_p, p_vid, p_sum = p_result
                return (
                    o_t1,
                    o_c,
                    o_ms,
                    o_p,
                    None,
                    o_vid,
                    o_sum,
                    p_t1,
                    p_c,
                    p_ms,
                    p_p,
                    None,
                    p_vid,
                    p_sum,
                )

            # Chế độ Image hoặc Webcam: suy luận trên cả ảnh gốc và ảnh đã qua xử lý.
            o_t1, o_c, o_ms, o_p, o_ann = predict_single_image(orig_img)
            target_proc = proc_img if proc_img is not None else orig_img
            p_t1, p_c, p_ms, p_p, p_ann = predict_single_image(target_proc)
            return (
                o_t1,
                o_c,
                o_ms,
                o_p,
                o_ann,
                None,
                "",
                p_t1,
                p_c,
                p_ms,
                p_p,
                p_ann,
                None,
                "",
            )

        btn_predict_both.click(
            fn=run_predict_both,
            inputs=[
                input_mode,
                m1_view_orig_img,
                m1_view_proc_img,
                m1_view_orig_vid,
                m1_view_proc_vid,
                shared_video,
            ],
            outputs=[
                orig_top1,
                orig_conf,
                orig_time,
                orig_label_chart,
                orig_ann_img,
                orig_ann_vid,
                orig_vid_stats,
                proc_top1,
                proc_conf,
                proc_time,
                proc_label_chart,
                proc_ann_img,
                proc_ann_vid,
                proc_vid_stats,
            ],
        )

        def export_prediction(mode, ann_img, ann_vid, prefix):
            """Lưu và xuất ảnh/video đã gắn nhãn dự đoán, trả về đường dẫn file để tải."""
            if mode == "Video":
                return ann_vid or None
            export_path, _ = save_and_export_image(ann_img, prefix=prefix)
            return export_path

        orig_btn_export.click(
            fn=lambda mode, img, vid: export_prediction(mode, img, vid, "ai_original"),
            inputs=[input_mode, orig_ann_img, orig_ann_vid],
            outputs=[orig_export_file],
        )
        proc_btn_export.click(
            fn=lambda mode, img, vid: export_prediction(mode, img, vid, "ai_processed"),
            inputs=[input_mode, proc_ann_img, proc_ann_vid],
            outputs=[proc_export_file],
        )

    return demo


if __name__ == "__main__":
    demo = build_app()
    # Bật queue để xử lý video (nặng, chạy lâu) không bị timeout / treo UI
    demo.queue()
    demo.launch(server_name="127.0.0.1", server_port=7860, share=False, show_error=True)
