import numpy as np
import cv2


class MyImageEditor:
    """
    Class ở lab 1
    """

    def __init__(self):
        return

    # --------- Color transformations ---------
    ## Linear mapping
    @staticmethod
    def linear_mapping(src, alpha=1.0, beta=0.0):
        """
        Ánh xạ tuyến tính: g(x, y) = alpha * f(x, y) + beta
        - alpha: điều chỉnh độ tương phản (contrast)
        - beta: điều chỉnh độ sáng (brightness)
        """
        result_image = alpha * src.astype(np.float32) + beta
        return np.clip(result_image, 0, 255).astype(np.uint8)

    ## Non-linear mapping
    @staticmethod
    def logarithmic_mapping(src, _c=1.0):
        """
        g(x, y) = _c * log f(x, y)

        Công thức gốc bị số 0 -> lỗi log -> cộng hết lên 1 thì 0->1 => log(1) = 0, ko lỗi nữa
        g(x, y) = _c * log(1 + f(x, y))
        """
        result_image = _c * np.log(1.0 + src.astype(np.float32))
        return np.clip(result_image, 0, 255).astype(np.uint8)

    @staticmethod
    def exponential_mapping(src):
        """
        Ánh xạ hàm mũ: chuẩn hóa pixel về [0, 1] trước khi lấy e^x
        rồi kéo dãn lại về [0, 255] để tránh tràn số.
        """
        normalized_src = src.astype(np.float32) / 255.0
        new_pixel = ((np.exp(normalized_src) - 1.0) / (np.e - 1.0)) * 255.0
        return np.clip(new_pixel, 0, 255).astype(np.uint8)

    ## Probability Density Function-based mapping

    ### Histogram equalization
    @staticmethod
    def histogram_equalization(src):
        """
        Cân bằng histogram lần lượt từng kênh màu R, G, B bằng OpenCV:
        - Nếu ảnh xám thì cân bằng trực tiếp.
        - Nếu ảnh màu RGB thì tách từng kênh ra cân bằng rồi gộp lại.
        """
        if len(src.shape) == 2:
            return cv2.equalizeHist(src)

        channels = cv2.split(src)
        eq_channels = [cv2.equalizeHist(ch) for ch in channels]
        return cv2.merge(eq_channels)

    @staticmethod
    def get_affine_matrix(center, angle=0.0, scale=1.0, tx=0.0, ty=0.0):
        """
        Tạo ma trận biến đổi Affine 2x3 gồm xoay quanh tâm, thu phóng và tịnh tiến.
        """
        M = cv2.getRotationMatrix2D(center, angle, scale)
        M[0, 2] += tx
        M[1, 2] += ty
        return M

    @staticmethod
    def warpAffine(src, M, dsize=None, flags=cv2.INTER_LINEAR):
        """
        Thực hiện phép biến đổi Affine lên ảnh bằng OpenCV.
        """
        if dsize is None:
            dsize = (src.shape[1], src.shape[0])
        return cv2.warpAffine(src, M, tuple(dsize), flags=flags)

    @staticmethod
    def flip(src, flip_mode="None"):
        """
        Lật ảnh theo chiều ngang, chiều dọc hoặc cả hai chiều.
        """
        if flip_mode == "Horizontal":
            return cv2.flip(src, 1)
        elif flip_mode == "Vertical":
            return cv2.flip(src, 0)
        elif flip_mode == "Both":
            return cv2.flip(src, -1)
        return src

    # --------- Thuật toán làm mịn ảnh (Image smoothing) ---------

    @staticmethod
    def _parse_ksize(ksize):
        """Đảm bảo kích thước kernel luôn là cặp số nguyên lẻ dương (kx, ky)."""
        if isinstance(ksize, (int, np.integer)):
            k = max(1, int(ksize))
            if k % 2 == 0:
                k += 1
            return (k, k)
        kx = max(1, int(ksize[0]))
        ky = max(1, int(ksize[1]))
        if kx % 2 == 0:
            kx += 1
        if ky % 2 == 0:
            ky += 1
        return (kx, ky)

    # Average filter
    @staticmethod
    def blur(src, ksize=3):
        """Lọc trung bình (Average filter)."""
        kx, ky = MyImageEditor._parse_ksize(ksize)
        return cv2.blur(src, (kx, ky))

    @staticmethod
    def GaussianBlur(src, ksize=3, sigma=1.0):
        """Lọc Gauss (Gaussian filter) với độ lệch chuẩn sigma."""
        kx, ky = MyImageEditor._parse_ksize(ksize)
        return cv2.GaussianBlur(src, (kx, ky), sigmaX=float(sigma), sigmaY=float(sigma))

    @staticmethod
    def medianBlur(src, ksize=3):
        """Lọc trung vị (Median filter) giúp khử nhiễu muối tiêu."""
        kx, _ = MyImageEditor._parse_ksize(ksize)
        return cv2.medianBlur(src, kx)

    # --------- Phát hiện biên (Edge detection) ---------

    @staticmethod
    def Sobel(src):
        """Toán tử Sobel phát hiện biên theo 2 hướng x, y."""
        gx = cv2.Sobel(src, cv2.CV_64F, 1, 0, ksize=3)
        gy = cv2.Sobel(src, cv2.CV_64F, 0, 1, ksize=3)
        mag = np.sqrt(gx**2 + gy**2)
        return np.clip(mag, 0, 255).astype(np.uint8)

    @staticmethod
    def Laplacian(src):
        """Toán tử Laplace phát hiện biên bậc 2."""
        lap = cv2.Laplacian(src, cv2.CV_64F, ksize=3)
        return np.clip(np.abs(lap), 0, 255).astype(np.uint8)

    @staticmethod
    def Canny(src, threshold1=100, threshold2=200):
        """Thuật toán phát hiện biên Canny."""
        return cv2.Canny(src, threshold1, threshold2)
