# Dataset description

## Dữ liệu gốc (Original Dataset)

Source: <https://www.kaggle.com/datasets/asdasdasasdas/garbage-Classification>

Tập dữ liệu được sử dụng cho bài toán multi-Class image classification với 6 lớp: cardboard (393 ảnh), glass (491 ảnh), metal (400 ảnh), paper(584 ảnh), plastic (472 ảnh) and trash(127 ảnh).

## Quy trình xử lý dữ liệu (Preprocessing Pipeline)

Quy trình tiền xử lý được tổ chức trong thư mục `preprocessing/` bao gồm 3 giai đoạn chính:

### Xử lý tập dữ liệu gốc

1. Di chuyển các thư mục ảnh (cardboard/, glass/, metal/, paper/, plastic/, trash/) từ thư mục dataset gốc `Garbage-Classification/Garbage classification/Garbage classification` ra ngoài `Garbage-Classification/` và xóa các thư mục rỗng còn lại.
2. Xóa các file text cũ trong `Garbage-Classification/` (nếu có) để tránh nhầm lẫn với các file text vector hóa được tạo ra sau này.
3. Xóa bỏ thư mục trash do có nhiều ảnh trùng class với các lớp khác (ví dụ trash là plastic) và số ảnh ít hơn nhiều so với các class còn lại. Chỉ giữ lại 5 lớp: cardboard, glass, metal, paper, plastic.

### Chia tập dữ liệu thành train/val/test

Phần này dùng lại code từ Lab2 `preprocessing/dataset_split.py` để chia tập dữ liệu thành 3 tập train, val, test theo tỷ lệ 8:1:1.

1. Chia tập dữ liệu thành 3 tập: train (80%), val (10%), test (10%) theo tỷ lệ 8:1:1. Mỗi thư mục gồm 5 thư mục con tương ứng với 5 class: cardboard, glass, metal, paper, plastic.

Kết quả thu được sẽ là

```plaintext
[cardboard] Train: 322 | Val: 40 | Test: 41
[glass]     Train: 400 | Val: 50 | Test: 51
[metal]     Train: 328 | Val: 41 | Test: 41
[paper]     Train: 475 | Val: 59 | Test: 60
[plastic]   Train: 385 | Val: 48 | Test: 49
```

### Data Augmentation trên train

Dùng class MyCV từ lab 1 để thực hiện data augmentation trên tập train để tăng số lượng ảnh cho mỗi class.

1. Thực hiện data augmentation trên tập train để tăng số lượng ảnh cho mỗi class:

### Vector hóa ảnh

Dùng lại code của lab 2 để vector hóa ảnh trong các thư mục train/, val/, test/ thành các file text train.txt, val.txt, test.txt. Mỗi dòng trong file text có định dạng: `<label> <vectorized_image>`, trong đó `<label>` là nhãn của ảnh (0-4) và `<vectorized_image>` là vector hóa ảnh (dạng 1D). Pixel được chuẩn hóa về khoảng [0, 1].

1. Vector hóa ảnh trong các thư mục train/, val/, test/ thành các file text train.txt, val.txt, test.txt. Mỗi dòng trong file text có định dạng: `<label> <vectorized_image>`, trong đó `<label>` là nhãn của ảnh (0-4) và `<vectorized_image>` là vector hóa ảnh (dạng 1D). Pixel được chuẩn hóa về khoảng [0, 1].

## Cấu trúc và định dạng dữ liệu đầu ra (.txt)

## Cấu trúc thư mục

```plaintext
dataset/
├── README.md              # File mô tả chi tiết dataset
├── train/
│   ├── cardboard/
│   ├── glass/
│   ├── metal/
│   ├── paper/
│   └── plastic/
├── val/
│   ├── cardboard/              
│   ├── glass/       
│   ├── metal/       
│   ├── paper/       
│   └── plastic/    
└── test/
    ├── cardboard/
    ├── glass/
    ├── metal/
    ├── paper/
    └── plastic/
```
