import argparse


def parse_arguments():
    parser = argparse.ArgumentParser(description="Video sequence model configuration")
    parser.add_argument(
        "--sequence-length",
        type=int,
        default=20,
        help="Số lượng khung hình trong một video (default: 20)",
    )
    parser.add_argument(
        "--image-height",
        type=int,
        default=128,
        help="Chiều cao của mỗi khung hình (default: 128)",
    )
    parser.add_argument(
        "--image-width",
        type=int,
        default=128,
        help="Chiều rộng của mỗi khung hình (default: 128)",
    )
    parser.add_argument(
        "--classes-list",
        nargs="+",
        default=["NonViolence", "Violence"],
        help="Danh sách các lớp (nhãn), cách nhau bởi dấu cách",
    )

    parser.add_argument("--hidden-size", default=32)
    parser.add_argument("--dropout-prob", default=0.5)

    args = parser.parse_args()
    return args
