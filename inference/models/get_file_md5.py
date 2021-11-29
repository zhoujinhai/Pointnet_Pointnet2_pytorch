import hashlib
import os


def get_file_md5(file_path):
    if not os.path.isfile(file_path):
        print("file is not exists!")
        return

    with open(file_path, "rb") as f:
        data = f.read()

    file_md5 = hashlib.md5(data).hexdigest()
    return file_md5


if __name__ == "__main__":
    file_path = r"./best_model.pth"
    md5_value = get_file_md5(file_path)
    print("{} md5 value is : {}".format(file_path, md5_value))
