import os


def count_image_files(directory):
    image_extensions = (".jpg", ".jpeg", ".png", ".gif", ".bmp", ".tiff")
    image_count = 0

    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.lower().endswith(image_extensions):
                image_count += 1

    return image_count


# Example usage
directory_path = "J:/20K DATASET/REAL_FAKE/Real"
print(f"Number of image files: {count_image_files(directory_path)}")
