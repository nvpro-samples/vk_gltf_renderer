import sys
import struct

def png_to_h(png_file, header_file):
    with open(png_file, "rb") as f:
        png_data = f.read()

    # Generate the header content
    header_content = f"unsigned char {header_file[:-2]}[] = {{\n"
    for i in range(0, len(png_data), 19):
        header_content += "    " + ", ".join(f"0x{png_data[j]:02x}" for j in range(i, min(i + 19, len(png_data)))) + ",\n"
    header_content += "};\n"
    header_content += f"unsigned int {header_file[:-2]}_len = {len(png_data)};\n"

    with open(header_file, "w") as h:
        h.write(header_content)

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python png_to_h.py <input_png> <output_h>")
        sys.exit(1)

    png_to_h(sys.argv[1], sys.argv[2])
