# import argparse
import enum
import os
import sys
import subprocess

def _pad(x, size):
    chars = str(x)
    pad_len = size - len(chars)
    ret = chars + ' ' * pad_len    
    return bytearray(ret, 'utf-8')
    
    
GLOBAL_HEADER_SIZE = 8
GLOBAL_HEADER = b"!<arch>\n"
GLOBAL_THIN_HEADER = b"!<thin>\n"
END_OF_FILE_HEADER_MARKER = b"\x60\x0A"
ENTRY_SIZE = 60
SPECIAL_ENTRIES = ("/", "//", "/SYM64/")

COMMON_MODIFICATION_TIME_STAMP = _pad(476064000, 12)
DEFAULT_MODE = _pad(100644, 8)
ZERO_BYTES = _pad(0, 6)


def _modify_buffer(buffer: bytearray):
    """
    Header attribute length in bytes
    + 16 : Name
    + 12 : Timestamp
    + 6  : UserID
    + 6  : GroupID
    + 8  : Mode
    + 10 : Size
    + 2  : End Header
    """
    pos = 0
    # skip the name
    name = buffer[pos:pos+16]
    pos += 16
    # Modify Timestamp
    buffer[pos:pos+12] = COMMON_MODIFICATION_TIME_STAMP
    pos += 12
    # Modify UserID
    buffer[pos:pos+6]  = ZERO_BYTES
    pos += 6
    # Modify GroupID
    buffer[pos:pos+6]  = ZERO_BYTES
    pos += 6
    # Modify Mode
    buffer[pos:pos+8]  = DEFAULT_MODE
    pos += 8
    # Skip Size
    size = buffer[pos:pos+10]
    # Skip End Header
    pos += 10 
    end_header = buffer[pos:pos+2]
    # sanity check
    if end_header != END_OF_FILE_HEADER_MARKER:
        raise Exception("invalid file magic")
    return name, int(size)


def scrub(bianry_path: str):
    with open(bianry_path, "r+b") as f:
        header = f.read(GLOBAL_HEADER_SIZE)
        if header not in [GLOBAL_HEADER, GLOBAL_THIN_HEADER]:
            raise Exception("invalid global header")
        thin = header == GLOBAL_THIN_HEADER
        while True:    
            pos = f.tell()
            buffer = bytearray(f.read(ENTRY_SIZE))
            if len(buffer) < ENTRY_SIZE:
                break
            name, size = _modify_buffer(buffer)
            name = name.decode().strip()
            f.seek(pos)
            written = f.write(buffer)
            if written != ENTRY_SIZE:
                raise Exception("Not all bytes have been written")
            if not thin or name in SPECIAL_ENTRIES:
                f.seek(size, os.SEEK_CUR)


            
def main():
    ret = subprocess.call(sys.argv[1:])
    if ret != 0:
        sys.exit(ret)
    scrub(sys.argv[3])

if __name__ == "__main__":
    main()
