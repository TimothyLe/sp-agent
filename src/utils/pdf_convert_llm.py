import pymupdf4llm
import sys
import pathlib
import subprocess

try:
    fn = sys.argv[1]
    fout = sys.argv[2]
except:
    print("Usage: python3 pdf_convert_llm.py <SRC> <DEST>")  
    sys.exit(1)

# convert pdf to parsable format
md_text = pymupdf4llm.to_markdown(fn)

# now work with the markdown text, e.g. store as a UTF8-encoded file
f_md = fout+".md" 
f_txt = fout+".txt" 
pathlib.Path(f_md).write_bytes(md_text.encode())
subprocess.run(f"pandoc {f_md} -o {f_txt}".split())

# remove artifact
try: 
    subprocess.run(f"rm {f_md}".split())
    print(f"Complete. {f_txt} generated.")
except FileNotFoundError:
    print(f"File {f_md} not found.")
except Exception as e:
    print(f"An error occurred: {e}")
