import pymupdf4llm

fn = 'Kubernetes_book.pdf'
fout = 'output.md'

md_text = pymupdf4llm.to_markdown(fn)

# now work with the markdown text, e.g. store as a UTF8-encoded file
import pathlib
pathlib.Path(fout).write_bytes(md_text.encode())
# pandoc output.md -o output.txt
