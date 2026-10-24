import markdown, sys
md = open("REPORT.md").read()
body = markdown.markdown(md, extensions=["tables"])
css = """@page{size:A4;margin:12mm 13mm}body{font-family:'DejaVu Sans',Arial,sans-serif;font-size:8.6pt;line-height:1.3;color:#111}
h1{font-size:14pt;margin:0 0 3pt}h2{font-size:10.5pt;margin:7pt 0 2pt;border-bottom:1px solid #999}p{margin:2pt 0 3pt}
ul{margin:2pt 0 3pt 14pt;padding:0}li{margin:0 0 1.5pt}table{border-collapse:collapse;width:100%;margin:3pt 0 4pt;font-size:7.8pt}
th,td{border:1px solid #bbb;padding:1.5pt 3pt;vertical-align:top}th{background:#eee}code{font-size:7.6pt;background:#f3f3f3}
blockquote{margin:3pt 0;padding:3pt 6pt;border:1.5px dashed #c00;background:#fff4f4}"""
open("report/REPORT.html", "w").write(f"<html><head><meta charset='utf-8'><style>{css}</style></head><body>{body}</body></html>")
