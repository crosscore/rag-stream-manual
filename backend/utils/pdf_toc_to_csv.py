import fitz  # PyMuPDF
import pandas as pd
import os
import glob
import time

def extract_toc_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    toc = doc.get_toc()
    return [(item[1], item[2]) for item in toc]  # (text, page_num)

def write_toc_to_csv(toc, pdf_path, csv_output_path):
    toc_data = []
    index = 1
    file_name = os.path.basename(pdf_path)
    for entry in toc:
        toc_data.append([index, file_name, entry[0], entry[1]])
        index += 1
    df = pd.DataFrame(toc_data, columns=["index", "file_name", "toc", "page"])
    df.to_csv(csv_output_path, index=False)


pdf_dir = '../data/pdf/'
csv_dir = '../data/csv/original/'
os.makedirs(csv_dir, exist_ok=True)

pdf_files = glob.glob(os.path.join(pdf_dir, '*.pdf'))
print(pdf_files)

# 各PDFファイルから目次を抽出してCSVに保存
for pdf_path in pdf_files:
    base_name = os.path.splitext(os.path.basename(pdf_path))[0]
    csv_output_path = os.path.join(csv_dir, f"{base_name}.csv")

    if os.path.exists(pdf_path):
        toc_pdf = extract_toc_from_pdf(pdf_path)
        write_toc_to_csv(toc_pdf, pdf_path, csv_output_path)
    else:
        print(f"Error: {pdf_path} was not created.")

print("PDF to CSV conversion complete.")
