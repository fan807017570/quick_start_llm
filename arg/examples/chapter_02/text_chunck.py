import PyPDF2
from tqdm import tqdm

with open('pdf/solving-mathematical-problems-terence-tao.pdf', 'rb') as file:
    reader = PyPDF2.PdfReader(file)
    principles_of_ds = ''
    for page in tqdm(reader.pages):
        text = page.extract_text()
        print(text)
        principles_of_ds += '\n\n' + text[text.find(' ]') + 2:]
    principles_of_ds = principles_of_ds.strip()
