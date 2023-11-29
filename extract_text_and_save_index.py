import pandas as pd
from tqdm import tqdm
import pdfplumber
from langchain.document_loaders import DirectoryLoader
from langchain.schema import Document
# splits
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.text_splitter import NLTKTextSplitter

# prompts
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

# vector stores
from langchain.vectorstores import FAISS

# models
from langchain.llms import HuggingFacePipeline
from InstructorEmbedding import INSTRUCTOR
from langchain.embeddings import HuggingFaceInstructEmbeddings

# retrievers
from langchain.chains import RetrievalQA


### Open pdf file and extract text including tables

pdf_reader = pdfplumber.open("books/Cambridge IGCSE and O Level Computer Science.pdf")
    
included_pages_intervals = [[14, 52],
                 [57, 82],
                 [87, 155],
                 [159, 188],
                 [192, 225],
                 [229, 264],
                 [270, 306],
                 [311, 348],
                 [351, 365],
                 [368, 393]]

included_pages = []
for interval in included_pages_intervals:
    l = list(range(interval[0], interval[1]+1))
    included_pages = included_pages + l


def include_page(page_number):
    one_based_page_number = page_number + 1
    if one_based_page_number in included_pages:
        return True
    else:
        return False

parts = []
def include_text(obj):
    if 'size' in obj and obj['size'] >= 10:  # include all main body text (exclude figures , tables, header/footer etc.)
        return True
    else:
        return False

def extract_single_page(page):
    f_page = page.filter(include_text)
    text = f_page.extract_text()
    tables = page.find_tables()
    table_text = ''
    for table in tables:
        table_df = pd.DataFrame.from_records(table.extract())
        # test if table is empty (if all values are either '' or null)
        if (table_df == '').values.sum() + table_df.isnull().values.sum() == table_df.shape[0]*table_df.shape[1]:
           pass  # Table is empty 
        else:
            table_text =  table_text + '\n\n' + table_df.to_html(header=False, index=False)

    return text + '\n\n' + table_text


def extract_pages(pdf_reader, source):
    documents = []
    
    for page_number, page in tqdm(enumerate(pdf_reader.pages), total=len(pdf_reader.pages)):
        if include_page(page_number):
            doc = Document(
                    page_content = extract_single_page(page),
                    metadata={"source": source, "page": page_number - 11},
                    ) 
            documents.append(doc)
            global parts
            parts =[]
    return documents




documents = extract_pages(pdf_reader, "Cambridge IGCSE and O Level Computer Science.pdf")

print('pages extracted: ' + str(len(documents)))

### Split text

# text_splitter = NLTKTextSplitter()
# texts = text_splitter.split_documents(documents)

# print(f'We have created {len(texts)} chunks from {len(documents)} pages')

# print('done')

### Create vector embeddings

### download embeddings model
embeddings = HuggingFaceInstructEmbeddings(
    model_name = 'sentence-transformers/all-MiniLM-L6-v2',
    model_kwargs = {"device": "cpu"}
)

vectordb = FAISS.from_documents(
    documents = documents, 
    embedding = embeddings
)

### Persist embeddings in vector database
vectordb.save_local("faiss_index_hp")