from PyPDF2 import PdfReader
from langchain.document_loaders import PyPDFLoader
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



reader = PdfReader("books/Cambridge IGCSE and O Level Computer Science.pdf")
    
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
def visitor_body(text, cm, tm, fontDict, fontSize):
    if tm[5] > 60 and tm[5] < 740:  # exclude header and footer
        parts.append(text)

def extract_single_page(page):
    page.extract_text(visitor_text=visitor_body)
    text_body = "".join(parts)
    return text_body


def extract_pages(pdf_reader, source):
    documents = []
    
    for page_number, page in enumerate(pdf_reader.pages):
        if include_page(page_number):
            doc = Document(
                    page_content = extract_single_page(page),
                    metadata={"source": source, "page": page_number},
                    ) 
            documents.append(doc)
            global parts
            parts =[]
    return documents




documents = extract_pages(reader, "Cambridge IGCSE and O Level Computer Science.pdf")

print('pages extracted: ' + str(len(documents)))

# text_splitter = RecursiveCharacterTextSplitter(
#     chunk_size = CFG.split_chunk_size,
#     chunk_overlap = CFG.split_overlap
# )

# texts = text_splitter.split_documents(documents)

text_splitter = NLTKTextSplitter()

texts = text_splitter.split_documents(documents)

print(f'We have created {len(texts)} chunks from {len(documents)} pages')

print('done')

# ### download embeddings model
# embeddings = HuggingFaceInstructEmbeddings(
#     model_name = CFG.embeddings_model_repo,
#     model_kwargs = {"device": "cpu"}
# )

# ### create embeddings and DB
# vectordb = FAISS.from_documents(
#     documents = texts, 
#     embedding = embeddings
# )

# ### persist vector database
# vectordb.save_local("faiss_index_hp")