# Interact with your book 📖❓🙋🏻‍♀️

A simple demonstration of how you can implement retrieval augmented generation (RAG) for a book.

## 🚀 How retrieval augmented generation works

Following are the high level steps needed for the implementation for retrieval augmented generation.

1. Extract text from source. If the source is unstructured, like PDF, the extraction can be a challenge.
2. Index the extracted text, often as vector embeddings and store.
3. Let the user ask questions related to the source.
4. Perform a similarity search in the index and retrieve relevant text chunks.
5. Insert these text chunks in the prompt along with the question.
6. Request an LLM (e.g. chatgpt) to produce an answer *only* based on the context

## 🌟 What would you find here

The notebook demonstrate the following steps

1. Extraction of the relevant text from Cambridge O Level Computer Science book.
    - Only the main body text including tables is extracted.
    - Following sections are excluded: table of content, index, sample questions at the end of each chapter, and diagrams.
    - Every document needs to be carefully analyzed in order to extract useful text. This step is of utmost importance since answers to any user questions will depend upon the quality of input provided to the large language model.
2. The text was splitted into chunks using NLTKTextSplitter and vector embeddings were created using HuggingFaceInstructEmbeddings.
3. Langchain's FAISS vector store is used for saving embeddings.
4. Text relevant to the user query were retrieved from the index database using similarity search.
5. The final prompt is generated based on the searched context and copied to the clipboard.
6. You just need to open you favorite LLM (e.g. chat.openai.com) and past the prompt to get the required answer.

The idea of creating this simple implementation is to quickly demonstrate how thing actually work in retrieval augmented generation. You do not need to acquire any API key or install a LLM locally.

## 🧨 Name of the game

The success of any RAG implementation mainly depends on the following aspects

1. The quality of input data. If the data is unstructured, you need to carefully analyze and extract relevant and useful text only. The quality of input determines the quality of the answers you will get from LLM.
2. You should experiment with different text splitters and decide which one works best for your application.
3. You should test different number of top similar results that gives the best performance.
4. You can not have a universal prompt template that will work with every LLM. You will have to fine-tune the prompt for the specific LLM you are using in order to get best results.

## 🎬 Getting started

You can run the notebook locally and use the final prompt to generate an answer with the help of your favorite LLM.

### Pre-requisite

1. Python 3.11, preferably in a virtual environment.
2. Access to a large language model for example:
    - <https://chat.openai.com>
    - <https://www.llama2.ai>
    - <https://huggingface.co/chat>

### Running locally

1. Clone the repo locally

    ```bash
    git clone https://github.com/mamiriqbal1/rag_book_qa_prompt.git
    ```

2. CD to local repo directory

    ```bash
    cd rag_book_qa_prompt
    ```

3. Install all the requirements

    ```bash
    pip install -r requirements.txt
    ```

4. Open jupyter notebook and run all the cells. Wait for execution to complete. You can then change your question and re-run the last cell again and again to get the final prompt for using with you favorite LLM.

    ```bash
    jupyter notebook rag_book_qa_prompt.ipynb
    ```

## Using your own PDF file

You can use your own PDF file with this notebook.

1. Add the PDF file to books subfolder.
2. Customize the relevant code in the python file extract_text_and_save_index.py to adequately extract needed text from the PDF.
3. Run the modified python file to extract the text and create and save index.
4. Use the provided jupyter notebook to perform QA on your new source file.
