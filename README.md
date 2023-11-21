# Interact with your book 📖❓🙋🏻‍♀️

A simple demonstration of how you can implement retrieval augmented generation for a book.

## How retrieval augmented generation works

Following are the high level steps needed for the implementation for retrieval augmented generation.

1. Extract text from source. If the source is unstructured, like PDF, the extraction can be a challenge.
2. Index the extracted text, often as vector embeddings and store.
3. Let the user ask questions related to the source.
4. Perform a similarity search in the index and retrieve relevant text chunks.
5. Insert these text chunks in the prompt along with the question.
6. Request an LLM (e.g. chatgpt) to produce an answer *only* based on the context

## What do you find here

The notebook demonstrate the following steps 

1. Extraction of the relevant text from Cambridge O Level Computer Science book.
    - Only the main body text is extracted.
    - Following sections are excluded: table of content, index, sample questions at the end of each chapter, diagrams, tables, and other elements that were not part of the main text.
    - Every document needs to be carefully analyzed in order to extract useful text. This step is of utmost importance since answers to any user questions will depend upon the quality of input provided to the large language model.
2. The text was splitted into chunks and vector embeddings were created using HuggingFaceInstructEmbeddings.
3. Lanchain's FAISS vector store is used for saving embeddings.
4. Relevant texts were retrieved using similarity search
5. The final prompt is generated based on the searched context and copied to clipboard.
6. You just need to open you favorite LLM (e.g. chat.openai.com) and past the prompt to get the required answer.

The idea of creating this simple implementation is to quickly demonstrate how thing actually work in retrieval augmented generation. You do not need to acquire any API key or install a LLM locally.

## Name of the game

The success of any RAG implementation mainly depends on the following aspects

1. The quality of input data. If the data is unstructured, you need to carefully analyze and extract relevant and useful text only. The quality of input determines the quality of the answers you will get from LLM.
2. You should to experiment with chunk sizes and overlaps that are suitable for your application
3. You should test different number of top similar results that gives the best performance.
4. You can not have a universal prompt template that will work with every LLM. You will have to fine-tune the prompt for the specific LLM you are using in order to get best results.

## Getting started

You can run the notebook on google colab or locally.

### Using google colab

1. Open the notebook on google colab
2. Execute all the steps till the last cell
3. Experiment by changing the question

### Running locally

1. Clone the repo locally
2. Cd to local rep directory
3. Install all the requirements using pip install -r requirements.txt
4. Open jupyter notebook and execute

## Using your own pdf file

1. Add the pdf file(s) to books subfolder.
2. Change the relevant code to successfully extract needed text from the code.
3. Un-comment and execute the code that creates index and saves it on the disk.
4. Rest stays the same