from langchain_community.document_loaders import PyPDFLoader
from langchain_experimental.text_splitter import  SemanticChunker
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

# def prepare_data():
#     loader = PyPDFLoader(r'C:\Users\hp\Downloads\华娴(中文简历) .pdf')
#     documents = loader.load()
#     rag_embeddings = HuggingFaceEmbeddings(model_name=r'D:\EmbaddingModels\bge-small-zh-v1.5')
#     text_splitter = SemanticChunker(buffer_size=500,  embeddings=rag_embeddings)
#     chunks = text_splitter.split_documents(documents)
#     # print(chunks[0].page_content)
#     return chunks
#
# def embedding_data(chunks):
#     rag_embeddings = HuggingFaceEmbeddings(model_name=r'D:\EmbaddingModels\bge-small-zh-v1.5')
#     vector_store = Chroma.from_documents(documents=chunks, embedding=rag_embeddings,
#                                          collection_name='demo',
#                                          persist_directory='./chroma_langchain_db')
#     retriever = vector_store.as_retriever()
#     return vector_store, retriever
#
#
# # chunks = prepare_data()
# # retriever = embedding_data(chunks)
#
# vector_store = Chroma(
# collection_name='demo',
# embedding_function=HuggingFaceEmbeddings(model_name=r'D:\EmbaddingModels\bge-small-zh-v1.5'),
# persist_directory='./chroma_langchain_db'
# )
# print(vector_store.similarity_search(query="华娴的学历"))