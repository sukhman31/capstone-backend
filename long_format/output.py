from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.retrievers.multi_query import MultiQueryRetriever
import os
from dotenv import load_dotenv
import cohere
from langchain.embeddings import CohereEmbeddings
from langchain.vectorstores.chroma import Chroma
from langchain.llms.replicate import Replicate
import logging
import io
import ast
import re
from typing import List
from langchain.chains import LLMChain
from langchain.output_parsers import PydanticOutputParser
from langchain.prompts import PromptTemplate
from pydantic import BaseModel, Field

log_capture_string = io.StringIO()
ch = logging.StreamHandler(log_capture_string)
ch.setLevel(logging.INFO)

logger = logging.getLogger("langchain.retrievers.multi_query")
logger.addHandler(ch)
logger.setLevel(logging.INFO)

logging.basicConfig()
logger.setLevel(logging.INFO)

class LineList(BaseModel):
    # "lines" is the key (attribute name) of the parsed output
    lines: List[str] = Field(description="Lines of text")


class LineListOutputParser(PydanticOutputParser):
    def __init__(self) -> None:
        super().__init__(pydantic_object=LineList)

    def parse(self, text: str) -> LineList:
        lines = text.strip().split("\n")
        return LineList(lines=lines)

load_dotenv()

cohere_key = os.getenv("COHERE_API_KEY")
co = cohere.Client(cohere_key)
REPLICATE_API_TOKEN = os.getenv("REPLICATE_API_TOKEN")
os.environ["REPLICATE_API_TOKEN"] = REPLICATE_API_TOKEN

def get_rerank(texts,queries):
    result_list = []
    for q in queries:
        if q=='':
            continue
        results = co.rerank(query=q, documents=texts, top_n=3, model='rerank-english-v2.0')
        result_list.extend(results)
    
    unique_indices = set()
    filtered_results = [result for result in result_list if result.index not in unique_indices and result.relevance_score > 0.85 and (unique_indices.add(result.index) or True)]
    sorted_results = sorted(filtered_results, key=lambda x: x.relevance_score, reverse=True)
    return sorted_results


async def get_queries(question,texts):
    output_parser = LineListOutputParser()

    QUERY_PROMPT = PromptTemplate(
        input_variables=["question"],
        template="""You are an AI language model assistant. Your task is to generate five 
        different versions of the given user question to retrieve relevant documents from a vector 
        database. By generating multiple perspectives on the user question, your goal is to help
        the user overcome some of the limitations of the distance-based similarity search. 
        Provide these alternative questions separated by newlines.
        Original question: {question}""",
    )
    llm = Replicate(
        model="a16z-infra/llama13b-v2-chat:df7690f1994d94e96ad9d568eac121aecf50684a0b0963b25a41cc40061269e5",
        model_kwargs={"temperature": 0.75, "max_length": 500, "top_p": 1},
    )

    llm_chain = LLMChain(llm=llm, prompt=QUERY_PROMPT, output_parser=output_parser)
    embeddings = CohereEmbeddings(cohere_api_key=cohere_key)
    vectorstore = Chroma.from_documents(documents=texts,embedding=embeddings)
    retriever = MultiQueryRetriever(
        retriever=vectorstore.as_retriever(), llm_chain=llm_chain, parser_key="lines"
    )

    unique_docs = retriever.get_relevant_documents(
        query=question
    )
    log_contents = log_capture_string.getvalue()
    match = re.search(r'\[.*?\]', log_contents)

    if match:
        log_contents = match.group(0)
    parsed_list = ast.literal_eval(log_contents)
    return parsed_list


async def output(document,question):

    #split text recursively to obtain better results
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = 100,
        chunk_overlap = 20
    )
    texts = text_splitter.create_documents([document])
    list_of_texts = [t.page_content for t in texts]

    # use multi-query retrieval
    queries = await get_queries(question,texts)
    queries = queries[:9]
    queries.append(question)
    print('---------------')
    print(queries)

    # use cohere reranking to get best outcome
    final_chunks = get_rerank(list_of_texts,queries)    

    print('---------------')
    print(final_chunks)

    context = ''
    for i,chunk in enumerate(final_chunks):
        context += str(i+1)+chunk.document['text']+'\n'
    
    print('------------------')
    print(context)

    #pass through llm and get output
    llm = Replicate(
        model="a16z-infra/llama13b-v2-chat:df7690f1994d94e96ad9d568eac121aecf50684a0b0963b25a41cc40061269e5",
        model_kwargs={"temperature": 0.75, "max_length": 500, "top_p": 1},
    )
    QUERY_PROMPT = PromptTemplate(
        input_variables=["question","context"],
        template="""This is the question : {question}. Convert the following points into a paragraph which answers the question.
        {context}""",
    )
    llm_chain = LLMChain(llm=llm, prompt=QUERY_PROMPT)
    final_response = llm_chain.predict(question=question, context=context)
    print('---------------------')
    print(final_response)
    return final_response

