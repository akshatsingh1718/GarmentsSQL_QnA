from dotenv import load_dotenv
from langchain.llms import GooglePalm
from langchain.utilities import SQLDatabase
from langchain_experimental.sql import SQLDatabaseChain
from langchain.prompts.prompt import PromptTemplate
from langchain.chains.sql_database.prompt import PROMPT_SUFFIX, _mysql_prompt
from few_shot_examples import few_shots
from langchain.prompts import SemanticSimilarityExampleSelector
from langchain.vectorstores import Chroma
from langchain.prompts import FewShotPromptTemplate
import os
import sys

load_dotenv()

USE_OPENAI_EMBEDDINGS= True
USE_FEWSHOT_EXAMPLES= True
VERBOSE= True

llm = GooglePalm(google_api_key=os.environ.get("GOOGLE_API_KEY"), temperature=0.2)


def get_database_connection(llm):
    DB_USER = os.environ.get("DB_USER")
    DB_PASSWORD = os.environ.get("DB_PASSWORD")
    DB_HOST = os.environ.get("DB_HOST")
    DB_NAME = os.environ.get("DB_NAME")
    db = SQLDatabase.from_uri(
        f"mysql+pymysql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}/{DB_NAME}",
        sample_rows_in_table_info=3,
    )
    return db


def get_vectorstore():
    few_shots_concatenated = ["\n".join(ex.values()) for ex in few_shots]

    embeddings = get_embeddings(USE_OPENAI_EMBEDDINGS)
    vectorstore = Chroma.from_texts(
        few_shots_concatenated, embedding=embeddings, metadatas=few_shots
    )

    return vectorstore


def get_chain():
    if not USE_FEWSHOT_EXAMPLES:
        db = get_database_connection(llm)
        new_chain = SQLDatabaseChain.from_llm(llm, db, verbose=VERBOSE)
        return new_chain
    
    vectorstore = get_vectorstore()
    example_selector = SemanticSimilarityExampleSelector(
        vectorstore=vectorstore,
        k=1,  # select top k =2 examples which matches most with the given string/query
    )
    example_prompt_variables = ["Question", "SQLQuery", "SQLResult", "Answer"]

    example_prompt = PromptTemplate(
        input_variables=example_prompt_variables,
        template="\n".join(
            [
                (f"{ex_var}: " + "{" + ex_var + "}")
                for ex_var in example_prompt_variables
            ]
        ),
    )
    few_shot_prompt = FewShotPromptTemplate(
        example_selector=example_selector,
        example_prompt=example_prompt,
        prefix=_mysql_prompt,
        suffix=PROMPT_SUFFIX,
        input_variables=["input", "table_info", "top_k"],
    )

    db = get_database_connection(llm)
    new_chain = SQLDatabaseChain.from_llm(llm, db, verbose=True, prompt=few_shot_prompt)
    return new_chain


def get_embeddings(use_openai_embeddings=True):
    embeddings = None

    if use_openai_embeddings:
        from langchain.embeddings import OpenAIEmbeddings

        embeddings = OpenAIEmbeddings()
    else:
        from langchain.embeddings import HuggingFaceEmbeddings

        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-vs"
        )
    return embeddings


if __name__ == "__main__":
    chain = get_chain()
    print(
        chain.run("How much is the price of the inventory for all small size t-shirts?")
    )
