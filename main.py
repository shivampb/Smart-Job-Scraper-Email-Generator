from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_ollama import OllamaLLM
from langchain.schema import Document
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import GoogleGenerativeAI
from langchain_core.output_parsers import JsonOutputParser
import pandas as pd
import chromadb
import uuid

# Configuration
URL = "JOB BLOG"
API_KEY = "GEMINI API KEY"
VECTORSTORE_PATH = "vectorstore"
PORTFOLIO_CSV = "my_portfolio.csv"

def create_prompts():
    return {
        "extract": PromptTemplate.from_template(
            """
            ### SCRAPED TEXT:
            {page_data}
            
            ### INSTRUCTION:
            Extract job postings from the About Job section. Return a single JSON with these keys: `role (with company name example: AT xyz company)`, `experience`, `skills`, and `description`.
            - Respond with valid JSON only, no extra text.
            """
        ),
        "clean": PromptTemplate.from_template(
            """
            {page_data}  this is scraped data. Make this data human-readable.
            """
        ),
        "email": PromptTemplate.from_template(
            """
            ### JOB DESCRIPTION:
            {job_description}
            
            ### INSTRUCTION:
            You are SHIVAM, a business development executive at Gambler. Gambler is an AI & Software Consulting company dedicated to facilitating
            the seamless integration of business processes through automated tools. 
            Over our experience, we have empowered numerous enterprises with tailored solutions, fostering scalability, 
            process optimization, cost reduction, and heightened overall efficiency. 
            Your job is to write a structured cold EMAIL to the client regarding the job mentioned above describing the capability of Gambler 
            in fulfilling their needs.
            Also, add the most relevant ones from the following links to showcase Gambler's portfolio: {link_list}
            
            Remember, you are Shivam, BDE at Gambler.
            Do not provide a preamble.
            
            ### EMAIL (NO PREAMBLE):
            """
        )
    }

def load_web_content(url):
    try:
        loader = WebBaseLoader(url)
        return loader.load().pop().page_content
    except Exception as e:
        print(f"Error loading content: {e}")
        return ""

def initialize_llm():
    return GoogleGenerativeAI(model="gemini-2.0-flash", google_api_key=API_KEY)

def process_text(llm, prompt, extracted_content):
    chain = prompt | llm
    return chain.invoke(input={"page_data": extracted_content})

def parse_json_response(response):
    try:
        return JsonOutputParser().parse(response)
    except Exception as e:
        print(f"Error parsing JSON: {e}")
        return {}

def load_portfolio():
    try:
        return pd.read_csv(PORTFOLIO_CSV)
    except FileNotFoundError:
        print("Portfolio CSV file not found.")
        return pd.DataFrame()

def initialize_vectorstore():
    client = chromadb.PersistentClient(VECTORSTORE_PATH)
    return client.get_or_create_collection(name="portfolio")

def populate_vectorstore(collection, portfolio_data):
    if not collection.count() and not portfolio_data.empty:
        for _, row in portfolio_data.iterrows():
            collection.add(
                documents=row["Techstack"],
                metadatas={"links": row["Links"]},
                ids=[str(uuid.uuid4())]
            )

def query_vectorstore(collection, skills):
    if skills:
        return collection.query(query_texts=skills, n_results=2).get("metadatas", [])
    return []

def generate_email(llm, prompt, json_response, links):
    chain_email = prompt | llm
    return chain_email.invoke(input={"job_description": str(json_response), "link_list": links})

def main():
    prompts = create_prompts()
    extracted_content = load_web_content(URL)
    llm = initialize_llm()
    
    response_clean = process_text(llm, prompts["clean"], extracted_content)
    # print("Human-readable Job Posting:\n", response_clean)
    
    final_response = process_text(llm, prompts["extract"], response_clean)
    print("Extracted Job Details:\n", final_response)
    
    json_response = parse_json_response(final_response)
    
    portfolio_data = load_portfolio()
    collection = initialize_vectorstore()
    populate_vectorstore(collection, portfolio_data)
    
    links = query_vectorstore(collection, json_response.get("skills"))
    # print("Relevant Portfolio Links:", links)
    
    email = generate_email(llm, prompts["email"], json_response, links)
    print("Generated Cold Email:\n", email)

if __name__ == "__main__":
    main()
