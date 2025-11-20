import re
import warnings
warnings.filterwarnings("ignore")
from openai import OpenAI
import functools
import requests
from bs4 import BeautifulSoup
import operator
import openai
import re
import os
import requests
import json
import tiktoken
#from typing import Annotated, Any, Callable, Dict, List, Optional, Sequence, TypedDict, Union
import pandas as pd
import pickle
import json
import os
from langchain.agents import AgentExecutor, create_openai_tools_agent, create_openai_functions_agent
from langchain_core.messages import BaseMessage, HumanMessage
from langchain.output_parsers.openai_functions import JsonOutputFunctionsParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.tools import tool
from langchain_openai import ChatOpenAI
from langchain.agents import load_tools, initialize_agent
from openai import OpenAI
from langchain_core.messages import AIMessage, BaseMessage, ChatMessage, FunctionMessage, HumanMessage, SystemMessage
import requests
from bs4 import BeautifulSoup
from datetime import datetime
import re
import numpy as np
import time
import pickle
import sys
from sklearn.metrics import accuracy_score, f1_score, precision_score
from difflib import SequenceMatcher
import itertools
from numpy.linalg import norm

import os

os.environ["OPENAI_BASE_URL"] = ""
os.environ["OPENAI_API_KEY"] = ""

def count_tokens(text, model="cl100k_base"):
    enc = tiktoken.get_encoding(model)
    
    tokens = enc.encode(text)
    
    return len(tokens)

def find_label_in_second_last_paragraph(text):
    paragraphs = text.split('\n\n')

    if len(paragraphs) < 2:
        return text

    pattern_0 = r'0|Real-News'
    pattern_1 = r'1|Rumors'

    match_0 = re.search(pattern_0, paragraphs[-2])
    match_1 = re.search(pattern_1, paragraphs[-2])

    if match_0:
        return 0  
    elif match_1:
        return 1  
    else:
        return text 
    
from langchain_community.utilities import SerpAPIWrapper

def evaluate_predictions(y_true, y_pred):
    accuracy = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    return accuracy, f1, precision

def write_list_to_file(my_list, file_path):
    with open(file_path, 'w') as file:
        for item in my_list:
            file.write(f"{item}\n")


def extract_final_result(input_data):
    content = input_data

    last_line = content.strip().split('\n')[-1]
    pattern_0 = r'(0|real)'
    pattern_1 = r'(1|fake)'

    if re.search(pattern_0, last_line):
        return 0
    elif re.search(pattern_1, last_line):
        return 1
    else:
        return content

from langchain.output_parsers import StructuredOutputParser, ResponseSchema
from langchain.prompts import PromptTemplate, ChatPromptTemplate, HumanMessagePromptTemplate
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
import itertools

def get_formatmessage():
  response_schemas = [
      ResponseSchema(name="answer", description="""should be "1" or "0" """),
  ]
  output_parser = StructuredOutputParser.from_response_schemas(response_schemas)
  format_instructions = output_parser.get_format_instructions()
  # input your prompt
  prompt = PromptTemplate(
      template=
      """""",
      input_variables=["analysis_text"],
      partial_variables={"format_instructions": format_instructions}
  )
  return output_parser,prompt


def enter_chain(message: str):
    results = {
        "input":[HumanMessage(content=message)],
    }
    return results



output_parser,prompt = get_formatmessage()
model_judge = OpenAI(temperature=0)

from openai import OpenAI
def chat_gpt_once(input):
    client = OpenAI(
        api_key=os.environ.get("OPENAI_API_KEY"),
        base_url=os.environ.get("OPENAI_BASE_URL"),
    )
    completion = client.chat.completions.create(
      # choose your model
      # model = ""
      messages=[
        {"role": "user", "content":input}
      ]
    )
    return completion.choices[0].message.content

class ChatGPTClient:
    def __init__(self, prompt,temperature,api_key=None, base_url=None):
        self.api_key = os.environ.get("OPENAI_API_KEY")
        self.base_url = os.environ.get("OPENAI_BASE_URL")
        self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)
        self.preix = prompt
        self.temperature = temperature

    def chat(self, input):
        try:
            completion = self.client.chat.completions.create(
                model="",
                messages=[
                    {"role": "user", "content": self.preix + input}
                ],
                temperature=self.temperature
            )
        except:
          x = self.preix[:1500] + input[:1500]
          completion = self.client.chat.completions.create(
              model="",
              messages=[
                  {"role": "user", "content": x}
              ],
              temperature=self.temperature
          )
        return completion.choices[0].message.content
Judger = ChatGPTClient(prompt =""" """,temperature=0)

Reluer = ChatGPTClient(prompt ="""Let's think step by step""",temperature=1)

def get_label_with_choose(chonse_model,lines,prompt,output_parser):
  _input = prompt.format_prompt(analysis_text=lines)
  output = chonse_model(_input.to_string())
  try:
    if not output:
      _input = prompt.format_prompt(analysis_text=lines)
      output = chonse_model(_input.to_string())
    l = output_parser.parse(output)
  except:
    l = lines
  return l

def get_related_experience (all_questtion_emb,all_experiences,test_qeuestion):
  top_indices, top_cosine_similarities = find_top_n_cosine_similarities(all_questtion_emb, test_qeuestion, n=3)

  chosed_experiences = all_experiences.iloc[top_indices].values.tolist()

  indexed_experiences = [f"esp{i + 1}:{experience[0]}" for i, experience in enumerate(chosed_experiences)]

  format_experiences = "|".join(indexed_experiences)

  return format_experiences

def find_top_n_cosine_similarities(A, B, n=3):
    cosine_similarities = np.dot(A, B) / (np.linalg.norm(A, axis=1) * np.linalg.norm(B))

    top_indices = np.argsort(cosine_similarities)[-n:][::-1]
    top_cosine_similarities = cosine_similarities[top_indices]
    return top_indices, top_cosine_similarities

def store_question_emb_to_file(file_name, array):
    with open(file_name, 'ab') as file:
        pickle.dump(array, file)
def load_question_emb_from_file(file_name):
    loaded_arrays = []
    with open(file_name, 'rb') as file:
        while True:
            try:
                loaded_array = pickle.load(file)
                loaded_arrays.append(loaded_array)
            except EOFError:
                break
    return np.array(loaded_arrays)[0]

class SerperClient:
    def __init__(self):
        self.url = ""
        self.headers = {
            "X-API-KEY":  os.environ["SERPAPI_API_KEY"] ,
            "Content-Type": "application/json"
        }

    def serper(self, query: str):
        # Configure the query parameters for Serper API
        serper_settings = {"q": query, "page": 2}

        # Check if the query contains Chinese characters and adjust settings accordingly
        if self._contains_chinese(query):
            serper_settings.update({"gl": "cn", "hl": "zh-cn",})

        payload = json.dumps(serper_settings)

        # Perform the POST request to the Serper API and return the JSON response
        response = requests.request("POST", self.url, headers=self.headers, data=payload)
        return response.json()

    def _contains_chinese(self, query: str):
        # Check if a string contains Chinese characters using a regular expression
        pattern = re.compile(r'[\u4e00-\u9fff]+')
        return bool(pattern.search(query))

    def extract_components(self, serper_response: dict):
        # Initialize lists to store the extracted components
        titles, links, snippets = [], [], []

        # Iterate through the 'organic' section of the response and extract information
        for item in serper_response.get("organic", []):
            titles.append(item.get("title", ""))
            links.append(item.get("link", ""))
            snippets.append(item.get("snippet", ""))

        # Retrieve additional information from the response
        query = serper_response.get("searchParameters", {}).get("q", "")
        count = len(links)
        language = "zh-cn" if self._contains_chinese(query) else "en-us"

        # Organize the extracted data into a dictionary and return
        output_dict = {
            'query': query,
            'language': language,
            'count': count,
            'titles': titles,
            'links': links,
            'snippets': snippets
        }

        return output_dict

def compare_time(time_str):
    try:
        time_obj = datetime.strptime(time_str, "%b. %d, %Y")
    except ValueError:
        try:
            time_obj = datetime.strptime(time_str, "%B , %d, %Y")
        except ValueError:
            time_obj =  None
    if time_obj is None:
        return True
    else:
      threshold = datetime(2017, 12, 31)
      if time_obj < threshold:
        return True
      else:
        return False
      
import requests
from bs4 import BeautifulSoup

def web_search(keyword):
  url = 'https://www.snopes.com'
  response = requests.get(f'{url}/search/{keyword}')
  if response.status_code == 200:
    first_soup = BeautifulSoup(response.text, 'html.parser')
  count= 3
  news_conent  = ""
  for i in range(count) :
    time_str = first_soup.find("span", class_="article_date").get_text().strip()
    if compare_time(time_str):
      conent_tmp =str(i)+ ". The news lable is: "
      article_url = first_soup.find('input', id='article_url_'+str(i))
      if article_url is not None :
        article_url = article_url.get("value")
      else:
        return "No relevant information found"
      response = requests.get(article_url)
      soup = BeautifulSoup(response.text, 'html.parser')

      rating_section = soup.find('div', class_='rating_wrapper')
      if rating_section is not None:
        rating = rating_section.find('div', class_='rating_title_wrap').text.strip()+"."
      else:
        rating = ""
      conent_tmp= conent_tmp+rating+" article_content:"
      article_content = soup.find('article', id='article-content')
      paragraphs = article_content.find_all('p')
      for j in range(1,len(paragraphs)):
        conent_tmp = conent_tmp+paragraphs[j].text.strip()
      conent_tmp = conent_tmp+"|\n"
      news_conent = news_conent+conent_tmp
    else:
      return "There is no information matching the time period."
  return news_conent

def extract_questions(text):
    questions = re.findall(r'\b[A-Z][^.?!]*\?', text)

    # Return the list of questions
    return questions
def Websearch(query: str) -> str:
    client = SerperClient()
    response = client.serper(query)
    components = client.extract_components(response)["snippets"]
    return components

@tool("fact_check")
def fact_check(query: str) -> str:
    search = SerpAPIWrapper()
    return search.run(query)

@tool("content_analysis")
def content_analysis(content: str) -> str:
    chat = ChatOpenAI()
    messages = [
      SystemMessage(
        # input your prompt
          content=""

      ),
      HumanMessage(
          content=content
      ),
    ]
    response = chat(messages)
    return response.content

@tool("review_analysis")
def review_analysis(content: str) -> str:
    chat = ChatOpenAI()
    messages = [
      SystemMessage(
        # input your prompt

          content=""
      ),
      HumanMessage(
          content=content
      ),
    ]
    response = chat(messages)
    return response.content

@tool("Notools")
def Notools(content: str) -> str:
   pass

def Websearch(query: str) -> str:
    client = SerperClient()
    response = client.serper(query)
    components = client.extract_components(response)["snippets"]
    all_components  = ""
    for  component in components:
      all_components = all_components+component
    return all_components

def Get_finally_label(result: str) -> str:
    return "not a rumor"

wikipedia_tools = load_tools(["wikipedia"])

def create_agent(llm: ChatOpenAI, tools: list, system_prompt: str):
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                system_prompt,
            ),
            MessagesPlaceholder(variable_name="input"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ]
    )
    agent = create_openai_tools_agent(llm, tools, prompt)
    executor = AgentExecutor(agent=agent, tools=tools)
    return executor

def create_agent_without_tools(llm: ChatOpenAI,tools, system_prompt: str):
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                system_prompt,
            ),
            MessagesPlaceholder(variable_name="input"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ]
    )
    agent = create_openai_tools_agent(llm,tools,prompt)
    executor = AgentExecutor(agent=agent,tools=tools)
    return executor

def agent_node(state, agent, name):
    print("input:",state)
    result = agent.invoke(state)

    print(name+"-"+result["output"])
    return name+"-"+result["output"]

def enter_chain(message: str):
    results = {
        "input":[HumanMessage(content=message)],
    }
    return results



def ues_new_api():

  llm = ChatOpenAI(model="")


  content_analyst_agent = create_agent_without_tools(ChatOpenAI(model="",temperature=0), [Notools],
# input your prompt
"""
""")
  content_analyst_node = functools.partial(agent_node, agent=content_analyst_agent, name="Writing Style Analysis Agent")


  review_analyst_agent = create_agent_without_tools(ChatOpenAI(model="",temperature=0), [Notools],
# input your prompt

"""
""")
  review_analyst_node = functools.partial(agent_node, agent=review_analyst_agent, name="Comment Analysis Agent")

  fact_question_agent = create_agent_without_tools(ChatOpenAI(model="",temperature=0), [Notools],
# input your prompt
"""
""")
  fact_question_node = functools.partial(agent_node, agent=fact_question_agent, name="Fact-Checking Questioning Agent")


  fact_executor_agent = create_agent_without_tools(ChatOpenAI(model="Meta-Llama-3.1-405B-Instruct",temperature=0), wikipedia_tools,
# input your prompt

"""
""")
  fact_executor_node = functools.partial(agent_node, agent=fact_executor_agent, name="Fact-Checking Agent")


  Fact_Checking_Summarizer_agent = create_agent_without_tools(ChatOpenAI(model="",temperature=0), [Notools],
# input your prompt

"""
""")
  Fact_Checking_Summarizer_node = functools.partial(agent_node, agent=Fact_Checking_Summarizer_agent, name="Fact-Checking Summarizer")

  expert_agent1 = create_agent_without_tools(ChatOpenAI(model="",temperature=0), [Notools],
# input your prompt
"""
""")

  expert_agent1_node = functools.partial(agent_node, agent=expert_agent1, name="Expert Agent")

  expert_question_agent_1 = create_agent_without_tools(ChatOpenAI(model="",temperature=0), [Notools],
# input your prompt
"""
""")
  expert_question_node_1 = functools.partial(agent_node, agent=expert_question_agent_1, name="Expert Questions Agent 1")

  expert_question_agent_2 = create_agent_without_tools(ChatOpenAI(model="",temperature=0), [Notools],
# input your prompt
"""
""")
  expert_question_node_2 = functools.partial(agent_node, agent=expert_question_agent_2, name="Expert Questions Agent")

  expert_question_agent_3 = create_agent_without_tools(ChatOpenAI(model="",temperature=0), [Notools],
# input your prompt
"""
""")
  expert_question_node_3 = functools.partial(agent_node, agent=expert_question_agent_3, name="Expert Questions Agent 3")

  Summarizer_agent = create_agent_without_tools(ChatOpenAI(model="",temperature=0.1), [Notools],

"""
""")
  Summarizer_node = functools.partial(agent_node, agent=Summarizer_agent, name="Experience Summarizer")


  Sub_summarizer_agent = create_agent_without_tools(ChatOpenAI(model="",temperature=0.1), [Notools],
# input your prompt
"""
""")
  Sub_summarizer_node = functools.partial(agent_node, agent=Summarizer_agent, name="Summarizer")


  return Sub_summarizer_node, Summarizer_node, content_analyst_node, review_analyst_node, fact_question_node, fact_executor_node, Fact_Checking_Summarizer_node, expert_agent1_node, expert_question_node_1, expert_question_node_2, expert_question_node_3

import re
def remove_label(text):
  keylist = ["Real-News", "Rumor", "real news", "rumor", "real", "fake"]
  for keyword in keylist:
    sentences = re.split(r'(?<=[.])\s+', text)
    filtered_sentences = [sentence for sentence in sentences if keyword not in sentence]

    text = ' '.join(filtered_sentences)
  return text

def Fact_check_system(enter,Original_news,Fact_check_system_message):

   result_tmp1= fact_question_node(enter)

   Fact_check_system_message.append([result_tmp1])

   questions_list = extract_questions(result_tmp1)
   answer_list = ["***Fact-Checking Question*** : " + question + " ***Information Collected from Search Engines*** : " + Websearch(question) for question in questions_list]
   answer_list = [",".join(answer_list)]
   Fact_check_system_message.append(answer_list)
   web_information = enter_chain(Fact_check_system_message[-1][-1])
   clues = Fact_Checking_Summarizer_node(web_information)
   Fact_check_system_message.append([clues])
   clues = enter_chain("***Please Perform Fact-Checking Task***: " + Original_news +" || "+ " Clues: " + Fact_check_system_message[-1][-1])
   fact_executor_report = fact_executor_node(clues)
   Fact_check_system_message.append([fact_executor_report])
   return Fact_check_system_message[-1]

def Fact_check_system_answer_exe_answer(enter):
   enter = enter_chain("***Please Perform Answer Questions Task*** " + enter)
   result_tmp2 = enter
   result= fact_executor_node(result_tmp2)
   Fact_check_system_message.append([result])
   return Fact_check_system_message[-1]

def Analysis_check_system(enter,Original_news_enter=None):
  if Original_news_enter is not None:
    result_one = content_analyst_node(Original_news_enter)
    print("*********generate content analysis report**********")
    result_two = review_analyst_node(enter)
    print("*********generate comment analysis report**********")
  else:
    enter  = enter_chain("***Please Perform the Answer Questions Task*** " + enter)
    result_one = content_analyst_node(enter)
    print("*********answer content analysis report**********")
    result_two = review_analyst_node(enter)
    print("*********answer comment analysis report**********")
  Analysis_check_system_message.append(["1. "+ result_one + " | "+ " 2. "+ result_two ])
  return Analysis_check_system_message[-1]

def Expert_check_system_ask(enter):
   result_one = expert_agent1_node(enter)
   result_two = expert_agent2_node(enter)
   result_three = expert_agent3_node(enter)
   Expert_check_system_message.append(["Expert A: "+result_one+" | "+" Expert B: "+result_two+" | "+" Expert C: "+result_three])
   return Expert_check_system_message[-1]

def Expert_check_system_debate_1(enter):
   result_one = expert_agent1_node(enter)
   result_two = expert_agent2_node(enter)
   result_three = expert_agent3_node(enter)
   Expert_check_system_message.append([" Expert A: " + result_one + " | " + " Expert B: " + result_two + " | " + " Expert C: " + result_three])

   result_one = expert_agent1_node(enter_chain("***Please Perform Judgment After Reference*** " + Expert_check_system_message[-1][-1]))
   result_two = expert_agent2_node(enter_chain("***Please Perform Judgment After Reference*** " + Expert_check_system_message[-1][-1]))
   result_three = expert_agent3_node(enter_chain("***Please Perform Judgment After Reference*** " + Expert_check_system_message[-1][-1]))
   Expert_check_system_message.append([" Expert A: " + result_one + " | " + " Expert B: " + result_two + " | " + " Expert C: " + result_three])

   return Expert_check_system_message[-1]

def Expert_check_system_debate_2(enter):
   result_one = expert_agent1_node(enter)
   result_two = expert_agent2_node(enter)
   result_three = expert_agent3_node(enter)
   Expert_check_system_message.append([" Expert A: " + result_one + " | " + " Expert B: " + result_two + " | " + " Expert C: " + result_three])

   report_enter2 = enter_chain("***Please Perform Judgment After Reference*** " + Expert_check_system_message[-1][-1])
   result_one = expert_agent1_node(report_enter2)
   result_two = expert_agent2_node(report_enter2)
   result_three = expert_agent3_node(report_enter2)
   Expert_check_system_message.append([" Expert A: " + result_one + " | " + " Expert B: " + result_two + " | " + " Expert C: " + result_three])

   report_enter3 = enter_chain("***Please Perform Judgment After Reference*** " + Expert_check_system_message[-1][-1])
   result_one = expert_agent1_node(report_enter3)
   result_two = expert_agent2_node(report_enter3)
   result_three = expert_agent3_node(report_enter3)
   Expert_check_system_message.append([" Expert A: " + result_one + " | " + " Expert B: " + result_two + " | " + " Expert C: " + result_three])

   return Expert_check_system_message[-1]

def judgment_system(enter):
   result_one = judgment_agent_node(enter)

   return result_one

def question_system(enter):
  result_one = expert_question_node_1(enter)
  Expert_question_system_message.append(["Question Expert : " + result_one])

  return Expert_question_system_message[-1]

def get_final(list):
  j_1 = 0
  j_0 = 0
  for j in list:
     if j == 1:
       j_1 += 1
     else:
       j_0 += 1
     if j_1 > j_0:
       final = 1
     elif j_1 < j_0:
       final = 0
  return final

def check_and_execute(list_elements):
    count_zero = list_elements.count(0)
    count_one = list_elements.count(1)

    if count_zero >= 2:
      final = 0
    elif count_one >= 2:
      final = 1
    return final

def extract_Content_Analysis_Agent_questions(text):
  pattern = re.compile(r'Content Analysis Agent:.*?Comment Analysis Agent:', re.DOTALL)

  match = pattern.search(text)
  if match is not None:
    questions_list = extract_questions(match.group())
    all_questions = ""

    for question in questions_list:
        all_questions += question + '\n'
    if all_questions:
        all_questions = all_questions
    else:
        all_questions = match.group()
  else:
    all_questions = text

  return all_questions



def extract_Comment_Analysis_Agent_questions(text):
  pattern = re.compile(r'Comment Analysis Agent:.*?Fact-Checking Agent:', re.DOTALL)

  match = pattern.search(text)
  if match is not None:
    questions_list = extract_questions(match.group())
    all_questions = ""

    for question in questions_list:
       all_questions += question + '\n'
    if all_questions:
       all_questions = all_questions
    else:
       all_questions = match.group()
  else:
    all_questions = text

  return all_questions


def extract_Fact_Checking_Agent_questions(text):
  pattern = re.compile(r'Fact-Checking Agent:\s*(.*)')
  match = pattern.search(text)
  if match is not None:
    checking_agent_content = match.group()
  else:
    checking_agent_content = text
  return checking_agent_content

from openai import OpenAI

# add your model and api_key
def get_Original_news_embedding(text, model=""):
  client = OpenAI(base_url="",api_key="")
  response = client.embeddings.create(input=[text], model=model, dimensions=512)
  return response.data[0].embedding

def get_experience(input_text):
  match = re.search(r'/%(.+?)%/', input_text)
  if match:
      experience = match.group(1).strip()
  else:
    experience = input_text 
  return experience

def match_number(text):
    pattern = r"Judgment:\s*(.*)"
    match = re.search(pattern, text)
    if match:
        text = match.group(1).strip()
        pattern = r"\d+"
        match = re.search(pattern, text)
        if match:
            return int(match.group(0))
    return None  

def toexact_match(input_text,i):
  rumor_match = re.search(r'\*\*([^\*]+)\*\*', input_text)
  if rumor_match:
      isrumor = rumor_match.group(1)
      label = SequenceMatcher(None, "rumor",isrumor).ratio()
      if label>0.5:
        return 1
      else:
        return 0
  else:
      label = SequenceMatcher(None, "rumor",input_text).ratio()
      if label>0.5:
        return 1
      else:
        return 0



def write_experience_to_file(file_path, data_str):
    single_line_str = ' '.join(data_str.strip().split())

    with open(file_path, 'a') as file:
        file.write(single_line_str + '\n')

def read_experience_from_file(file_path):
    return pd.read_csv(file_path, sep='\t', names=['experience'], encoding='ISO-8859-1')

def cosine_similarity_vectorized(embeddings, new_embedding):
    norms = np.linalg.norm(embeddings, axis=1)  
    new_norm = np.linalg.norm(new_embedding)    
    dot_products = np.dot(embeddings, new_embedding)  
    return dot_products / (norms * new_norm)  
def get_score(item):
    return float(item.split("score:")[-1])


def store_all_case_report(data_idx, all_report, select_class, filename):
    with open(filename, 'a') as file:
        file.write(f"{data_idx}\t{all_report}\n")  
    print("store all case report successfully")

def load_question_emb_from_file(file_name):
    loaded_arrays = []
    with open(file_name, 'rb') as file:
        while True:
            try:
                loaded_array = pickle.load(file)
                loaded_arrays.append(loaded_array)
            except EOFError:
                break
    return np.array(loaded_arrays)[0]
def get_all_report(file_path):
    indices = []
    embeddings = []
    cases = []

    with open(file_path, 'r', errors='replace') as file:
        for line in file:
            parts = line.strip().split('\t')
            idx = int(parts[0])  
            case = parts[-1]  

            indices.append(idx)
            cases.append(case)

    df = pd.DataFrame({
        'idx': indices,
        'cases': cases
    })

    return df

def read_INS_file(filename):
    with open(filename, 'r') as file:
        lines = file.readlines()

    data = []
    text = None
    score = None

    for line in lines:
        if line.startswith("rule:"):
            text = line.strip().split("rule:", 1)[1].strip()
        elif line.startswith("score:"):
            score_value = line.strip().split("score:", 1)[1].strip()
            score = float(score_value)
            data.append([text, score])

    return data

def find_max_score(data):
    max_entry = max(data, key=lambda x: x[1])
    return max_entry

def get_example_library(selected_file):
    # input your report path
    folder_path = ''  
    all_files = [
        'biz.txt',
        'cele.txt',
        'edu.txt',
        'entmt.txt',
        'polit.txt',
        'sports.txt',
        'tech.txt'        
    ]

    selected_df = pd.DataFrame(columns=['id', 'text'])
    combined_df = pd.DataFrame(columns=['id', 'text'])

    for file_name in all_files:
        file_path = os.path.join(folder_path, "judgment_example_library_"+file_name)

        temp_df = pd.read_csv(file_path, sep='\t', header=None, names=['id', 'text'], encoding='utf-8')

        if file_name == selected_file+".txt":
            selected_df = temp_df
        else:
            combined_df = pd.concat([combined_df, temp_df], ignore_index=True)

    return selected_df, combined_df
MAX_TOKENS = 15800

enc = tiktoken.encoding_for_model("gpt-3.5-turbo")

def truncate_text_to_fit_token_limit(text, max_tokens=MAX_TOKENS):
    tokens = enc.encode(text)
    
    if len(tokens) > max_tokens:
        tokens = tokens[:max_tokens]       
        return enc.decode(tokens), len(tokens)  
    return text, len(tokens) 

def check_text(text):
    pattern_0 = r'(0|real)'
    pattern_1 = r'(1|fake)'
    
    if re.search(pattern_0, text):
        return 0
    elif re.search(pattern_1, text):
        return 1
    else:
        return None 

def get_similarity(text, keyword):
    return SequenceMatcher(None, text, keyword).ratio()

def extract_real_fake(report):
    keywords_fake = ["false", "fake"]
    keywords_real = ["true", "real"]
    
    similarity_scores_fake = []
    similarity_scores_real = []
    
    for sentence in report.split('\n'):
        for keyword in keywords_fake:
            similarity_scores_fake.append(get_similarity(sentence, keyword))
        for keyword in keywords_real:
            similarity_scores_real.append(get_similarity(sentence, keyword))
    
    avg_similarity_fake = sum(similarity_scores_fake) / len(similarity_scores_fake) if similarity_scores_fake else 0
    avg_similarity_real = sum(similarity_scores_real) / len(similarity_scores_real) if similarity_scores_real else 0
    
    if avg_similarity_fake > avg_similarity_real:
        return 1
    else:
        return 0

def think_chain_for_test(idx, Best_INS,all_report,Test_pre_labels,original_news):
  Judger = ChatGPTClient(prompt = Best_INS,temperature=0)  
  all_reports = "***All Analysis Reports*** " + all_report + "\n***News*** " + original_news
  text,length = truncate_text_to_fit_token_limit(all_reports)
  print(f"len of input judgement：{length}")
  result = Judger.chat(text)
  if result == None:
     result = "None"
  print(result)
  result_temp = extract_final_result(result)
  temp=None
  if result_temp == 0 or result_temp == 1:
    Test_pre_labels.append((idx, result_temp))
    temp = result_temp
  else:
    number = match_number(result_temp)
    if number == 0 or number ==1:
      Test_pre_labels.append((idx, number))
      temp = number
    else:
      if find_label_in_second_last_paragraph(result) == 0 or find_label_in_second_last_paragraph(result) == 1:
        Test_pre_labels.append((idx,find_label_in_second_last_paragraph(result)))
        temp = find_label_in_second_last_paragraph(result)
      else:
        label = SequenceMatcher(None, "fake-news",result).ratio()
        if label>0.5:
          Test_pre_labels.append((idx,1))
          temp = 1
        else:
          Test_pre_labels.append((idx,0))
          temp = 0
  print("-------------------->RESULT-------------------->",temp)
  print("end of sample inference")
  

import os
import itertools
import pandas as pd
from openai import OpenAI
import time
model = ""

def process_test_samples(select_class, INS_filename, report_file_path, output_dir, all_apis):
    data = read_INS_file(INS_filename)

    
    for i, (Best_INS, Best_score) in enumerate(data):
        print("INS:", Best_INS)
        print("score:", Best_score)
        Test_pre_labels = []

        api_iterator = itertools.cycle(all_apis)
        True_labels = []
        # input your dataset path
        test_df = pd.read_csv(f'')
       
        test_df.rename(columns={'Unnamed: 0': 'idx'}, inplace=True)
            
        
        all_report = get_all_report(report_file_path)
        start_idx = int(test_df.iloc[0]["idx"])

        iteration_count = 0
        end_idx = start_idx

        for data_index, row in test_df.iterrows():
            try:
                if iteration_count == 0 or iteration_count % 3 == 0:
                    next_element = next(api_iterator)
                    api_index = all_apis.index(next_element)
                    os.environ["OPENAI_API_KEY"] = next_element
                    client = OpenAI()
        
                select_report = all_report[all_report['idx'] == int(row[0])]["cases"].values[0]
                think_chain_for_test(int(row[0]), Best_INS, select_report, Test_pre_labels, row[1])
                True_labels.append(int(row[3]))
                end_idx = int(row[0])
                iteration_count += 1

            except Exception as e:
                break
        print("store result")
        Pre_labels_new = Test_pre_labels.copy()
        output_path = f"{output_dir}/{select_class}_{start_idx}_to_{end_idx}_result_{i}_{model}.txt"
        write_list_to_file(Pre_labels_new, output_path)
        print(f"store to {output_path}")



all_apis = [""]
select_classes = ['biz', 'cele', 'edu', 'entmt', 'polit', 'sports', 'tech']

for select_class in select_classes:
    print(select_class)
    # input ins_rule path 
    INS_filename = ''
    # input report path
    report_file_path = ""
    # output path
    output_dir = ""
    process_test_samples(select_class, INS_filename, report_file_path, output_dir,all_apis)
    print(f"test {select_class} complete")