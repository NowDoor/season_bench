import random
from numpy import dot
from numpy.linalg import norm
import numpy as np
import operator
import torch
import numpy as np
from typing import Annotated, Sequence, TypedDict
import configparser

from langchain_core.documents import Document
from langchain_community.vectorstores import Milvus
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain_community.retrievers import BM25Retriever
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from langchain_core.messages import BaseMessage
from langchain import hub
from langchain.output_parsers import PydanticOutputParser
from langchain.prompts import PromptTemplate
from langchain.tools.render import format_tool_to_openai_function
from langchain_core.utils.function_calling import convert_to_openai_tool
from langchain_core.messages import BaseMessage, FunctionMessage
from langchain.output_parsers.openai_tools import PydanticToolsParser
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import ToolInvocation
from langchain_core.output_parsers import StrOutputParser
from langgraph.prebuilt import ToolExecutor
from langchain.tools.retriever import create_retriever_tool
from langgraph.graph import END, StateGraph
from langchain_core.messages import HumanMessage
from dotenv import load_dotenv

config = configparser.ConfigParser()
config.read('config/key.ini')

def exp_normalize(x):
      b = x.max()
      y = np.exp(x - b)
      return y / y.sum()

class Retriever:
    def __init__(self, col_name, llm, top_k):
        self.embedder = HuggingFaceEmbeddings(model_name = 'BAAI/bge-m3',
                                model_kwargs = {"device":'cuda'},
                                encode_kwargs = {'normalize_embeddings': True, 'batch_size': 4})
        self.db = Milvus(
            self.embedder,
            connection_args={"host": "127.0.0.1", "port": config.get('db', 'milvus')},
            collection_name= col_name,)
        

        self.llm = llm
        self.dense_retriever = self.db.as_retriever(
                    search_type='similarity',
                    search_kwargs=({"k": top_k
                    }))
        
        self.docs = self.db.search( 
            query = 'p', 
            search_type = 'mmr' ,
            k =  10e9
        )
        self.bm25_retriever = BM25Retriever.from_documents(documents = self.docs, k = top_k)


    def get_docs(self, message):
        context = ''

        for text in self.dense_retriever.get_relevant_documents(message):
            context += text.page_content
        return context


class Reranker(Retriever):
    def __init__(self, col_name, llm, top_k):
        super().__init__(col_name, llm, top_k)
        self.tokenizer =  AutoTokenizer.from_pretrained('BAAI/bge-reranker-v2-m3')
        self.model = AutoModelForSequenceClassification.from_pretrained('BAAI/bge-reranker-v2-m3')


    def get_reranker_docs(self, message, top_k):
        bm25_results = self.bm25_retriever.get_relevant_documents(message)
        dense_results = self.dense_retriever.get_relevant_documents(message)
        self.model.eval()

        text_pair = []
        for doc in bm25_results:
            text_pair.append([message, doc.page_content])

        flag = False
        for doc in dense_results:
            for pair_doc in text_pair:
                if doc.page_content == pair_doc[1]:
                    flag = True
                    break
            if flag == True: 
                flag = False
                continue
            text_pair.append([message, doc.page_content])

        with torch.no_grad():
            inputs = self.tokenizer(text_target = text_pair, padding=True, truncation=True, return_tensors='pt', max_length=512)
            c_loss = self.model(**inputs, return_dict=True).logits.view(-1, ).float()
            scores = exp_normalize(c_loss.numpy())
        
        score_pair = []
        for idx, score in enumerate(scores):
            score_pair.append([text_pair[idx][1], score])

        score_pair.sort(reverse = True, key= lambda score_pair: score_pair[1])

        relevant_docs = '\n'
        for text in score_pair[:top_k]:
            relevant_docs += text[0]

        return relevant_docs
    
class graph_node(Reranker):
    def __init__(self, col_name, llm, top_k):
        self.check_model = ChatOpenAI(openai_api_key = config.get('key', 'gpt') , 
                                 temperature=0.7,  
                                 model_name="gpt-4o-mini")
        
        self.results = self.db.search( 
            query = 'p', 
            search_type = 'mmr' ,
            k =  10e9
        )
        self.bm25_retriever = BM25Retriever.from_documents(documents = self.results, k = top_k)

        self.tool = create_retriever_tool(
            self.dense_retriever,
            "retrieve",
            "Search Tax Law in vectorDB",
        )

        self.tools = [self.tool]


# 도구들을 실행할 ToolExecutor 객체를 생성합니다.
        self.tool_executor = ToolExecutor(self.tools)
    
    def should_retrieve(self, state):
        messages = state["messages"]
        last_message = messages[-1]
        ''''''
        # 함수 호출이 없으면 종료합니다.
        if "function_call" not in last_message.additional_kwargs:
            return "end"
        # 그렇지 않으면 함수 호출이 있으므로 계속합니다.
        else:
            return "continue"

    def grade_documents(self, state):
        """
        검색된 문서가 질문과 관련이 있는지 여부를 결정합니다.

        Args:
            state (messages): 현재 상태

        Returns:
            str: 문서가 관련이 있는지 여부에 대한 결정
        """

        print("---CHECK RELEVANCE---")

        # 데이터 모델
        class grade(BaseModel):
            """관련성 검사를 위한 이진 점수."""

            binary_score: str = Field(description="'yes' 또는 'no'의 관련성 점수")

        # 도구
        grade_tool_oai = convert_to_openai_tool(grade)

        # 도구와 강제 호출을 사용한 LLM
        k = self.check_model
        llm_with_tool = self.check_model.bind(
            tools=[convert_to_openai_tool(grade_tool_oai)],
            tool_choice={"type": "function", "function": {"name": "grade"}},
        )

        # 파서
        parser_tool = PydanticToolsParser(tools=[grade])

        prompt = PromptTemplate(
            template="""You are a grader assessing relevance of a retrieved document to a user question. \n 
            Here is the retrieved document: \n\n {context} \n\n
            Here is the user question: {question} \n
            If the document contains keyword(s) or semantic meaning related to the user question, grade it as relevant. \n
            Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question.""",
            input_variables=["context", "question"],
        )

        # 체인
        chain = prompt | llm_with_tool | parser_tool

        messages = state["messages"]
        last_message = messages[-1]

        question = messages[0].content
        docs = last_message.content

        score = chain.invoke({"question": question, "context": docs})

        grade = score[0].binary_score

        if grade == "yes":
            print("---DECISION: DOCS RELEVANT---")
            return "yes"

        else:
            print("---DECISION: DOCS NOT RELEVANT---")
            print(grade)
            return "no"
        

    def agent(self, state):
        messages = state["messages"]
        model = ChatOpenAI(openai_api_key=config.get('key', 'gpt') , 
                                 temperature=0.7,  
                                 model_name="gpt-4o")
        functions = [format_tool_to_openai_function(t) for t in self.tools]
        model = model.bind_functions(functions)
        response = model.invoke(messages)
        # 이것은 기존 목록에 추가될 것이므로 리스트를 반환합니다.
        return {"messages": [response]}\
    
    def retrieve(self, state):
        print("---EXECUTE RETRIEVAL---")
        messages = state["messages"]
        # 계속 조건을 기반으로 마지막 메시지가 함수 호출을 포함하고 있음을 알 수 있습니다.
        last_message = messages[-1]
        # 함수 호출에서 ToolInvocation을 구성합니다.
        action = ToolInvocation(
            tool=last_message.additional_kwargs["function_call"]["name"],
            tool_input=json.loads(
                last_message.additional_kwargs["function_call"]["arguments"]
            ),
        )
        # 도구 실행자를 호출하고 응답을 받습니다.
        response = self.tool_executor.invoke(action)
        function_message = FunctionMessage(content=str(response), name=action.tool)

        # 이것은 기존 목록에 추가될 것이므로 리스트를 반환합니다.
        return {"messages": [function_message]}
    
    def rewrite(self, state):

        print("---TRANSFORM QUERY---")
        messages = state["messages"]
        question = messages[0].content

        msg = [
            HumanMessage(
                content=f""" \n 
        Look at the input and try to reason about the underlying semantic intent / meaning. \n 
        Here is the initial question:
        \n ------- \n
        {question} 
        \n ------- \n
        Formulate an improved question: """,
            )
        ]

        # 평가자
        model = ChatOpenAI(openai_api_key= config.get('key', 'gpt') , 
                                 temperature=0.7,  
                                 model_name="gpt-4o")
        response = model.invoke(msg)
        return {"messages": [response]}



class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]

class langgraph:
    def __init__(self,col_name, top_k):
        load_dotenv()
        self.workflow = StateGraph(AgentState)
        node = graph_node(col_name, top_k)
        self.workflow.add_node("agent", node.agent)  # 에이전트 노드를 추가합니다.
        self.workflow.add_node("retrieve", node.retrieve)  # 정보 검색 노드를 추가합니다.
        self.workflow.add_node("rewrite", node.rewrite)  # 정보 재작성 노드를 추가합니다.

        self.workflow.set_entry_point("agent")

        # 검색 여부 결정
        self.workflow.add_conditional_edges(
            "agent",
            # 에이전트 결정 평가
            node.should_retrieve,
            {
                # 도구 노드 호출
                "continue": "retrieve",
                "end": END,
            },
        )

        # `action` 노드 호출 후 진행될 경로
        self.workflow.add_conditional_edges(
            "retrieve",
            # 에이전트 결정 평가
            node.grade_documents,
            {
                "yes": END,
                "no": "rewrite",
            },
        )
        self.workflow.add_edge("rewrite", "agent")

        self.app = self.workflow.compile()

    def get_lg_docs(self, message):
        context = ''
        for output in self.app.stream(message):
            output = dict(output)
            if 'retrieve' in output:
                context += str(output['retrieve']['messages'][-1].content)
            elif 'agent' in output:
                context += str(output['agent']['messages'][-1].content)
            else:
                continue

        return context




