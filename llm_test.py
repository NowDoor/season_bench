import random
import pandas as pd
import itertools
import configparser
import matplotlib.pyplot as plt
import gradio as gr
import pickle
import os

from langchain.chat_models import ChatOpenAI
from langchain.schema import AIMessage, HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_anthropic import ChatAnthropic

import elo
import milvus_manager as mm
from set_retriver import Retriever
from set_retriver import Reranker



config = configparser.ConfigParser()
config.read('config/key.ini')

llm_list = {'gpt-4o': ChatOpenAI(openai_api_key = config.get('key' , 'gpt') , 
                                 temperature=0,  
                                 model_name="gpt-4o"), 
        #'gpt-3.5-turbo' : ChatOpenAI(openai_api_key = config.get('key' , 'gpt') , 
        #                             temperature=0,  
        #                             model_name="gpt-3.5-turbo"),
        'claude-3-5-sonnet-20240620' : ChatAnthropic(model='claude-3-5-sonnet-20240620', 
                                                     anthropic_api_key = config.get('key' , 'claude') , 
                                                     temperature=0),
        #'llama-3-Korean-Bllossom-8B': ChatOpenAI(model='MLP-KTLim/llama-3-Korean-Bllossom-8B',
        #                         openai_api_base = config.get('port' , '0'), openai_api_key='EMPTY', # vllm 
        #                         temperature=0),
        #'EEVE-Korean-10.8B': ChatOpenAI(openai_api_base = config.get('port' , '01'), openai_api_key='EMPTY', model_name='EEVE-Korean-10.8B:latest', temperature=0),
        'llama3.1-70b': ChatOpenAI(openai_api_base = config.get('port' , '1'), openai_api_key='EMPTY', model_name='hugging-quants/Meta-Llama-3.1-70B-Instruct-AWQ-INT4', temperature=0),
        #'wizardLM-2-8x22B': ChatOpenAI(openai_api_base= config.get('port' , 'ex'), 
        #    openai_api_key= config.get('key' , 'wizard'), 
        #    temperature= 0, 
        #    top_p= 1 , 
        #    model_name="microsoft/wizardlm-2-8x22b"),
        'gpt-4o-mini' : ChatOpenAI(openai_api_key=config.get('key' , 'gpt') , 
                                 temperature=0,  
                                 model_name="gpt-4o-mini"), 
        #'llama-3.1-8b': ChatOpenAI(model='meta-llama/Meta-Llama-3.1-8B-Instruct',
        #                         openai_api_base = config.get('port' , '1'), openai_api_key='EMPTY', # vllm 
        #                         temperature=0),
        #'Qwen2-72B-Instruct': ChatOpenAI(model='Qwen/Qwen2-72B-Instruct-GPTQ-Int4',
        #                         openai_api_base = config.get('port' , '1'), openai_api_key='EMPTY', # vllm 
        #                         temperature=0),    
}


llm1, llm2 = random.sample(list(llm_list.items()) , 2)

#일반 retriever 사용
def response(message, rag_data):
        if rag_data == 'None':
            prompt = ChatPromptTemplate.from_messages([
            ("system", ""),
            
            ("user", "{input}")
            ])
            chain1 = prompt | llm1[1]
            chain2 = prompt | llm2[1]
        else :
            retriever = Retriever(rag_data, llm1[1], 5)

            context = retriever.get_docs(message)

            prompt = ChatPromptTemplate.from_messages([
            ("system", f"내가 제공하는 문서 기반으로 답변해\n \
             {context}"),
            
            ("user", "{question}")
            ])
            
            chain1 = ({"question": RunnablePassthrough()} 
                    | prompt
                    | llm1[1])
            
            chain2 = ({"question": RunnablePassthrough()} 
                    | prompt
                    | llm2[1])
        
        gpt_response1 = chain1.stream(message)
        gpt_response2 = chain2.stream(message)
        history1, history2 = '', ''
        for response1, response2 in itertools.zip_longest(gpt_response1, gpt_response2):
            if not response1 == None:
                history1 += response1.content
            if not response2 == None:
                history2 += response2.content
            yield history1, history2
    

def selct_interactive(submit):
    return gr.Button(interactive=True), gr.Button(interactive=True), gr.Button(interactive=True), gr.Button(interactive=False)

def select_model(select):
    try:
        df = pd.read_csv('data/score_files/llm_score.csv')
    except:
        df = pd.DataFrame({'llm':[], 'score':[], 'vote':[]})

    if not (df['llm'] == llm1[0]).any():
        df.loc[df.shape[0]] = [llm1[0], 1000, 0]

    if not (df['llm'] == llm2[0]).any():
        df.loc[df.shape[0]] = [llm2[0], 1000, 0]

    llm1_score = df[df['llm'] == llm1[0]]['score'].iloc[0]
    llm1_vote =  df[df['llm'] == llm1[0]]['vote'].iloc[0]

    llm2_score =  df[df['llm'] == llm2[0]]['score'].iloc[0]
    llm2_vote =  df[df['llm'] == llm2[0]]['vote'].iloc[0]
    
    if select == 'Model-A':
        llm1_score, llm2_score = elo.calculate_elo(llm1_score, llm2_score, 1, 128)
    elif select == 'Model-B':
        llm1_score, llm2_score = elo.calculate_elo(llm1_score, llm2_score, 0, 128)
    elif select == 'Tie':
        llm1_score, llm2_score = elo.calculate_elo(llm1_score, llm2_score, 0.5, 128)

    llm1_vote += 1
    llm2_vote += 1
    
    df.loc[df['llm'] == llm1[0], 'score']= int(llm1_score)
    df.loc[df['llm'] == llm1[0], 'vote']= int(llm1_vote)
    df.loc[df['llm'] == llm2[0], 'score']= int(llm2_score)
    df.loc[df['llm'] == llm2[0], 'vote']= int(llm2_vote)
    
    df.to_csv('data/score_files/llm_score.csv', index = False)

    return gr.Textbox(label = llm1[0]), gr.Textbox(label = llm2[0]), gr.Button(interactive=False), gr.Button(interactive=False), gr.Button(interactive=False), gr.Button(interactive=True)

def retry(btn):
    global llm1, llm2
    llm1, llm2 = random.sample(list(llm_list.items()) , 2)
    return gr.Textbox('',label = 'modelA'), gr.Textbox('',label = 'modelB'), gr.Textbox(''),gr.Button(interactive=False), gr.Button(interactive=True)

def bench_clicked(bench_test, bench_llm, rag_data):
    llm = llm_list[bench_llm]
    df = pd.read_csv(os.path.join('data','bench_files',bench_test+'.csv'))
    #df = df.sample(frac=1)
    top_k = 5
    top_p = 5
    if not 'category' in df.columns:
        df['category'] = bench_test

    datasets = []
    for idx,data in df.iterrows():
        prompt = '\n'.join([data['prompt'], 'A :' + data['A'],'B :' +  data['B'], 'C :' + data['C'], 'D :' + data['D'], 'E :' + data['E']])
        datasets.append([prompt, data['answer'], data['category']])            
        
    
    category_score = {}
    category_idx = {}

    for key in df['category'].unique():
        category_score[key] = 0
        category_idx[key] = 0

    df = pd.DataFrame({'Idx': [], 'Acc':[]})
    fig = plt.figure()
    plt.title(f'{bench_test} Score!')

    text_score = f'Idx : 0, Score : 0'
    for idx, data in enumerate(datasets):
        if rag_data == 'None':
            prompt = ChatPromptTemplate.from_messages([
            ("system", """
            주어지는 문제에 대한 맞는 답을 알파벳 하나만 출력해야 돼
            해답에 대한 설명과 문제 내용은 쓰지마
            대답 예시: A
            """),
            ("user", "{input}")])

        else :
            retriever = Retriever(rag_data, llm,  top_k)
            #retriever = Reranker(rag_data, llm,  8)
            #context = retriever.get_reranker_docs(data[0], top_p)
            context = retriever.get_docs(data[0])
            print(context)
            prompt = ChatPromptTemplate.from_messages([
            ("system", f"{context}"),
            
            ("user", " 참조 : 라고 써있는 부분은 답변하기 어려울 것을 생각해서 참고 자료를 너에게 주는거야\
            주어지는 문제에 대한 맞는 답을 알파벳 하나만 출력해야 돼\
            해답에 대한 설명과 문제 내용은 쓰지마\
            대답 예시: A\
            문제 : \n{input}")
            ])
        
        chain = prompt | llm

        result = chain.invoke({"input": data[0]})

        
        if data[1] in result.content:
            category_score[data[2]] += 1

        category_idx[data[2]] += 1

        temp = [idx+1, round((sum(category_score.values()) / (idx + 1))*100, 2)]
        df.loc[df.shape[0]] = temp
        plt.clf()
        plt.title(f'{bench_test} Category Score!')

        y = []
        plt.ylim(0,100)
        plt.xticks(rotation=20)
        for score, count in zip(category_score.values(), category_idx.values()):
            y.append(score / count * 100 if count != 0 else 0)
        plt.bar(category_score.keys(), y)

        if ((idx+1) % 10 == 0):
            text_score = f'Idx : {temp[0]}, Score : {temp[1]}'

        if idx+1 == len(datasets):
            with open(f'output/bench_log/{bench_test}_{bench_llm}_{rag_data}_{top_k}','wb') as fw:
                pickle.dump(category_score, fw)
            yield fig, f'Idx : {temp[0]}, Score : {temp[1]}'
        else:
            yield fig, text_score



def score_selected():
    try:
        llm_score = pd.read_csv('data/score_files/llm_score.csv')
    except:
        llm_score = pd.DataFrame({'llm':[], 'score':[], 'vote':[]})

    try:
        llm_detail = pd.read_csv('data/llm_detail.csv')
    except:
        llm_detail = pd.DataFrame({'llm' : [],'Organization' : []})

    df = pd.merge(llm_score, llm_detail, on = 'llm', how = 'left') 
    
    return gr.DataFrame(df.sort_values(by=['score'], axis = 0, ascending=False))


markdown = '''
<h1 style="font-size: 24px; font-family: 'Nanum Pen Script';">GPT 성능 측정기</h1>
        
        <p style="font-size: 18px; font-family: 'Nanum Pen Script';">
        상대적인 정성 평가로 점수 산출
        </p>
        <p style="font-size: 18px; font-family: 'Nanum Pen Script';">
        <p style="font-size: 18px; font-family: 'Nanum Pen Script';">
        submit 이후에 답 보고 모델 셀랙해야 리트라이 가능함 \n
        </p>
        <p style="font-size: 18px; font-family: 'Nanum Pen Script';">
        히스토리 기능 제공 없음
        </p>
        

        <ul style="font-size: 16px; font-family: 'Nanum Pen Script';">
            <li>make according to LMsys</li>
            <li>LMsys: <a href="https://huggingface.co/spaces/lmsys/chatbot-arena-leaderboard">https://huggingface.co/spaces/lmsys/chatbot-arena-leaderboard</a></li>
        </ul>
        
        <p style="font-size: 20px; font-family: 'Nanum Pen Script';">🐶🐱🐭🐹🐰</p>
'''


with gr.Blocks() as app:
    with gr.Tab('LLM_Arena') as LLM_Arena:
        gr.HTML(markdown)
        with gr.Row():
            out1 = gr.Textbox(label = 'modelA', lines = 20)
            out2 = gr.Textbox(label = 'modelB', lines = 20)

        with gr.Row():
            inp = gr.Textbox(label = 'prompt', min_width = 1000)
            submit= gr.Button("Submit")
        

        with gr.Row():
            modelA = gr.Button("Model-A", interactive=False)
            modelB = gr.Button("Model-B", interactive=False)
            tie = gr.Button("Tie", interactive=False)

        with gr.Row():
            btn_retry = gr.Button('다시하기', interactive=False)
        
        manager = mm.MilvusManager()
        db_list = ["None"] + manager.list_collection()
        rag_data = gr.Dropdown(
            db_list, label="vectorstore",value='None' ,info = "Retriever 검색을 적용하는 VectorStore를 골라주세요(None = No RAG)"
        )
        
        #frist_step
        submit.click(fn = response, inputs = [inp, rag_data], outputs = [out1, out2])
        submit.click(fn = selct_interactive, inputs = submit, outputs= [modelA, modelB, tie, submit])
        
        '''
        if flag == True:
            inp.submit(fn = response , inputs = inp, outputs = [out1, out2])
            inp.submit(fn = selct_interactive, inputs = submit, outputs= [modelA, modelB, tie, submit])
        '''

        #second_step
        modelA.click(fn=select_model, inputs = modelA, outputs=[out1, out2, modelA, modelB, tie, btn_retry])
        modelB.click(fn=select_model, inputs = modelB, outputs=[out1, out2, modelA, modelB, tie, btn_retry])
        tie.click(fn=select_model, inputs = tie, outputs=[out1, out2, modelA, modelB, tie, btn_retry])

        #retry
        btn_retry.click(fn = retry, inputs = btn_retry, outputs = [out1, out2, inp, btn_retry, submit])

    with gr.Tab('LLM_Bench'):
        bench_graph = gr.Plot()
        bench_score = gr.Textbox(label= 'Score') 
        file_list = os.listdir('data/bench_files')
        file_name = []
        for file in file_list:
            if file.count(".") == 1:
                name = file.split('.')[0]
                file_name.append(name)
            else:
                for k in range(len(file)-1,0,-1):
                    if file[k] == '.':
                        file_name.append(file[:k])
                        break
        bench_test = gr.Dropdown(file_name, label = 'test_data', value = 'hi', info = 'Test할 데이터 셋을 골라주세요' , interactive= True)
        manager = mm.MilvusManager()
        db_list = ["None"] + manager.list_collection()
        bench_llm = gr.Dropdown(list(llm_list.keys()), label='llm_liest', value=list(llm_list.keys())[0], info = 'Test할 모델을 골라주세요', interactive= True)
        rag_bench_data = gr.Dropdown(
            db_list, label="vectorstore",value='None' ,info = "Retriever 검색을 적용하는 VectorStore를 골라주세요(None = No RAG)", interactive= True
        )
        bench_submit = gr.Button('Start')

        bench_submit.click(fn = bench_clicked, inputs = [bench_test, bench_llm, rag_bench_data], outputs=[bench_graph, bench_score])

    with gr.Tab('LLM_Score') as LLM_Bench:
        try:
            llm_score = pd.read_csv('data/score_files/llm_score.csv')
        except:
            llm_score = pd.DataFrame({'llm':[], 'score':[], 'vote':[]})

        try:
            llm_detail = pd.read_csv('data/llm_detail.csv')
        except:
            llm_detail = pd.DataFrame({'llm' : [],'Organization' : []})

        df = pd.merge(llm_score, llm_detail, on = 'llm', how = 'left') 
        score = gr.DataFrame(df.sort_values(by=['score'], axis = 0, ascending=False))
    
    LLM_Bench.select(fn = score_selected, outputs = score)
    
#app.launch(server_name= '220.82.71.6', server_port = 5656)

app.launch()