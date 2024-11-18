import streamlit as st
import json
import requests
import pandas as pd
# from st_aggrid import AgGrid, GridOptionsBuilder, GridUpdateMode, ColumnsAutoSizeMode
import io
import base64
import shutil
import os

from dotenv import dotenv_values, load_dotenv
load_dotenv()

from langchain_community.document_loaders import AzureAIDocumentIntelligenceLoader

st.set_page_config(page_title='Financial Agent Q&A',layout='wide',
# initial_sidebar_state='collapsed'
)

def show_pdf(uploaded_file):
    # with open(file_path,"rb") as f:
    with io.BytesIO() as buffer:
        buffer.write(uploaded_file.read())
        buffer.seek(0)
        base64_pdf = base64.b64encode(buffer.read()).decode('utf-8')

        # pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="800" height="800" type="application/pdf"></iframe>'
        pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="100%" height="800" type="application/pdf"></iframe>'
        st.markdown(pdf_display, unsafe_allow_html=True)

def save_uploaded_file(uploaded_file, folder_path, overwrite_folder=True):
    file_path = os.path.join(folder_path, uploaded_file.name)
    
    if overwrite_folder:
        if os.path.exists(folder_path):
            shutil.rmtree(folder_path)  # Remove the existing folder and all its contents
        os.makedirs(folder_path)  # Create a new folder
    
    if not os.path.exists(file_path):  # Check if the file doesn't exist
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        st.success("File saved successfully!")
    else:
        st.warning(f"File '{uploaded_file.name}' already exists in the target folder.")
    
    return file_path

@st.cache_data(show_spinner=False)
def load_table_defs():
    with open("app_streamlit/table_defs.json") as json_file:
        return json.load(json_file)

def get_source_text(x):
    pre_text = '\n'.join(x['pre_text'])
    table = x['table']
    post_text = '\n'.join(x['post_text'])
    return f"{pre_text}\n\ntable:\n\n{table}\n\n{post_text}"

def split_qa(x):
    qa = x['qa']
    qa_0 = x['qa_0']
    qa_1 = x['qa_1']

    qa_list = []

    if str(qa) != 'nan':
        qa_list.append(qa)
    else:
        if str(qa_0) != 'nan':
            qa_list.append(qa_0)
        if str(qa_1) != 'nan':
            qa_list.append(qa_1)
    
    return qa_list

@st.cache_data(show_spinner=False)
def process_document(text, question, ground_truth):
    # url = 'http://localhost:5000/process_document'
    # url = 'http://localhost:5000/process_document_advanced'
    # url = 'http://localhost:5000/process_document_advanced_agent'
    # url = 'http://flask_app:5000/process_document'
    # url = 'http://flask_app:5000/process_document_advanced'
    url = 'http://flask_app:5000/process_document_advanced_agent'
    payload = {
        'text': text,
        'question': question,
        'ground_truth': ground_truth,
    }
    with st.sidebar:
        with st.expander("Payload",expanded=False):
            st.write(payload)

    response = requests.post(url, json=payload)
    if response.status_code == 200:
        return response.json()['response']
    else:
        st.write(response.content)
        return f"Error: {response.status_code}"

@st.cache_data(show_spinner=False)
def load_data():
    with open("notebooks/train.json") as json_file:
        data_json = json.load(json_file)

    data = pd.DataFrame(data_json)
    data['source_text'] = data.apply(get_source_text, axis=1)
    data['qa_exploded'] = data.apply(split_qa, axis=1)
    data = data.explode('qa_exploded', ignore_index=True)
    data['question'] = data['qa_exploded'].apply(lambda x: x['question'] if str(x)!='nan' else None)
    data['answer'] = data['qa_exploded'].apply(lambda x: x['answer'] if str(x)!='nan' else None)

    return data.drop(columns=['qa_exploded','id','pre_text','post_text','table','table_ori','filename','annotation','qa_0','qa_1','qa'])

@st.cache_data(show_spinner=False)
def load_doc(file_path):

    parser = AzureAIDocumentIntelligenceLoader(
    api_endpoint=dotenv_values().get('AZURE_DOC_INTELLIGENCE_ENDPOINT'),
    api_key=dotenv_values().get('AZURE_DOC_INTELLIGENCE_API_KEY'),
    file_path=file_path,
    api_model="prebuilt-layout",
    api_version=dotenv_values().get('AZURE_DOC_INTELLIGENCE_API_VERSION')
    )

    text = parser.load()[0].page_content

    return text


def user_inputs():
    # Number of fields to be added
    # field_count = st.number_input("Number of fields", min_value=1, max_value=20, value=3)
    field_count = 1

    # Create a DataFrame with the required columns
    columns = ["Field Name", "Description", "Default Value", "Data Type"]
    data = pd.DataFrame(columns=columns)

    # Populate the DataFrame with empty rows based on field_count
    data = data.reindex(range(int(field_count)), fill_value="")

    data = pd.DataFrame({
        "Question": ["" for _ in range(field_count)],
        "Ground Truth": ["" for _ in range(field_count)],
        # "Default Value": [None for _ in range(field_count)],  # Default to NoneType
        # "Data Type": ["str" for _ in range(field_count)]
    })

    # Edit the data using st.data_editor
    # edited_data = st.data_editor(data, num_rows="dynamic", use_container_width=True)
    edited_data = st.data_editor(data, use_container_width=True)

    # Convert the edited data to a dictionary
    field_dict = {}
    for _, row in edited_data.iterrows():
        field_dict = {
            'question': row["Question"],
            'ground_truth': row["Ground Truth"] 
        }
    
    return field_dict

st.title("Financial Document Q&A")

# Sidebar contents
with st.sidebar:
    st.title('🤗💬\nFinancial Doc Q&A Tool')
    st.markdown('''
    ## About
    This app is an LLM-powered document reader built using:
    - [Streamlit](https://streamlit.io/)
    - [LangChain](https://python.langchain.com/)
    - [LangGraph](https://www.langchain.com/langgraph)
    - [OpenAI](https://platform.openai.com/docs/models) LLM model

    ''')

tab1, tab2 = st.tabs(['Training Data','File Upload'])
response_t1 = None
response_t2 = None

with tab1:

    df = load_data()
    st.dataframe(df,use_container_width=True)

    table_defs = load_table_defs()
    grid_options_breaking_news = table_defs["train_data"]

    # Display dataframe and allow user to select a single row
    selected_row_index = st.selectbox("Select a row to process", df.index.tolist())
    
    # Get the selected row as a single row dataframe
    selected_row = df.loc[selected_row_index]
    text = selected_row['source_text']
    # qa = selected_row['qa_exploded']
    question = selected_row['question']
    answer = selected_row['answer']

    st.write(f'**Question:** {question}')
    st.write(f'**Ground-truth:** {answer}')
    with st.expander("**Show Source Text**",expanded=False):
        with st.container(height=500):
            st.text(text)

    custom_col1, custom_col2 = st.columns(2)
    with custom_col1:
        question_custom = st.text_input("Custom Question", value=question)
    with custom_col2:
        ground_truth_custom = st.text_input("Custom Ground-truth", value=answer)

    if question_custom:
        question = question_custom
    if ground_truth_custom:
        answer = ground_truth_custom

    if st.button("Process Selected Row"):
        # Process the selected row
        with st.spinner('Processing...'):
            response_t1 = process_document(text, question, answer)

    if response_t1:
        messages = response_t1['messages']
        # score_percent = str(float(response_t1['overall_score']) * 100) + '%'

        st.write(f'**Agent Answer:** {response_t1["response"]}')
        st.write(f'**Score:** {response_t1["overall_score"]}')
        # st.write(f'**Score:** {score_percent}')

        t1_col1, t1_col2 = st.columns(2)
        with t1_col1:
            with st.expander('**Detailed Score**',expanded=True):
                st.write(response_t1['detailed_score'])
        with t1_col2:
            with st.expander("**Show Agent Flow**"):
                with st.container(height=300):
                    for c,m in enumerate(messages):
                        st.text(f'{c}) {m}')
            with st.expander("**Show Full Result**"):
                with st.container(height=300):
                    st.write(response_t1)
            # with st.expander("**Show Source Text**",expanded=False):
            #     with st.container(height=500):
            #         st.text(text)
        
        if st.button('Reset'):
            st.cache_data.clear()

with tab2:
    file = st.file_uploader("Upload your File", type=['pdf', 'docx','txt'])

    if file is not None:

        file_path = save_uploaded_file(file,'samples/temp_docparser/')
        text = None

        if '.pdf' in file.name:
            show_pdf(file)

        field_dict = user_inputs()

        submitted = st.button("Ask Agent!")

        if submitted:
            with st.spinner('Reading the file...'):
                text = load_doc(file_path)

            if (field_dict is not None):
                # st.write(field_dict)
                question = field_dict['question']
                ground_truth = field_dict['ground_truth']
                try:
                    with st.spinner('Answering Question...'):
                        response_t2 = process_document(text, question, ground_truth)
                        st.success("Response Generated Successfully!")
                except Exception as e:
                    st.error(f"Error: {e}")
    
    if response_t2:
        st.write(f'**Question:** {question}')
        st.write(f'**Ground-truth:** {ground_truth}')
        messages = response_t2['messages']
        st.write(f'**Agent Answer:** {response_t2["response"]}')
        st.write(f'**Score:** {response_t2["overall_score"]}')

        # st.write({k:v for k,v in response_t2.items() if k != 'messages'})
        # with st.expander("Show Agent Flow"):
        #     for c,m in enumerate(messages):
        #         st.write(f'{c}) {m}')
        t2_col1, t2_col2 = st.columns(2)
        with t2_col1:
            with st.expander('**Detailed Score**',expanded=True):
                st.write(response_t2['detailed_score'])
        with t2_col2:
            with st.expander("**Show Agent Flow**"):
                with st.container(height=300):
                    for c,m in enumerate(messages):
                        st.text(f'{c}) {m}')
            with st.expander("**Show Full Result**"):
                with st.container(height=300):
                    st.write(response_t2)
            with st.expander("**Show Source Text**",expanded=False):
                with st.container(height=500):
                    st.text(text)

        if st.button('Reset'):
            st.cache_data.clear()