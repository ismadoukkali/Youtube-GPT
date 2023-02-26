import streamlit as st
import whisper
import pandas as pd
from pytube import YouTube
import openai
import numpy as np
import tiktoken
import validators
from transformers import GPT2TokenizerFast
from streamlit_chat import message

EMBEDDING_MODEL = "text-embedding-ada-002"
COMPLETIONS_MODEL = "text-davinci-003"

def count_tokens(text: str) -> int:
    """count the number of tokens in a string"""
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    return len(tokenizer.encode(text))

def create_database(output, title):
    res = []
    segments = output['segments']
    for segment in segments:
        res.append((title, segment['id'], segment['text'], count_tokens(segment['text'])))

    df = pd.DataFrame(res, columns=["title", "heading", "content", "tokens"])
    return df

def get_embedding(text: str, model: str=EMBEDDING_MODEL) -> list[float]:
    result = openai.Embedding.create(
      model=model,
      input=text
    )
    return result["data"][0]["embedding"]

def compute_doc_embeddings(df: pd.DataFrame) -> dict[tuple[str, str], list[float]]:
    return {
        idx: get_embedding(r.content) for idx, r in df.iterrows()
    }

def vector_similarity(x: list[float], y: list[float]) -> float:
    
    return np.dot(np.array(x), np.array(y))

def order_document_sections_by_query_similarity(query: str, contexts: dict[(str, str), np.array]) -> list[(float, (str, str))]:
    
    query_embedding = get_embedding(query)
    
    document_similarities = sorted([
        (vector_similarity(query_embedding, doc_embedding), doc_index) for doc_index, doc_embedding in contexts.items()
    ], reverse=True)
    
    return document_similarities

def construct_prompt(question: str, context_embeddings: dict, df: pd.DataFrame) -> str:
    
    most_relevant_document_sections = order_document_sections_by_query_similarity(question, context_embeddings)
    
    chosen_sections = []
    chosen_sections_len = 0
    chosen_sections_indexes = []
     
    for _, section_index in most_relevant_document_sections:
        # Add contexts until we run out of space.        
        document_section = df.loc[section_index]
        
        chosen_sections_len += document_section.tokens + separator_len
        if chosen_sections_len > MAX_SECTION_LEN:
            break
            
        chosen_sections.append(SEPARATOR + document_section.content.replace("\n", " "))
        chosen_sections_indexes.append(str(section_index))
            
    # Useful diagnostic information
    print(f"Selected {len(chosen_sections)} video sections from database:")
    print("\n".join(chosen_sections_indexes))
    
    header = """Answer the question as truthfully as possible using the provided context, and if the answer is not contained within the text below, say "I don't know."\n\nContext:\n"""
    
    return header + "".join(chosen_sections) + "\n\n Q: " + question + "\n A:"

COMPLETIONS_API_PARAMS = {
    # We use temperature of 0.0 because it gives the most predictable, factual answer.
    "temperature": 1.0,
    "max_tokens": 300,
    "model": COMPLETIONS_MODEL,
}


def answer_query_with_context(
    query: str,
    df: pd.DataFrame,
    document_embeddings: dict[(str, str), np.array],
    show_prompt: bool = False
) -> str:
    prompt = construct_prompt(
        query,
        document_embeddings,
        df
    )
    
    if show_prompt:
        print(prompt)

    response = openai.Completion.create(
                prompt=prompt,
                **COMPLETIONS_API_PARAMS
            )

    return response["choices"][0]["text"].strip(" \n"), prompt


##################### STREAMLIT APP #######################

st.markdown('<h1>ChatGPT x Youtube 🤖<small>by Isma<small </h1>', unsafe_allow_html=True)
st.write("Start a chat with any Youtube video you would like. You just need to add your OpenAI API Key, paste your desired Youtube Video to transcribe and then enjoy chatting with the Bot in the 'Chat with the video tab' .")
api= st.text_input("OpenAI API key, How to get it [here](https://platform.openai.com/account/api-keys)", type = "password")
url = st.text_input("Video URL", value="")
if len(url)> 0:
    if not validators.url(url):
                st.write('Enter a valid URL')
                url = st.text_input("Video URL", value="")
                st.experimental_rerun()
if url and api:
    start_analysis = st.button("Start Analysis")
    if start_analysis:
        st.text('Analising')
tab1, tab2, tab3, tab4 = st.tabs(["Intro", "Transcription", "Embedding", "Chat with the Video"])
flag = False
with tab1:
    st.markdown("### How does it work?")
    st.markdown('Read the following py notebook to know how it works: [Notebook](https://github.com/ismadoukkali/Youtube-GPT/blob/main/youtube-gpt-explanation.ipynb)')
    st.write("Youtube GPT was written with the following tools:")
    st.markdown("#### Streamlit")
    st.write("The design was written with [Streamlit](https://streamlit.io/).")
    st.markdown("#### Whisper")
    st.write("Video transcription is done by [OpenAI Whisper](https://openai.com/blog/whisper/).")
    st.markdown("#### Embedding")
    st.write('[Embedding](https://platform.openai.com/docs/guides/embeddings) is done via the OpenAI API with "text-embedding-ada-002"')
    st.markdown("#### GPT-3")
    st.write('The chat uses the OpenAI API with the [GPT-3](https://platform.openai.com/docs/models/gpt-3) model "text-davinci-003""')
    st.markdown("""---""")
    st.write('Author: [Ismael Doukkali](https://www.linkedin.com/in/ismael-doukkali/)')
    st.write('Repo: [Github](https://github.com/ismadoukkali/Youtube-GPT)')

with tab2:
    model = whisper.load_model('base')
    st.header("Transcription:")
    st.write('Disclaimer: The whisper transcription model will take a couple of minutes transcribing your video. The longer the video, the longer the wait.')
    openai.api_key = api
    if url and api:
        if start_analysis:
            loading_text = st.text('Transcribing, this can take a couple of minutes...')
            youtube_video = YouTube(url)
            streams = youtube_video.streams.filter(only_audio=True)
            stream = streams.first()
            stream.download(filename='audios.mp4')
            video_title = streams[0].title
            output = model.transcribe('audios.mp4')
            st.write(youtube_video.title)
            st.video(url)
            loading_text_2 = st.text('Loading database...')
            df = create_database(output, video_title)
            df.to_csv('transcription.csv', index=False)
            st.write(df.sample(20))
            flag = True
            loading_text_2.text('DONE! Here your transcribed text! From the following video: {} '.format(url))
            loading_text.empty()
            

with tab3:
    st.header("Embeddings:")
    if flag:
        video_embeddings = compute_doc_embeddings(df)
        pd.DataFrame(video_embeddings).to_csv('word_embeddings.csv') 
        example_entry = list(video_embeddings.items())[0]
        st.text('Here an example entry, all loaded succesfully')
        st.write(str(f"{example_entry[0]} : {example_entry[1][:5]}... ({len(example_entry[1])} entries)"))
    else:
        st.text('Data not available yet... head to "Trascription" tab and insert URL & OpenAI API')


with tab4:
    st.header("Chat with the video:")
    MAX_SECTION_LEN = 500
    SEPARATOR = "\n* "
    ENCODING = "gpt2"  # encoding for text-davinci-003

    encoding = tiktoken.get_encoding(ENCODING)
    separator_len = len(encoding.encode(SEPARATOR))
    
    
    df = pd.read_csv ('transcription.csv')
    video_embeddings = compute_doc_embeddings(df)

    if not url and not api:
        st.text('Data not available yet... head to "Trascription" tab and insert URL & OpenAI API')
    
    else:
        if 'generated' not in st.session_state:
            st.session_state['generated'] = []

        if 'past' not in st.session_state:
            st.session_state['past'] = []
        
        def get_text():
            if api:
                st.header("Ask me something about the video:")
                input_text = st.text_input("You: ","", key="input")
                return input_text
        user_input = get_text()
        
        if user_input:
            loading_text_3 = st.text('Sending query to chatgpt, this can take a couple of seconds...')
            output, prompt = answer_query_with_context(user_input, df, video_embeddings)
            # st.write('Output for chatgpt: {}'.format(prompt))
            st.session_state['past'].append(user_input)
            st.session_state['generated'].append(output)
        if st.session_state['generated']:
            for i in range(len(st.session_state['generated'])-1, -1, -1):
                message(st.session_state["generated"][i], key=str(i))
                message(st.session_state['past'][i], is_user=True, key=str(i) + '_user')
