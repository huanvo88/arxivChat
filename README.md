# Chat with arXiv Documents

Chat with arXiv documents using GPT models from OpenAI ("gpt-3.5-turbo" or "gpt-4")

## Instructions
- First we clone the repo
```
git clone https://github.com/huanvo88/arxivChat.git
cd arxivChat
```
- Create a python virtual environment
```
python3 -m venv venv
source venv/bin/activate
```
- Install the requirements 
```
pip install -e .
```
- Create `.env` file with your `OPENAI_API_KEY` (recommended) or simply export it
```
export OPENAI_API_KEY={Your API Key here}
```
- Test that you have access to OpenAI API by running the notebook [test_nb](notebooks/test_nb.ipynb). 
- Launch the streamlit app with your choice of port (default is **8501**)
```
streamlit run chat_st.py --server.port 8010
```
- Access the chat interface at `http://localhost:8010`
