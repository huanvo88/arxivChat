# Chat with URLs

Chat with URLs using GPT models from OpenAI ("gpt-3.5-turbo" or "gpt-4")

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

## Notes
- One can select `gpt-3.5-turbo` or `gpt-4` in the UI. Even though `gpt-4` is much more expensive, it is better at keeping track of the chat history. 
- One also has the option to chat `with context` or `without context`, but need to refresh the streamlit app and clear the cache (press the letter `C`) when switching.
- To test the model, ask the question: "What is the GDP per capita of Canada in 2023?" If the model does not know the answer, augment with the url [Economy of Canada](https://en.wikipedia.org/wiki/Economy_of_Canada).
