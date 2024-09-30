
# PDFAsk

PDFAsk is a Streamlit application that allows users to upload PDF files, extract text, and ask questions based on the content of the PDFs. The application uses Google Generative AI for question-answering and includes speech recognition for user input. Additionally, it provides audio output for the answers.

## Features

- Upload multiple PDF files and extract text.
- Ask questions based on the content of the PDFs.
- Use speech recognition to input questions.
- Get audio output for the answers.

## Installation

1. Clone the repository:
   ```sh
   git clone https://github.com/rufevean/pdfask.git
   cd pdfask
2. Create a virtual environment and activate it:
    ```
    pyenv activate venv2
    ```
3. Install the required packages:
    ```
    pip install -r requirements.txt
    ```
4. Setup environment variables:
    ```
    export GOOGLE_APPLICATION_CREDENTIALS="path/to/your/credentials.json"
    GOOGLE_API_KEY="your_google_api_key"
    ```

## Usage

1. Run the Streamlit application:
    ```
    streamlit run app.py
    ```

2. Open your web browser and go to http://localhost:8501.

3. Upload your PDF files using the sidebar.

4. Once the PDFs are processed, you can ask questions in the text input box or use the speech recognition feature to input your question.

5. The answer will be displayed on the screen, and you can also listen to the audio output of the answer.


## Dependencies

streamlit
PyPDF2
langchain
langchain-google-genai
google-generativeai
langchain-community
python-dotenv
speechrecognition
gtts
faiss-cpu
