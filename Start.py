import base64
import os
import tempfile
import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    CSVLoader
)
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

# App config
st.set_page_config(page_title="Streamlit Chatbot", page_icon="🤖", layout="wide")
st.title("Chatbot with File Context & Charts")

# Initialize session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
        AIMessage(content="Hello! Upload a file. If it's a CSV, I can also visualize it!"),
    ]

if "document_context" not in st.session_state:
    st.session_state.document_context = ""

if "active_dataframe" not in st.session_state:
    st.session_state.active_dataframe = None


def save_uploaded_file(uploaded_file):
    try:
        suffix = f".{uploaded_file.name.split('.')[-1]}"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            return tmp_file.name
    except Exception as e:
        st.error(f"Error saving file: {e}")
        return None


def process_file(uploaded_file):
    """Process the file and update session state context."""
    temp_file_path = save_uploaded_file(uploaded_file)
    if temp_file_path:
        loader = None
        text_content = ""
        # Normalize extension for comparison
        file_name = uploaded_file.name.lower()

        # Reset dataframe state on new file upload
        st.session_state.active_dataframe = None

        try:
            # 1. Handle CSVs specially for Charting AND Context
            if file_name.endswith(".csv"):
                # Load into Pandas for Charts
                df = pd.read_csv(temp_file_path)
                st.session_state.active_dataframe = df

                # Load into LangChain for Chat
                loader = CSVLoader(temp_file_path)

            elif file_name.endswith(".pdf"):
                loader = PyPDFLoader(temp_file_path)
            elif file_name.endswith(".txt"):
                loader = TextLoader(temp_file_path)
            elif file_name.endswith((".png", ".jpg", ".jpeg")):
                # Reverted to your requested model
                llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

                with open(temp_file_path, "rb") as image_file:
                    image_data = base64.b64encode(image_file.read()).decode("utf-8")

                mime_type = uploaded_file.type if uploaded_file.type else "image/jpeg"

                msg = HumanMessage(
                    content=[
                        {"type": "text",
                         "text": "Extract all text from this image and provide a detailed description of its visual contents to serve as context for a chatbot."},
                        {"type": "image_url", "image_url": {"url": f"data:{mime_type};base64,{image_data}"}}
                    ]
                )

                with st.spinner("Analyzing image with Gemini Vision..."):
                    response = llm.invoke([msg])
                    text_content = response.content

            # Handle Document Loaders (PDF, TXT, CSV)
            if loader:
                docs = loader.load()
                text_content = "\n".join([doc.page_content for doc in docs])

            # Update session state if content was extracted
            if text_content:
                st.session_state.document_context = text_content
                st.success(f"File '{uploaded_file.name}' processed! Context loaded.")
            else:
                st.warning("Could not extract text from file or format not supported.")

        except Exception as e:
            st.error(f"Error loading file: {e}")

        finally:
            # Clean up temp file
            if os.path.exists(temp_file_path):
                os.unlink(temp_file_path)


def get_response(user_query, chat_history, context):
    template = """
    You are a helpful assistant. Answer the following questions considering the history of the conversation, and the user's question based strictly on the context provided below.
    If the user asks about data visualization, guide them to the 'Data Visualization' tab.

    If the answer is not in the context, say I only answer questions about uploaded files.

    Context:
    {context}

    Chat history: {chat_history}

    User question: {user_question}
    """

    prompt = ChatPromptTemplate.from_template(template)

    # Reverted to your requested model
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0.3,
    )

    chain = prompt | llm | StrOutputParser()

    return chain.stream({
        "chat_history": chat_history,
        "user_question": user_query,
        "context": context
    })


# --- Sidebar for File Upload ---
with st.sidebar:
    st.header("Upload Document")
    uploaded_file = st.file_uploader(
        "Upload a PDF, TXT, CSV, or Image",
        type=["pdf", "txt", "csv", "png", "jpg", "jpeg"]
    )

    if uploaded_file is not None:
        if st.button("Process File"):
            with st.spinner("Processing..."):
                process_file(uploaded_file)

    # Show preview of dataframe in sidebar if loaded
    if st.session_state.active_dataframe is not None:
        st.divider()
        st.write("📊 **Data Loaded**")
        st.dataframe(st.session_state.active_dataframe.head(3), height=100, width="stretch")

# --- Tabs for Chat vs Charts ---
tab1, tab2 = st.tabs(["💬 Chat", "📈 Data Visualization"])

with tab1:
    # --- Chat Interface ---
    for message in st.session_state.chat_history:
        if isinstance(message, AIMessage):
            with st.chat_message("AI"):
                st.write(message.content)
        elif isinstance(message, HumanMessage):
            with st.chat_message("Human"):
                st.write(message.content)

    user_query = st.chat_input("Type your message here...")

    if user_query is not None and user_query != "":
        st.session_state.chat_history.append(HumanMessage(content=user_query))

        with st.chat_message("Human"):
            st.markdown(user_query)

        with st.chat_message("AI"):
            response_stream = get_response(
                user_query,
                st.session_state.chat_history,
                st.session_state.document_context
            )
            response = st.write_stream(response_stream)

        st.session_state.chat_history.append(AIMessage(content=response))

with tab2:
    # --- Charting Interface ---
    if st.session_state.active_dataframe is not None:
        df = st.session_state.active_dataframe
        st.subheader("Visualize your Data")

        # 1. Column Selection controls
        c1, c2, c3 = st.columns(3)
        with c1:
            chart_type = st.selectbox("Chart Type", ["Bar", "Line", "Scatter", "Area"])
        with c2:
            x_col = st.selectbox("X-Axis", df.columns)
        with c3:
            # Filter for numeric columns for Y-axis to prevent errors
            numeric_cols = df.select_dtypes(include=['number']).columns
            y_cols = st.multiselect("Y-Axis", numeric_cols, default=numeric_cols[0] if len(numeric_cols) > 0 else None)

        st.divider()

        # 2. Render Charts
        if x_col and y_cols:
            # Streamlit charts index by the X-axis
            chart_data = df.set_index(x_col)[y_cols]

            if chart_type == "Bar":
                st.bar_chart(chart_data)
            elif chart_type == "Line":
                st.line_chart(chart_data)
            elif chart_type == "Area":
                st.area_chart(chart_data)
            elif chart_type == "Scatter":
                st.scatter_chart(chart_data)
        else:
            st.info("Select columns to generate a chart.")

        with st.expander("See Raw Data"):
            st.dataframe(df)

    else:
        st.info("Upload a CSV file to see visualizations here.")
