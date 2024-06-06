import streamlit as st
import pandas
import re
import os
import html
import warnings
from src.apps.chat_layout import format_message,message_function
from PIL import Image
from .process import Qna
from datetime import datetime
warnings.filterwarnings("ignore")
chat_history = []

 

chat_placeholder = st.container()
# promt_placeholder = st.form('chat-form')
obj=Qna()

 

 

def app(data):
    try:
        
        

        # Specify directory to save uploaded files
        upload_dir = "uploads"
        os.makedirs(upload_dir, exist_ok=True)  # Create directory if it doesn't exist
        # Upload image file
        uploaded_file = st.file_uploader("Upload a image file in png format", type="png")

        if uploaded_file is not None:

            st.success('Image file successfully uploaded!')
            # Save the uploaded file
            
            file_path = obj.save_uploaded_file(uploaded_file, upload_dir)
            st.write(f"Image file saved to: {file_path}")

            
            
            # Display chat interface
            
            st.title("QAS - Question-Answering Service")
            st.caption(
                "Welcome to the Question-Answering Service, where you can ask any question on image you uploaded")
            st.caption(
                "This POC has been developed by the Maheep Singh"
            )
            st.subheader("Chat with the Image")

            model = st.sidebar.radio(
            "",
            options=["Microsoft-Phi-Vision-128k","GPT-4 Vision"],
            index=0,
            horizontal=True,
                )


            st.sidebar.markdown('For more enquiry <a href="mailto:maheeps99@gmail.com"> contact us</a>', unsafe_allow_html=True)

    

            INITIAL_MESSAGE = [
                {"role": "user", "content": "Hi!"},
                {
                    "role": "assistant",
                    "content": "Ask any question about the uploaded Image. 🔍",
                },
            ]

    

            # Initialize the chat messages history
            if "messages" not in st.session_state.keys():
                st.session_state["messages"] = INITIAL_MESSAGE

    

            if "history" not in st.session_state:
                st.session_state["history"] = []

    

    

            for message in st.session_state.messages:
                if message["content"] == "Ask any question about the Image. 🔍":
                    message_function(
                        message["content"],
                        True if message["role"] == "user" else False,
                        True if message["role"] == "data" else False,
                    )
                else:
                    message_function(
                        message["content"],
                        True if message["role"] == "user" else False,
                        True if message["role"] == "data" else False,
                    )

    

    

            # User-provided prompt
            if prompt := st.chat_input():
                st.session_state.messages.append({"role": "user", "content": prompt})
                # st.write(prompt)
                # with st.chat_message("user"):
                #     st.write(prompt)
            if st.session_state.messages[-1]["role"] != "assistant":
                message = {"role": "user", "content": prompt}
                message_function(
                                message["content"],
                                True if message["role"] == "user" else False,
                                True if message["role"] == "data" else False,
                            )
            with st.spinner("Thinking..."):
                if st.session_state.messages[-1]["role"] != "assistant":

    

                    response = obj.main(prompt,model,file_path) 
                    if response==None:
                        st.error("Selected chat feature is under Maintenance!!!")
                    else:
                        # print(response)
                        # st.write(response) 
                        # message = {"role": "user", "content": prompt}
                        # message_func(
                        #     message["content"],
                        #     True if message["role"] == "user" else False,
                        #     True if message["role"] == "data" else False,
                        # )

    

                        message = {"role": "assistant", "content": response}
                        message_function(
                            message["content"],
                            True if message["role"] == "user" else False,
                            True if message["role"] == "data" else False,
                        )
                        st.session_state.messages.append(message)

 

    except:
        st.error("This Site under Maintenance. Thank you for your Patience ")