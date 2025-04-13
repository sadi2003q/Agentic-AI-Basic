from langchain_community.tools.file_management.write import WriteFileTool
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
import os
load_dotenv()


def Summarise_Info(text):
    model = ChatOpenAI(model="gpt-4o-mini")
    prompt = PromptTemplate(
        template="""
                Write a short Report of the following text: {text}
                -   report the main points and key details.
                -   Include any important dates, names, or locations mentioned.
                -   Use your own words and avoid repeating phrases.
                -   Keep the summary concise and easy to understand.
                -   Add bullet point into the report
                -   highlight heading names
                -   report should not be just a  paragraph
                -   contain latest source
                -   add links to the sources
                -   Add an overall summary of the report finally
                -   make sure you return me the markdown format of the summary, not raw text version.
                -   markdown should not contain any '```markdown' at the start  
            """,
        input_variables=["text"]
    )

    chain = prompt | model | StrOutputParser()

    result = chain.invoke({'text':text})

    return result


def Save_Markdown_file(text: str):
    try:
        write_file_tool = WriteFileTool(root_dir="../../../")
        markdown_content = f"# Summary\n\n{text}"
        file_path = "Files/markdown_file.md"
        write_file_tool.invoke({"file_path": file_path, "text": markdown_content})
        print("Successful")
        return file_path
    except Exception as e:
        print(f"Error writing file: {e}")

