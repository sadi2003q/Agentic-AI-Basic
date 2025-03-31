import smtplib
from langchain_openai import ChatOpenAI
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

from langchain.prompts import PromptTemplate
from pydantic import BaseModel, Field
from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_loaders import UnstructuredMarkdownLoader
from email.mime.application import MIMEApplication
import os
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders
from typing import Optional
from dotenv import load_dotenv

load_dotenv()


class Email_Template(BaseModel):
    subject: str = Field("Subject of the Email", title="Subject")
    body: str = Field("Body of the Email", title="Body")


class Email_Agent:
    def __init__(self, email_from: str, email_to: str, file: str, text: str):
        self.email_Template = None
        self.email_from = email_from
        self.email_to = email_to
        self.file = file
        self.text = text

    def Email_draft(self):

        model = ChatOpenAI(model='gpt-4o-mini')

        prompt = PromptTemplate(
            template="""
                -   make an Appropriate subject for the email based on the following text: {text}
                -   make an Appropriate body for the email based on the following text: {text}
                -   make sure the subject is not more then 20 words
                -   make sure the body is not more then 100 words
                -   Finally add 2 newline and inform user about the file attached in the email because summary was make from that file
                -
                """,
            input_variables=['text']
        )

        structure_output = model.with_structured_output(Email_Template)

        chain = prompt | structure_output

        self.email_Template = chain.invoke({"text": self.text})

    def check_file_validity(self, file_path):
        """Check if the given path exists and is a valid file."""
        if not os.path.exists(file_path):
            print(f"Error: The file path '{file_path}' does not exist.")
            return False
        if not os.path.isfile(file_path):
            print(f"Error: '{file_path}' is not a valid file.")
            return False
        return True

    def Proceed_Email(self):
        self.Email_draft()
        msg = MIMEMultipart()
        msg['From'] = self.email_from
        msg['To'] = self.email_to
        msg['Subject'] = self.email_Template.subject

        # Add body text
        msg.attach(MIMEText(self.email_Template.body, 'plain'))

        file_path = self.file

        if not self.check_file_validity(file_path):
            print(f"Error: The file path '{file_path}'")
            return

        try:
            # Verify file exists
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"Attachment file not found: {file_path}")

            # Add attachment
            with open(file_path, "rb") as attachment:
                part = MIMEBase('application', 'octet-stream')
                part.set_payload(attachment.read())

            # Encode file in ASCII characters to send by email
            encoders.encode_base64(part)

            # Add header
            part.add_header(
                'Content-Disposition',
                f'attachment; filename= {os.path.basename(file_path)}',
            )

            msg.attach(part)
            print("Attachment successfully added to email")

        except Exception as e:
            print(f"Failed to attach file: {e}")

        try:
            server = smtplib.SMTP('smtp.gmail.com', 587)
            server.starttls()
            server.login(self.email_from, os.getenv('EMAIL_AGENT'))
            server.sendmail(msg["From"], msg["To"], msg.as_string())
            print("Email sent successfully")
        except Exception as e:
            print(f"Error sending email: {e}")
        return
