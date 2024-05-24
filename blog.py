import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_community.document_loaders import TextLoader
from fastapi import FastAPI,File, UploadFile

load_dotenv()

app=FastAPI()

template = """
You are a highly skilled technical writer with expertise in explaining complex code in an easy-to-understand manner. Your task is to write a detailed and informative blog post about the provided code. Follow the instructions below to structure the blog post effectively:

1. **Introduction**:
    - Begin with an engaging introduction that provides context about the code. Explain what the code does and why it is important or useful.
    - Mention the programming language used and any relevant technologies or frameworks.

2. **Code Breakdown**:
    - Break the code into logical sections. For each section, include the following:
        - A clear heading that describes the section's purpose.
        - The code snippet, formatted for readability.
        - A detailed explanation of what the code does. Describe how it works, why certain approaches were taken, and any important details.
        - If applicable, mention best practices, potential pitfalls, or common issues related to the code.

3. **Use Cases and Examples**:
    - Provide examples of how the code can be used in real-world scenarios. Explain any customization options or variations that could be applied.

4. **Conclusion**:
    - Summarize the key points covered in the blog post.
    - Highlight the benefits of using this code and any final thoughts or recommendations.

5. **SEO Optimization**:
    - Naturally incorporate relevant SEO keywords throughout the blog post. Ensure the content remains clear and engaging while improving its search engine visibility.

6. **Proofreading**:
    - Ensure the blog post is free from grammatical errors and aligns with a professional and approachable tone.

Here is the code you need to explain:

```
{code}"""


readme_template="""
You are a highly skilled technical writer with expertise in creating clear, concise, and informative README files for GitHub repositories. Your task is to write a detailed README.md file for the provided code. Follow the instructions below to structure the README effectively:

1. **Project Title**:
    - Provide a clear and concise title for the project.

2. **Description**:
    - Write a brief description of the project. Explain its purpose, functionality, and the problem it solves.

3. **Table of Contents**:
    - Include a table of contents to help users navigate the README easily.

4. **Installation**:
    - Provide step-by-step instructions on how to install and set up the project. Include any prerequisites or dependencies.

5. **Usage**:
    - Explain how to use the project. Provide examples and code snippets to illustrate common use cases.

6. **Code Overview**:
    - Break down the provided code into logical sections. For each section, include the following:
        - A clear heading that describes the section's purpose.
        - The code snippet, formatted for readability.
        - A detailed explanation of what the code does. Describe how it works, why certain approaches were taken, and any important details.

7. **Features**:
    - List the main features of the project.

8. **Contributing**:
    - Describe how others can contribute to the project. Include guidelines for submitting issues and pull requests.

9. **License**:
    - Specify the license under which the project is distributed.

10. **Contact Information**:
    - Provide contact information for users who have questions or need support.

Here is the code you need to include in the README:

```python
{code}"""


llama = ChatGroq(
    model="llama3-8b-8192",
    api_key=os.getenv("GROQ_API_KEY")
)

prompt_template=PromptTemplate.from_template(readme_template)

prompt_template_blog=PromptTemplate.from_template(template)

loader = TextLoader('./code_to_blog.py')

code = loader.load()

prompt = prompt_template.invoke({"code": code[0].page_content})

@app.get("/generate-readme")
async def generate_readme():
    result = llama.invoke(prompt)
    return {"result": result.content}

@app.post("/generate-readme-with-file")
async def create_upload_file(file:UploadFile):
    content = await file.read()
    content_string = content.decode("utf-8")
    prompt = prompt_template.invoke({"code": content_string})
    result = llama.invoke(prompt)
    return result.content

@app.post("/generate-readme-with-code")
async def generate_readme_with_code(code: str):
    prompt = prompt_template.invoke({"code": code})
    result = llama.invoke(prompt)
    return result.content

@app.post("/generate-blog-with-file")
async def create_upload_file(file:UploadFile):
    content = await file.read()
    content_string = content.decode("utf-8")
    prompt = prompt_template_blog.invoke({"code": content_string})
    result = llama.invoke(prompt)
    return result.content

@app.post("/generate-readme-with-code")
async def generate_readme_with_code(code: str):
    prompt = prompt_template_blog.invoke({"code": code})
    result = llama.invoke(prompt)
    return result.content