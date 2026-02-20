# Project Setup Guide

## 1. Create Project Folder

-   Create a project folder named **Langsmith**
-   Locate or clone the project code from GitHub into this folder

## 2. Create and Activate Virtual Environment

### a. Create virtual environment

``` bash
python -m venv labenv
```

### b. Activate virtual environment (Windows)

``` bash
.\labenv\Scripts\Activate
```

## 3. Install Requirements

Run the following command:

``` bash
pip install -U langgraph langchain-openai langsmith python-dotenv
```

## 4. Create OpenAI API Key

-   Visit: https://platform.openai.com/settings/organization/api-keys
-   Generate a new API key
-   Save it securely

## 5. Create LangSmith API Key

-   Visit: https://smith.langchain.com/
-   Sign in and generate your LangSmith API key
-   Save it securely

## 6. Learning Resources

-   https://docs.langchain.com/langsmith/evaluate-complex-agent
