import os
import streamlit as st
from dotenv import load_dotenv
from langchain_community.document_loaders import WikipediaLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

# 環境変数の読み込み
load_dotenv()

# Streamlitの設定
st.set_page_config(page_title="Wikipedia RAG Q&A", layout="wide")
st.title("Wikipedia RAG Q&A システム")

# サイドバーでWikipediaの検索キーワードを入力
with st.sidebar:
    st.header("検索設定")
    wiki_query = st.text_input("Wikipediaで検索するキーワード", value="中田英寿")
    if st.button("データを更新"):
        with st.spinner("Wikipediaからデータを取得中..."):
            # Wikipediaからデータを取得
            loader = WikipediaLoader(query=wiki_query, lang="ja")
            documents = loader.load()
            
            # テキストを分割
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            splits = text_splitter.split_documents(documents)
            
            # Chromaにデータを保存
            embeddings = OpenAIEmbeddings()
            vectorstore = Chroma.from_documents(
                documents=splits,
                embedding=embeddings,
                persist_directory=os.getenv("CHROMA_PERSIST_DIRECTORY")
            )
            vectorstore.persist()
            st.success("データの更新が完了しました！")

# メインエリアで質問を受け付ける
st.header("質問応答")
question = st.text_input("質問を入力してください")

if question:
    try:
        # Chromaからデータを読み込む
        embeddings = OpenAIEmbeddings()
        vectorstore = Chroma(
            persist_directory=os.getenv("CHROMA_PERSIST_DIRECTORY"),
            embedding_function=embeddings
        )
        
        # プロンプトテンプレートの作成
        template = """
        以下の質問に対して、与えられたコンテキストを使用して○か×で回答してください。
        回答は必ず「○」または「×」で始めてください。
        その後に、簡単な説明を追加してください。

        コンテキスト: {context}
        質問: {question}

        回答: """

        PROMPT = PromptTemplate(
            template=template,
            input_variables=["context", "question"]
        )

        # RAGチェーンの作成
        llm = ChatOpenAI(temperature=0)
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=vectorstore.as_retriever(),
            chain_type_kwargs={"prompt": PROMPT}
        )
        
        # 質問に回答
        with st.spinner("回答を生成中..."):
            response = qa_chain.invoke({"query": question})
            st.write(response["result"])
            
    except Exception as e:
        st.error(f"エラーが発生しました: {str(e)}") 