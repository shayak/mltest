from llama_index import TreeIndex, SimpleDirectoryReader

resume = SimpleDirectoryReader("documents").load_data()
new_index = TreeIndex.from_documents(resume)
