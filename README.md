I fine-tuned the Inception Jais 7B model on a custom legal dataset, which I created since there was no available Arabic question-and-answer dataset. To build the dataset, I extracted data from online PDFs and generated additional content using AI. I implemented Retrieval-Augmented Generation (RAG) with Langgraph ,LangChain and built a router and a retrieval grader. If the result from RAG had a cosine similarity below a set threshold, the system routed the query to a Google search node. The extracted document was then summarized, and both the summarization and the original question were passed to my fine-tuned model, which generated the final answer interface is in chat form .



![Screenshot (1401)](https://github.com/user-attachments/assets/7750c67a-471b-4c4a-bb8f-eae65b4440d9)



![Screenshot (1178)](https://github.com/user-attachments/assets/1e4540d5-6bd0-49a6-994c-7b661ff1b2a3)
