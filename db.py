from pymongo import MongoClient
import os
from dotenv import load_dotenv

load_dotenv()

client = MongoClient(os.getenv("MONGO_URI"))
db = client["studymate"]

users_col = db["users"]
chats_col = db["chats"]
semantic_cache_col = db["semantic_cache"]
file_summary_col = db["file_summaries"]
