import tiktoken

# 创建一个用户聊天记录的字典
user_chat_records = {}
# 设置最大token限制
max_token_limit = 1000  

# 添加聊天记录的函数
def add_chat_record(user_id, message):
    if user_id not in user_chat_records:
        # 使用FIFO策略，保留最近的10条消息
        user_chat_records[user_id] = [] 
    user_chat_records[user_id].append(message)

# 判断token是否超长的函数
def is_token_limit_exceeded(user_id):
    if user_id not in user_chat_records:
        return False
    encoding = tiktoken.encoding_for_model('gpt2')
    token_integers = encoding.encode(str(user_chat_records[user_id]))
    length = len(token_integers);
    print(f"{length} tokens")
    if length > max_token_limit:
        return True
    else:
        return False

# 获取用户的历史数据    
def get_chat_record(userId):
    return user_chat_records[userId]

# 处理聊天消息
def process_chat_message(user_id, message):
    # 首次发言校验
    if user_id not in user_chat_records:
        add_chat_record(user_id, message)
        return get_chat_record(user_id)
    # 长度校验
    if is_token_limit_exceeded(user_id):
        # 弹出首条消息
        del user_chat_records[user_id][0]  
    add_chat_record(user_id, message)
    return get_chat_record(user_id)
