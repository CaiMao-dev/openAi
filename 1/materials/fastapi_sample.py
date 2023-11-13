# 导入 FastAPI 库
from fastapi import FastAPI
import uvicorn
from pydantic import BaseModel, EmailStr, conint


# 创建一个 FastAPI 应用
app = FastAPI()

# 创建一个 Pydantic 模型，用于校验请求参数
class Item(BaseModel):
    name: str
    description: str = None
    price: float
    tax: float = None

class User(BaseModel):
    username: str
    email: EmailStr
    age: conint(ge=0, le=120)  # 年龄必须在0到150之间
    bio: str = None  # 可选的字段，默认值为None

# 创建一个路由
@app.get("/")
async def read_root(dict):
    return {"message": "Hello, World"}

@app.post("/items")
async def create_item(item: Item):
    item_dict = item.dict()
    if item.tax is not None:
        total_price = item.price + item.tax
        item_dict.update({"total_price": total_price})
    return item_dict

@app.post("/users")
async def create_users(user: User):
    user_dict = user.dict()
    return user_dict

# 启动 FastAPI 应用
uvicorn.run(app, host="0.0.0.0", port=18001)