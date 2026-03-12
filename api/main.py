from fastapi import FastAPI

app = FastAPI()

@app.get("/")
def read_root():
    return {"message": "Dio canaglia"}

@app.get("/items")
def get_items():
    return ["prova"]

@app.get("/items1")
def get_items(limit = 2):
    items = ["apple", "banana", "orange"]
    return items[:int(limit)]

@app.get("/items/{item_id}")
def get_item(item_id: int):
    items = ["apple", "banana", "orange"]
    if 0 <= item_id < len(items):
        return {"item": items[item_id]}
    return {"error": "Item not found"}


from pydantic import BaseModel

class Item(BaseModel):
    name: str
    price: float

@app.post("/filippo")
def create_item(item: Item):
    return {"message": f"Item {item.name} with price {item.price} added!"}