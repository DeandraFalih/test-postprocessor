import uvicorn
from fastapi import FastAPI
from starlette.responses import FileResponse

app = FastAPI(title='Gauge Example')

gauge_value = 0


@app.get("/")
def read_root():
    return FileResponse('index.html')


@app.get('/value')
def get_gauge_value():
    print(gauge_value)
    return gauge_value


@app.post('/value')
def set_gauge_value(value: float):
    global gauge_value
    gauge_value = value


def start_api():
    uvicorn.run(app, port=8888, host="0.0.0.0")


if __name__ == '__main__':
    start_api()
