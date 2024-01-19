import os
from flask import Flask, render_template, request
import pandas as pd
from pandasai import SmartDataframe
from pandasai.llm import OpenAI


llm = OpenAI(api_token=os.environ.get("OPENAI_API_KEY", 8080))


app = Flask(__name__)


@app.route("/hello")
def hello():
    return "hello"


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/get_df")
def get_df():
    df = pd.read_csv("data_sales.csv")
    df = df.to_html(classes=["table table-df"])
    return render_template("get_df.html", table=df)


@app.route("/simple_htmx")
def simple_htmx():
    return render_template("simple_htmx.html")


@app.route("/get_htmx")
def get_htmx():
    return render_template("get_htmx.html")


@app.route("/pandas_ai", methods=["GET", "POST"])
def pandas_ai():
    if request.method == "POST":
        chat_input = request.form["chat"]
        df = pd.read_csv("data_sales.csv")
        df = SmartDataframe(df, config={"llm": llm})
        result = df.chat(chat_input)
        df_html = result.to_html(
            classes=["table table-stripped table-htmx table-responsive"], index=False)
        return render_template("htmx.html", df_html=df_html)
    else:
        return render_template("pandas_ai.html")


if __name__ == "__main__":
    app.run(debug=True,
            host="0.0.0.0",
            port=int(os.environ.get("PORT", 8080)))
