from flask import Flask, render_template, request
from normalizer import load_trained_transformer, load_tokenizer, normalize


app = Flask(__name__)
normalizer = load_trained_transformer("normalizer")
unreg_vocab_size = load_tokenizer("unregularized_bpe_tokenizer")
reg_vocab_size = load_tokenizer("regularized_bpe_tokenizer")


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    if request.method == "POST":
        unnormalized = request.form.get("unnormalized-phrase")

        if not unnormalized:
            return render_template("index.html", text=unnormalized, prediction=None)

        normalized = normalize(unnormalized, normalizer, unreg_vocab_size, reg_vocab_size)
        return render_template("index.html", text=unnormalized, prediction=normalized)


@app.route("/about")
def about():
    return render_template("about.html")


if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=True)
