from flask import Flask, render_template, send_from_directory, request, redirect, url_for
import json
import os
from ml.predict import analyze_image
from werkzeug.utils import secure_filename

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ANNOTATED_DIR = os.path.join(BASE_DIR, "annotated")
IMAGES_DIR = os.path.join(BASE_DIR, "uploads")
JSON_DIR = os.path.join(BASE_DIR, "JSON")
UPLOAD_FOLDER = "uploads"


app = Flask(
    __name__,
    static_folder="static"
)

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

REPORT_PATH = "JSON/group.json"


@app.route("/")
def dashboard():
    reports = []

    for file in os.listdir(JSON_DIR):
        if file.endswith(".json"):
            with open(os.path.join(JSON_DIR, file), "r") as f:
                report = json.load(f)
            reports.append(report)

    reports.sort(
        key=lambda x: x["quality"]["overall_score"],
        reverse=True
    )
    excellent = [
        r for r in reports if r["quality"]["verdict"] == "Excellent"
    ]
    good = [
        r for r in reports if r["quality"]["verdict"] == "Good"
    ]
    fair = [
        r for r in reports if r["quality"]["verdict"] == "Fair"
    ]
    poor = [
        r for r in reports if r["quality"]["verdict"] == "Poor"
    ]
    return render_template(
        "dashboard.html",
        excellent=excellent,
        good=good,
        fair=fair,
        poor=poor,
    )


@app.route("/upload")
def upload():
    return render_template("upload.html")

@app.route("/uploads/<path:filename>")
def images(filename):
    return send_from_directory(
        IMAGES_DIR,
        filename
    )

@app.route("/analyze-single", methods=["POST"])
def analyze():
    file = request.files["image"]
    filename = secure_filename(file.filename)
    save_path = os.path.join(
        app.config["UPLOAD_FOLDER"],
        filename
    )
    file.save(save_path)
    report = analyze_image(save_path)
    
    filename = os.path.splitext(
        report["image_info"]["filename"]
    )[0]
    return redirect(f"/report/{filename}")


@app.route("/analyze-batch", methods=["POST"])
def analyze_batch():
    files = request.files.getlist("images")
    for file in files:
        filename = secure_filename(file.filename)
        save_path = os.path.join(
            app.config["UPLOAD_FOLDER"],
            filename
        )
        file.save(save_path)    

        report = analyze_image(save_path)

    return redirect("/")



@app.route("/annotated/<path:filename>")
def annotated(filename):
    return send_from_directory(
        ANNOTATED_DIR,
        filename
    )

@app.route("/report/<filename>")
def report(filename):

    report_path = os.path.join(
        JSON_DIR,
        filename + ".json"
    )

    with open(report_path, "r") as f:

        report = json.load(f)

    return render_template(
        "index.html",
        report=report
    )
@app.route("/reanalyze/<filename>", methods=["POST"])
def reanalyze(filename):
    image_path = os.path.join(
        IMAGES_DIR,
        filename
    )
    analyze_image(image_path)
    stem=os.path.splitext(filename)[0]
    return redirect(url_for("report", filename=stem))

@app.route("/clear-dashboard", methods=["POST"])
def clear_dashboard():

    folders = [JSON_DIR, IMAGES_DIR, ANNOTATED_DIR]

    for folder in folders:
        for file in os.listdir(folder):
            path = os.path.join(folder, file)

            if os.path.isfile(path):
                os.remove(path)

    return redirect(url_for("dashboard"))

if __name__ == "__main__":
    app.run(debug=True)