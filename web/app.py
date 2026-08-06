from flask import Flask, render_template, send_from_directory, request, redirect, url_for
import json
import os
from ml.predict import analyze_image, ranking_results, update_rankings
from analyzer.report_generator import generate_verdict
from werkzeug.utils import secure_filename
import zipfile
import tempfile
from flask import send_file


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ANNOTATED_DIR = os.path.join(BASE_DIR, "annotated")
IMAGES_DIR = os.path.join(BASE_DIR, "uploads")
JSON_DIR = os.path.join(BASE_DIR, "JSON")
UPLOAD_FOLDER = "uploads"
EYE_PENALTY = 15.0


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
    update_rankings()
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

    update_rankings()

    return redirect("/")

@app.route("/reanalyze-all", methods=["POST"])
def reanalyze_all():
    ranking_results.clear()
    for file in os.listdir(JSON_DIR):

        if not file.endswith(".json"):
            continue

        with open(os.path.join(JSON_DIR, file), "r") as f:
            report = json.load(f)

        filename = report["image_info"]["filename"]
        image_path = os.path.join(UPLOAD_FOLDER, filename)

        if os.path.exists(image_path):
            analyze_image(image_path)
    update_rankings()

    return redirect(url_for("dashboard"))

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
    print(report["quality"])
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
    print(len(ranking_results))
    print(ranking_results[-1])
    update_rankings()
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


@app.route("/clear-image/<path:filename>", methods=["POST"])
def clear_image(filename):
    stem = os.path.splitext(filename)[0]

    folders_exact = [IMAGES_DIR, ANNOTATED_DIR]
    for folder in folders_exact:
        path = os.path.join(folder, filename)
        if os.path.isfile(path):
            os.remove(path)
            print(f"Deleted: {path}")

    json_path = os.path.join(JSON_DIR, stem + ".json")
    if os.path.isfile(json_path):
        os.remove(json_path)
        print(f"Deleted: {json_path}")
        
    annotated_path = os.path.join(ANNOTATED_DIR, "visual_" + filename)
    if os.path.isfile(annotated_path):
        os.remove(annotated_path)

    return redirect(url_for("dashboard"))



@app.route("/update-eye-decision", methods=["POST"])
def update_eye_decision():

    filename = request.form["filename"]
    stem = os.path.splitext(filename)[0]

    path = os.path.join(
        JSON_DIR,
        stem + ".json"
    )

    # Load report
    with open(path, "r", encoding="utf-8") as f:
        report = json.load(f)

    # Store every user's decision
    for key, value in request.form.items():

        if not key.startswith("face_"):
            continue

        face_index = int(key.replace("face_", ""))

        report["face_analysis"]["subjects"][face_index]["decision"] = value

    # ----------------------------
    # Reset report sections
    # ----------------------------

    report["report"]["general"] = [
        item for item in report["report"]["general"]
        if item != "Eye closure marked as intentional."
    ]

    report["report"]["problems"] = [
        p for p in report["report"]["problems"]
        if "closed" not in p.lower()
    ]

    report["report"]["suggestions"] = [
        s for s in report["report"]["suggestions"]
        if "eyes open" not in s.lower()
    ]

    report["report"]["summary"] = [
        s for s in report["report"]["summary"]
        if "closed" not in s.lower()
    ]

    intentional_count = 0
    not_intentional_count = 0
    penalty = 0

    # Count decisions
    for subject in report["face_analysis"]["subjects"]:
        decision = subject.get("decision")


        if subject["status"] == "Eyes open" or decision == "open":
            continue


        if decision == "intentional":
            intentional_count += 1

        elif decision == "not_intentional":
            not_intentional_count += 1
            penalty += EYE_PENALTY

        elif decision == "open/intentional" or decision == "closed/intentional":
            intentional_count += 1

        elif decision == "open/not_intentional" or decision == "closed/not_intentional":
            not_intentional_count += 1
            penalty += EYE_PENALTY

        elif decision == "Closed":
            penalty += EYE_PENALTY



    # General section
    if intentional_count > 0:
        report["report"]["general"].append(
            f"{intentional_count} eye closure(s) marked as intentional."
        )

    # Problems
    if report["face_analysis"]["one_eye_closed"] > 0 and not_intentional_count > 0:
        report["report"]["problems"].append(
            f"{report['face_analysis']['one_eye_closed']} subject(s) have one eye closed."
        )

    if report["face_analysis"]["eyes_closed"] > 0 and not_intentional_count > 0:
        report["report"]["problems"].append(
            f"{report['face_analysis']['eyes_closed']} subject(s) have both eyes closed."
        )

    # Suggestions
    if not_intentional_count > 0:

        report["report"]["suggestions"].append(
            "Capture another frame with all subjects' eyes open."
        )

        report["report"]["summary"].append(
            f"{not_intentional_count} eye closure(s) may affect the image quality."
        )
    report["quality"]["score_breakdown"]["eye_penalty"] = (
    -15 if report["quality"]["decision"]["eye_penalty_applied"] else 0
    )
    report["quality"]["decision"]["eye_penalty_applied"] = (
    penalty > 0
    )
    report["quality"]["score_breakdown"]["eye_penalty"] = -penalty
    # Update score
    base = report["quality"]["base_score"]
    score = max(base - penalty, 0)
    report["quality"]["score_breakdown"]["overall"] = score

    report["quality"]["overall_score"] = score
    report["quality"]["verdict"] = generate_verdict(
        score,
    )

    # Save
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=4)

    return redirect(url_for("report", filename=stem))

@app.route("/export-results", methods=["POST"])
def export_results():
    temp = tempfile.NamedTemporaryFile(
        delete=False,
        suffix=".zip"
    )

    with zipfile.ZipFile(temp.name, "w", zipfile.ZIP_DEFLATED) as zipf:

        BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

        folders = [
            os.path.join(BASE_DIR, "annotated"),
            os.path.join(BASE_DIR, "sorted_images")
        ]

        for folder in folders:

            print("Scanning:", folder)

            if not os.path.exists(folder):
                continue

            for root, dirs, files in os.walk(folder):

                print(root, files)

                for file in files:

                    filepath = os.path.join(root, file)
                    print("Adding:", filepath)

                    arcname = os.path.relpath(filepath)
                    zipf.write(filepath, arcname)
    return send_file(
    temp.name,
    as_attachment=True,
    download_name="Image-Quality-Results.zip"
)


if __name__ == "__main__":
    app.run(debug=True)
