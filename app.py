from flask import Flask, render_template, request, send_from_directory, redirect, url_for
import os
import sqlite3
from datetime import datetime

import fitz
import torch
import easyocr
from transformers import AutoTokenizer, AutoModel


app = Flask(__name__)

UPLOAD_FOLDER = "uploads"
DATABASE = "evaluations.db"

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

os.makedirs(UPLOAD_FOLDER, exist_ok=True)


# ============================================================
# DATABASE
# ============================================================

def get_db():
    conn = sqlite3.connect(DATABASE)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():

    conn = get_db()

    conn.execute("""
        CREATE TABLE IF NOT EXISTS evaluations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            filename TEXT NOT NULL,
            extracted_text TEXT,
            similarity REAL,
            concept_coverage REAL,
            suggested_marks REAL,
            max_marks REAL,
            found_concepts TEXT,
            final_marks REAL,
            feedback TEXT,
            status TEXT DEFAULT 'Pending',
            created_at TEXT
        )
    """)

    conn.commit()
    conn.close()


init_db()


# ============================================================
# EASYOCR
# ============================================================

print("Loading EasyOCR...")

reader = easyocr.Reader(
    ["en"],
    gpu=False
)

print("EasyOCR loaded successfully!")


# ============================================================
# DISTILBERT
# ============================================================

MODEL_NAME = "distilbert-base-uncased"

print("Loading DistilBERT...")

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME
)

model = AutoModel.from_pretrained(
    MODEL_NAME
)

model.eval()

print("DistilBERT loaded successfully!")


# ============================================================
# EMBEDDING
# ============================================================

def get_embedding(text):

    if not text or not text.strip():
        return torch.zeros(1, 768)

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=512
    )

    with torch.no_grad():
        outputs = model(**inputs)

    return outputs.last_hidden_state.mean(dim=1)


# ============================================================
# SIMILARITY
# ============================================================

def calculate_similarity(student_text, reference_text):

    student_embedding = get_embedding(student_text)
    reference_embedding = get_embedding(reference_text)

    similarity = torch.nn.functional.cosine_similarity(
        student_embedding,
        reference_embedding
    )

    return similarity.item()


# ============================================================
# CONCEPT COVERAGE
# ============================================================

def calculate_concept_coverage(student_text, concepts):

    if not student_text:
        return 0, []

    student_text = student_text.lower()

    found_concepts = []

    words = student_text.split()

    for concept in concepts:

        concept_lower = concept.lower()

        if concept_lower in student_text:
            found_concepts.append(concept)
            continue

        concept_words = concept_lower.split()

        for concept_word in concept_words:

            for word in words:

                clean_word = word.strip(
                    ".,!?;:()[]{}-_\"'"
                )

                if (
                    concept_word in clean_word
                    or clean_word in concept_word
                ):
                    found_concepts.append(concept)
                    break

            if concept in found_concepts:
                break

    found_concepts = list(
        dict.fromkeys(found_concepts)
    )

    if not concepts:
        return 0, []

    coverage = (
        len(found_concepts) / len(concepts)
    ) * 100

    return round(coverage, 2), found_concepts


# ============================================================
# OCR
# ============================================================

def recognize_handwriting(image_path):

    try:

        result = reader.readtext(
            image_path,
            detail=0
        )

        return " ".join(result).strip()

    except Exception as e:

        print("OCR Error:", e)

        return ""


# ============================================================
# HOME
# ============================================================

@app.route("/")
def home():

    return render_template("index.html")


# ============================================================
# EVALUATE PAGE
# ============================================================

@app.route("/evaluate")
def evaluate():

    return render_template("evaluate.html")


# ============================================================
# UPLOAD + AI EVALUATION
# ============================================================

@app.route("/upload", methods=["POST"])
def upload():

    if "pdf" not in request.files:
        return "No PDF selected."

    file = request.files["pdf"]

    if file.filename == "":
        return "No PDF selected."

    if not file.filename.lower().endswith(".pdf"):
        return "Please upload a PDF file."

    file_path = os.path.join(
        app.config["UPLOAD_FOLDER"],
        file.filename
    )

    file.save(file_path)

    # --------------------------------------------------------
    # PDF TO IMAGE + OCR
    # --------------------------------------------------------

    pdf = fitz.open(file_path)

    extracted_text = ""

    for page_number, page in enumerate(pdf):

        print(
            f"Processing page {page_number + 1}..."
        )

        matrix = fitz.Matrix(
            2.5,
            2.5
        )

        pix = page.get_pixmap(
            matrix=matrix
        )

        image_path = os.path.join(
            app.config["UPLOAD_FOLDER"],
            f"{file.filename}_{page_number + 1}.png"
        )

        pix.save(image_path)

        page_text = recognize_handwriting(
            image_path
        )

        extracted_text += (
            f"\n\n--- Page "
            f"{page_number + 1} ---\n"
        )

        extracted_text += page_text

    pdf.close()

    if not extracted_text.strip():

        extracted_text = (
            "No text could be detected."
        )


    # --------------------------------------------------------
    # REFERENCE ANSWER
    # --------------------------------------------------------

    reference_answer = """
    A computer network is a group of interconnected
    computers and devices that communicate with each
    other and share resources and information.
    """


    concepts = [
        "computer network",
        "interconnected",
        "computers",
        "devices",
        "communicate",
        "resources",
        "information"
    ]


    # --------------------------------------------------------
    # AI EVALUATION
    # --------------------------------------------------------

    similarity = calculate_similarity(
        extracted_text,
        reference_answer
    )

    similarity_percentage = round(
        max(0, similarity) * 100,
        2
    )

    concept_coverage, found_concepts = (
        calculate_concept_coverage(
            extracted_text,
            concepts
        )
    )


    max_marks = 10

    final_score = (
        similarity_percentage * 0.60
        +
        concept_coverage * 0.40
    )

    suggested_marks = round(
        (final_score / 100) * max_marks,
        1
    )


    # --------------------------------------------------------
    # SAVE TO DATABASE
    # --------------------------------------------------------

    conn = get_db()

    cursor = conn.execute(
        """
        INSERT INTO evaluations
        (
            filename,
            extracted_text,
            similarity,
            concept_coverage,
            suggested_marks,
            max_marks,
            found_concepts,
            final_marks,
            feedback,
            status,
            created_at
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            file.filename,
            extracted_text,
            similarity_percentage,
            concept_coverage,
            suggested_marks,
            max_marks,
            ", ".join(found_concepts),
            None,
            "",
            "Pending",
            datetime.now().strftime(
                "%Y-%m-%d %H:%M:%S"
            )
        )
    )

    evaluation_id = cursor.lastrowid

    conn.commit()
    conn.close()


    # --------------------------------------------------------
    # RESULT PAGE
    # --------------------------------------------------------

    return render_template(
        "result.html",
        filename=file.filename,
        text=extracted_text,
        similarity=similarity_percentage,
        concept_coverage=concept_coverage,
        found_concepts=found_concepts,
        marks=suggested_marks,
        max_marks=max_marks,
        evaluation_id=evaluation_id
    )


# ============================================================
# SERVE UPLOADED PDF
# ============================================================

@app.route("/uploads/<filename>")
def uploaded_file(filename):

    return send_from_directory(
        app.config["UPLOAD_FOLDER"],
        filename
    )


# ============================================================
# HISTORY
# ============================================================

@app.route("/history")
def history():

    conn = get_db()

    evaluations = conn.execute(
        """
        SELECT *
        FROM evaluations
        ORDER BY id DESC
        """
    ).fetchall()

    conn.close()

    return render_template(
        "history.html",
        evaluations=evaluations
    )


# ============================================================
# FACULTY REVIEW
# ============================================================

@app.route("/review/<int:evaluation_id>")
def review(evaluation_id):

    conn = get_db()

    evaluation = conn.execute(
        """
        SELECT *
        FROM evaluations
        WHERE id = ?
        """,
        (evaluation_id,)
    ).fetchone()

    conn.close()

    if evaluation is None:
        return "Evaluation not found."

    found_concepts = []

    if evaluation["found_concepts"]:

        found_concepts = [
            item.strip()
            for item in evaluation["found_concepts"].split(",")
        ]

    return render_template(
        "faculty_review.html",
        evaluation=evaluation,
        found_concepts=found_concepts
    )


# ============================================================
# SAVE FACULTY REVIEW
# ============================================================

@app.route(
    "/save-review/<int:evaluation_id>",
    methods=["POST"]
)
def save_review(evaluation_id):

    final_marks = request.form.get(
        "final_marks"
    )

    feedback = request.form.get(
        "feedback",
        ""
    )

    conn = get_db()

    conn.execute(
        """
        UPDATE evaluations
        SET
            final_marks = ?,
            feedback = ?,
            status = ?
        WHERE id = ?
        """,
        (
            final_marks,
            feedback,
            "Reviewed",
            evaluation_id
        )
    )

    conn.commit()
    conn.close()

    return redirect(
        url_for("history")
    )


# ============================================================
# FACULTY REVIEW SHORTCUT
# ============================================================

@app.route("/faculty-review")
def faculty_review():

    return redirect(
        url_for("history")
    )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    app.run(
        debug=True
    )