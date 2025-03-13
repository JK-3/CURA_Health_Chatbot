from flask import Flask, render_template, jsonify, request, session
from src.helper import download_hugging_face_embeddings, clean_ocr_text
from langchain_pinecone import PineconeVectorStore
from dotenv import load_dotenv
from groq import Groq
from PIL import Image
import pytesseract
import os
import io
from werkzeug.utils import secure_filename
pytesseract.pytesseract.tesseract_cmd = r"C:\\Program Files\\Tesseract-OCR"


# Initialize Flask app
app = Flask(__name__)
app.secret_key = 'JAM32631'

# Load environment variables
load_dotenv()

# Set API keys
os.environ["PINECONE_API_KEY"] = "pcsk_UWfTo_8FyHHoCj6NMrkyYrctTkc8QG8KCxAM5ofAojoRUwadDPTkTgL8uenKcuAJncteU"
GROQ_API_KEY = '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'

# Load embeddings and Pinecone index
embeddings = download_hugging_face_embeddings()
index_name = "testbot"
docsearch = PineconeVectorStore.from_existing_index(index_name=index_name, embedding=embeddings)
retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k": 3})

# Initialize Groq client
client = Groq(api_key=GROQ_API_KEY)


# Home route
@app.route("/")
def index():
    if 'mode' not in session:
        session['mode'] = 'chat'
    if 'chat_history' not in session:
        session['chat_history'] = []  # Initialize chat history
    return render_template("chat.html", chat_history=session['chat_history'])


# Mode switch route
@app.route("/set_mode", methods=["POST"])
def set_mode():
    selected_mode = request.form.get("mode")
    session['mode'] = selected_mode
    return jsonify({"status": "success", "mode": selected_mode})


# Chat handling route
@app.route("/get", methods=["POST"])
def chat():
    user_query = request.form["msg"]
    mode = session.get('mode', 'chat')

    if 'chat_history' not in session:
        session['chat_history'] = []

    if mode == 'chat':
        retrieved_docs = retriever.invoke(user_query)
        context_text = "\n\n".join([doc.page_content for doc in retrieved_docs])

        system_prompt = """You are an expert in healthcare, health, and things related to healthcare.Use the following pieces of information to answer the user's question.
If you don't know the answer, just say you don't know. Don't try to make up an answer.

Context: {context}
Question: {question}

Only return the helpful answer below and nothing else.
Helpful answer:"""
        final_prompt = system_prompt.format(context=context_text, question=user_query)
    
    elif mode == 'nutrition':
        final_prompt = f"""
Act as a nutrition ingredient analysis expert. Based on the following ingredient list or food label content, provide a clear, structured, and point-wise output.

1. **Plain Summary of Ingredients** (bullet points if many).
2. **Ingredient Categorization** (e.g., emulsifier, preservative, additive, natural, synthetic).
3. **Health Impact Tags** (e.g., impact on acne, PCOS, diabetes, gut health, etc).
4. **Red Flags or Vague Terms** (e.g., 'natural flavoring', 'spices', etc).
5. **Warnings**: High sugar, sodium, bad fats, hormone disruptors, etc.
6. **Diet Compatibility**: Mark if it's suitable for keto, sugar-free, low-carb, gluten-free, etc.
7. **Final Verdict**: Is it a health-conscious choice? (Yes/No with a reason).

Input: {user_query}

Format your output using:
- Numbered lists
- Bullet points
- Bold section titles
- Avoid long paragraphs.
"""

    response = client.chat.completions.create(
        model="llama3-8b-8192",
        messages=[
            {"role": "system", "content": "You are a helpful health and nutrition assistant."},
            {"role": "user", "content": final_prompt}
        ]
    )

    result = response.choices[0].message.content

    session['chat_history'].append({"sender": "user", "message": user_query})
    session['chat_history'].append({"sender": "bot", "message": result})

    return jsonify({"response": result, "chat_history": session['chat_history']})


# Image Upload + OCR + Nutrition Analysis Route
@app.route("/upload_image", methods=["POST"])
def upload_image():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'Empty filename'}), 400

    filename = secure_filename(file.filename)
    filepath = os.path.join("uploads", filename)
    os.makedirs("uploads", exist_ok=True)
    file.save(filepath)

    try:
        image = Image.open(filepath)
        extracted_text = pytesseract.image_to_string(image)
        cleaned_text = clean_ocr_text(extracted_text)

        if not cleaned_text.strip():
            return jsonify({'error': "Couldn't extract text from the image."}), 400

        session['mode'] = 'nutrition'
        user_query = cleaned_text

        final_prompt = f"""
Act as a nutrition ingredient analysis expert. Based on the following ingredient list or food label content, provide a clear, structured, and point-wise output.

1. **Plain Summary of Ingredients** (bullet points if many).
2. **Ingredient Categorization** (e.g., emulsifier, preservative, additive, natural, synthetic).
3. **Health Impact Tags** (e.g., impact on acne, PCOS, diabetes, gut health, etc).
4. **Red Flags or Vague Terms** (e.g., 'natural flavoring', 'spices', etc).
5. **Warnings**: High sugar, sodium, bad fats, hormone disruptors, etc.
6. **Diet Compatibility**: Mark if it's suitable for keto, sugar-free, low-carb, gluten-free, etc.
7. **Final Verdict**: Is it a health-conscious choice? (Yes/No with a reason).

Input: {user_query}

Format your output using:
- Numbered lists
- Bullet points
- Bold section titles
- Avoid long paragraphs.
"""

        response = client.chat.completions.create(
            model="llama3-8b-8192",
            messages=[
                {"role": "system", "content": "You are a helpful health and nutrition assistant."},
                {"role": "user", "content": final_prompt}
            ]
        )

        result = response.choices[0].message.content
        return jsonify({'response': result})

    except Exception as e:
        return jsonify({'error': 'Failed to process image'}), 500


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8080, debug=True)