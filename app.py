
from flask import Flask, render_template, jsonify, request, session
from src.helper import download_hugging_face_embeddings, clean_ocr_text
from langchain_pinecone import PineconeVectorStore
from dotenv import load_dotenv
from groq import Groq
from PIL import Image
import pytesseract
import os
from werkzeug.utils import secure_filename

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

app = Flask(__name__)
app.secret_key = 'your key'

load_dotenv()
os.environ["PINECONE_API_KEY"] = "your pinecone api key "
GROQ_API_KEY = " your groq api key"

embeddings = download_hugging_face_embeddings()
index_name = "testbot"
docsearch = PineconeVectorStore.from_existing_index(index_name=index_name, embedding=embeddings)
retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k": 3})
client = Groq(api_key=GROQ_API_KEY)


@app.route("/")
def index():
    if 'mode' not in session:
        session['mode'] = 'chat'
    if 'chat_history' not in session:
        session['chat_history'] = []
    return render_template("chat.html", chat_history=session['chat_history'])


@app.route("/set_mode", methods=["POST"])
def set_mode():
    selected_mode = request.form.get("mode")
    session['mode'] = selected_mode
    print("[DEBUG] Mode switched to:", selected_mode)
    return jsonify({"status": "success", "mode": selected_mode})


@app.route("/get", methods=["POST"])
def chat():
    user_query = request.form["msg"]
    mode = session.get('mode', 'chat')
    print("[DEBUG] /get triggered with mode:", mode)

    if 'chat_history' not in session:
        session['chat_history'] = []

    messages = [{"role": "system", "content": "You are a helpful health and nutrition assistant."}]

    if mode == 'chat':
        retrieved_docs = retriever.invoke(user_query)
        context_text = "\n\n".join([doc.page_content for doc in retrieved_docs])
        history = session['chat_history'][-8:]
        for chat in history:
            if chat['sender'] == 'user':
                messages.append({"role": "user", "content": chat['message']})
            elif chat['sender'] == 'bot':
                messages.append({"role": "assistant", "content": chat['message']})
        messages.append({
            "role": "user",
            "content": f"Context:\n{context_text}\n\nUser Question: {user_query}"
        })

    elif mode == 'nutrition':
        words = user_query.strip().split()
      
        messages.append({
                "role": "user",
                "content": f"""Act as a nutrition ingredient analysis expert. Analyze the following ingredient list or food label content. Provide a clear, structured, and spaced multiline output if it is a list of ingredients.

1. **Plain Summary of Ingredients**
- List ingredients clearly

2. **Ingredient Categorization**
- Categorize each (e.g., emulsifier, preservative)
-Give their affects on health and information about them 

3. **Health Impact Tags**
- Acne, PCOS, diabetes, gut health etc.

4. **Red Flags or Vague Terms**
- e.g., 'natural flavoring', 'spices'

5. **Warnings**
- High sugar, sodium, bad fats, hormone disruptors, etc.

6. **Diet Compatibility**
- Keto, sugar-free, low-carb, gluten-free,vegan etc.

7. **Final Verdict**
- Yes/No with reason. Is it a health-conscious choice?


Otherwise-You are a nutrition expert. Explain in a clear and factual way about the ingredient related or general nutrition related question asked.Give affects of the food on health and necessary precautions, risks associated if any:




Ensure spacing, numbered sections, and line breaks. Avoid single-paragraph responses.

Input: {user_query}"""
            })

    else:
        print("[WARNING] Unknown mode fallback to chat.")
        mode = 'chat'

    response = client.chat.completions.create(
        model="llama3-8b-8192",
        messages=messages
    )
    result = response.choices[0].message.content

    session['chat_history'].append({"sender": "user", "message": user_query})
    session['chat_history'].append({"sender": "bot", "message": result})

    return jsonify({"response": result, "chat_history": session['chat_history']})


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

        mode = session.get('mode', 'chat')  # Use current mode from session
        messages = [{"role": "system", "content": "You are a helpful health focused,healthcare,disease,sickness relieve,fitness,medicine and nutrition assistant."}]

        if mode == "nutrition":
            messages.append({
                "role": "user",
                "content": f"""Act as a nutrition ingredient analysis expert. Analyze the following ingredient list or food label content. Provide a clear, structured, and spaced multiline output.

                1. **Plain Summary of Ingredients**
                - List ingredients clearly

                2. **Ingredient Categorization**
                - Categorize each (e.g., emulsifier, preservative)
                -their their affects on health and information about them 

                3. **Health Impact Tags**
                - Acne, PCOS, diabetes, gut health, blood pressure etc.

                4. **Red Flags or Vague Terms**
                - e.g., 'natural flavoring', 'spices', 'apple juice concentrate', 'sugar substitutes like maltodextrin, maltitol, aspartame, etc.'

                5. **Warnings**
                - High sugar, sodium, bad fats, hormone disruptors, etc.

                6. **Diet Compatibility**
                - Keto, sugar-free, low-carb, gluten-free, vegan etc.

                7. **Final Verdict**
                - Yes/No with reason. Is it a health-conscious choice?

                Ensure spacing, numbered sections, and line breaks. Avoid single-paragraph responses.

                Input: {cleaned_text}"""
                            })

        else:
            # Chat Mode: Ask in conversational health-oriented tone
            messages.append({
                "role": "user",
                "content": f"""A user has uploaded a health related document ,food product label, medicine related document or any health realted document :\n\n{cleaned_text}\n\nProvide a health-oriented summary of this label in a conversational and helpful tone, focusing on general health and asking if any other help is needed related to it. Keep it friendly and simple."""
            })

        response = client.chat.completions.create(
            model="llama3-8b-8192",
            messages=messages
        )
        result = response.choices[0].message.content
        return jsonify({'response': result})

    except Exception as e:
        print("Image processing failed:", str(e))
        return jsonify({'error': 'Failed to process image'}), 500


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8080, debug=True)
