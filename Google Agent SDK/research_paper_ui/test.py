import os
from flask import Flask, render_template_string, request, jsonify
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA

# Load environment variables
load_dotenv()
os.environ['OPENAI_API_KEY'] = os.getenv('OPENAI_API_KEY')

app = Flask(__name__)

# Configuration
PDF_FOLDER = "pdfs"
PDF_FILES = [
    "Disease Markers - 2021 - Shi - Machine Learning of Schizophrenia Detection with Structural and Functional Neuroimaging.pdf",
    "Machine learning techniques in a structural and functional MRI diagnostic approach in schizophrenia a systematic review.pdf"
]

# HTML Templates
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Research Paper Analyzer</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css">
    <style>
        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(10px); }
            to { opacity: 1; transform: translateY(0); }
        }
        .fade-in { animation: fadeIn 0.5s ease-out forwards; }
        .prose { max-width: none; }
    </style>
</head>
<body class="bg-gray-50">
    <nav class="bg-blue-600 text-white shadow-lg">
        <div class="container mx-auto px-4 py-3 flex justify-between items-center">
            <div class="flex items-center space-x-2">
                <i class="fas fa-book-open text-2xl"></i>
                <span class="text-xl font-bold">ResearchAnalyzer</span>
            </div>
        </div>
    </nav>

    <div class="container mx-auto px-4 py-8">
        <div class="max-w-4xl mx-auto">
            <div class="text-center mb-12">
                <h1 class="text-4xl font-bold text-gray-800 mb-4">Research Paper Analysis System</h1>
                <p class="text-xl text-gray-600">Extract insights from medical research papers using AI</p>
            </div>

            <div class="bg-white rounded-xl shadow-lg p-6 mb-8">
                <div class="flex items-center mb-6">
                    <div class="bg-blue-100 p-3 rounded-full mr-4">
                        <i class="fas fa-search text-blue-600 text-xl"></i>
                    </div>
                    <h2 class="text-2xl font-semibold text-gray-800">Ask About the Research</h2>
                </div>

                <form id="searchForm" class="mb-4">
                    <div class="relative">
                        <input type="text" id="query" name="query" 
                               class="w-full p-4 pl-12 rounded-lg border border-gray-300 focus:ring-2 focus:ring-blue-500 focus:border-blue-500 outline-none transition"
                               placeholder="What would you like to know from the research papers?">
                        <i class="fas fa-search absolute left-4 top-1/2 transform -translate-y-1/2 text-gray-400"></i>
                    </div>
                    <button type="submit" 
                            class="mt-4 w-full bg-blue-600 hover:bg-blue-700 text-white font-semibold py-3 px-4 rounded-lg transition flex items-center justify-center">
                        <i class="fas fa-paper-plane mr-2"></i> Analyze Papers
                    </button>
                </form>

                <div class="text-sm text-gray-500">
                    <p>Example queries: "What features were used for detection?", "What machine learning models were used?", "What were the key findings?"</p>
                </div>
            </div>

            <div id="loadingIndicator" class="hidden text-center py-8">
                <div class="inline-block animate-spin rounded-full h-12 w-12 border-t-2 border-b-2 border-blue-600 mb-4"></div>
                <p class="text-gray-600">Analyzing research papers...</p>
            </div>

            <div id="resultsContainer" class="hidden bg-white rounded-xl shadow-lg p-6">
                <!-- Results will be displayed here -->
            </div>
        </div>
    </div>

    <footer class="bg-gray-800 text-white py-6 mt-12">
        <div class="container mx-auto px-4 text-center">
            <p>© 2023 Research Paper Analyzer. All rights reserved.</p>
        </div>
    </footer>

    <script>
        document.addEventListener('DOMContentLoaded', function() {
            const searchForm = document.getElementById('searchForm');
            const loadingIndicator = document.getElementById('loadingIndicator');
            const resultsContainer = document.getElementById('resultsContainer');

            searchForm.addEventListener('submit', function(e) {
                e.preventDefault();

                const query = document.getElementById('query').value.trim();
                if (!query) return;

                loadingIndicator.classList.remove('hidden');
                resultsContainer.classList.add('hidden');

                fetch('/search', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/x-www-form-urlencoded',
                    },
                    body: `query=${encodeURIComponent(query)}`
                })
                .then(response => {
                    if (!response.ok) {
                        throw new Error('Network response was not ok');
                    }
                    return response.json();
                })
                .then(data => {
                    loadingIndicator.classList.add('hidden');

                    if (data.error) {
                        resultsContainer.innerHTML = `
                            <div class="bg-red-100 border-l-4 border-red-500 text-red-700 p-4">
                                <p>${data.error}</p>
                            </div>
                        `;
                    } else {
                        resultsContainer.innerHTML = `
                            <div class="mb-6 fade-in">
                                <h2 class="text-xl font-semibold text-gray-800 mb-3">Your Question:</h2>
                                <div class="bg-gray-100 p-4 rounded-lg">
                                    <p>${query}</p>
                                </div>
                            </div>

                            <div class="mb-6 fade-in" style="animation-delay: 0.1s">
                                <h2 class="text-xl font-semibold text-gray-800 mb-3">Analysis Results:</h2>
                                <div class="bg-blue-50 p-4 rounded-lg prose">
                                    <p>${data.answer.replace(/\n/g, '<br>')}</p>
                                </div>
                            </div>

                            <div class="fade-in" style="animation-delay: 0.2s">
                                <h2 class="text-xl font-semibold text-gray-800 mb-3">Source Documents:</h2>
                                <ul class="space-y-2">
                                    ${data.sources.map(source => `
                                        <li class="flex items-start">
                                            <i class="fas fa-file-pdf text-red-500 mt-1 mr-2"></i>
                                            <span>${source}</span>
                                        </li>
                                    `).join('')}
                                </ul>
                            </div>
                        `;
                    }

                    resultsContainer.classList.remove('hidden');
                    resultsContainer.scrollIntoView({ behavior: 'smooth' });
                })
                .catch(error => {
                    console.error('Error:', error);
                    loadingIndicator.classList.add('hidden');
                    resultsContainer.innerHTML = `
                        <div class="bg-red-100 border-l-4 border-red-500 text-red-700 p-4">
                            <p>An error occurred: ${error.message}</p>
                        </div>
                    `;
                    resultsContainer.classList.remove('hidden');
                });
            });
        });
    </script>
</body>
</html>
"""


def get_pdf_path(filename):
    """Get absolute path to PDF file"""
    return os.path.join(os.path.dirname(__file__), PDF_FOLDER, filename)


def initialize_retriever():
    """Initialize the vector store and retriever"""
    documents = []

    for pdf_file in PDF_FILES:
        try:
            file_path = get_pdf_path(pdf_file)
            if not os.path.exists(file_path):
                print(f"Warning: PDF file not found: {file_path}")
                continue

            loader = PyPDFLoader(file_path)
            documents.extend(loader.load())
        except Exception as e:
            print(f"Error loading {pdf_file}: {str(e)}")
            continue

    if not documents:
        raise RuntimeError("No PDF documents could be loaded")

    vector_store = Chroma(
        embedding_function=OpenAIEmbeddings(),
        persist_directory=os.path.join(os.path.dirname(__file__), "chroma_db"),
        collection_name="Research-Paper-Prediction"
    )

    if vector_store._collection.count() == 0:
        vector_store.add_documents(documents)

    return vector_store.as_retriever(search_kwargs={"k": 3})


# Initialize retriever
retriever = None
try:
    retriever = initialize_retriever()
except Exception as e:
    print(f"Failed to initialize retriever: {str(e)}")


@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)


@app.route('/search', methods=['POST'])
def search():
    if retriever is None:
        return jsonify({
            "error": "System initialization failed. Please check if PDF files are available."
        }), 500

    query = request.form.get('query')
    if not query:
        return jsonify({
            "error": "Please provide a search query"
        }), 400

    try:
        model = ChatOpenAI(model='gpt-4')
        rag_chain = RetrievalQA.from_chain_type(
            llm=model,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=True,
            verbose=True
        )

        result = rag_chain.invoke(query)
        sources = list(set([doc.metadata.get("source") for doc in result["source_documents"]]))

        return jsonify({
            "answer": result["result"],
            "sources": sources
        })
    except Exception as e:
        return jsonify({
            "error": f"An error occurred during analysis: {str(e)}"
        }), 500


if __name__ == '__main__':
    app.run(debug=True)