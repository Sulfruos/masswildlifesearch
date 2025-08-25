from flask import Flask, request, jsonify, render_template
from search_engine import WildlifeSearchEngine

app = Flask(__name__)

search_engine = WildlifeSearchEngine()

# API endpoint
@app.route('/api/search', methods=['POST'])
def search():
    data = request.get_json()
    query = data.get('query', '')

    if not query:
        return jsonify({'error', 'No query provided.'}), 400
    
    results = search_engine.query_docs(query)

    print(results)

    return jsonify({
        'query': query,
        'results': results
    })

@app.route('/')
def index():
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)