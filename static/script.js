async function search() {
    const query = document.getElementById('searchInput').value.trim();
    
    if (!query) {
        alert('Please enter a search query!');
        return;
    }
    
    // Show loading state
    const resultsDiv = document.getElementById('results');
    resultsDiv.innerHTML = '<div class="loading">Searching...</div>';
    
    try {
        const response = await fetch('/api/search', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ query: query })
        });
        
        const data = await response.json();
        
        if (response.ok) {
            displayResults(data);
        } else {
            resultsDiv.innerHTML = `<div class="error">Error: ${data.error}</div>`;
        }
    } catch (error) {
        resultsDiv.innerHTML = '<div class="error">Something went wrong. Please try again.</div>';
    }
}

function displayResults(data) {
    const resultsDiv = document.getElementById('results');
    
    if (data.results.length === 0) {
        resultsDiv.innerHTML = '<div class="no-results">No results found. Try a different search!</div>';
        return;
    }
    
    let html = `<div class="results-header">Results for "${data.query}"</div>`;
    
    data.results.forEach((result, index) => {
        html += `
            <div class="result-item">
                <div class="result-score">RRF Score: ${(result.score * 100).toFixed(1)}</div>
                <div class="result-content">${result.content}</div>
            </div>
        `;
    });
    
    resultsDiv.innerHTML = html;
}

// Allow Enter key to trigger search
document.getElementById('searchInput').addEventListener('keypress', function(e) {
    if (e.key === 'Enter') {
        search();
    }
});