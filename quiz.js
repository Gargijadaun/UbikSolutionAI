let currentQuestion = 0;
let answers = [];
let questions = [];

document.addEventListener('DOMContentLoaded', () => {
    fetch('/get-questions')
        .then(res => res.json())
        .then(data => {
            questions = data;
            showQuestion();
        });
});

function showQuestion() {
    if (currentQuestion >= questions.length) {
        submitQuiz();
        return;
    }
    const q = questions[currentQuestion];
    document.getElementById('questionText').textContent = q.question;
}

function submitAnswer() {
    const input = document.getElementById('answerInput').value;
    answers.push(input);
    currentQuestion++;
    document.getElementById('answerInput').value = '';
    showQuestion();
}

function submitQuiz() {
    fetch('/submit-quiz', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ answers })
    })
    .then(res => res.json())
    .then(data => {
        renderSummary(data);
    });
}

function renderSummary(data) {
    const result = document.getElementById('result');
    result.innerHTML = `<h2>Total Score: ${data.total_score} / ${data.max_score}</h2>`;
    data.summary.forEach((entry, i) => {
        result.innerHTML += `
            <div>
                <h3>Q${i + 1}: ${questions[i].question}</h3>
                <p><strong>Score:</strong> ${entry.score}</p>
                <p><strong>Feedback:</strong> ${entry.feedback}</p>
                <p><strong>Ideal Answer:</strong> ${entry.ideal_answer}</p>
                <hr/>
            </div>
        `;
    });
}
