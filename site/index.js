// Dynamically load CSV data from an external file
let data = [];
let categories = [];

async function loadCSV(filePath) {
    const response = await fetch(filePath);
    const csvText = await response.text();
    
    const rows = csvText.trim().split('\n');
    const headers = rows.shift().split(',');

    return rows.map(row => {
        const rowData = row.split(',');
        return headers.reduce((obj, header, index) => {
            obj[header.trim()] = rowData[index]?.trim() || '';
            return obj;
        }, {});
    });
}

async function initialize() {
    data = await loadCSV('u_scores_filled.csv');

    // Extract categories dynamically
    if (data.length > 0) {
        const columns = Object.keys(data[0]);
        categories = columns.slice(2); // Assume categories start at index 2
    }

    displayFilters();
}

function displayFilters() {
    const table = document.getElementById('filterTable');
    const tableBody = table.querySelector('tbody');
    tableBody.innerHTML = ''; // Clear existing rows

    categories.forEach(category => {
        const row = tableBody.insertRow();
        row.innerHTML = `
            <td>${category}</td>
            <td>
                <label class="switch">
                    <input id="${category}" type="checkbox" onclick="displayRankings()">
                    <span class="slider round"></span>
                </label>
            </td>`;
    });
}

function toggleAllCheckboxes() {
    const checkboxes = document.querySelectorAll('#filterTable input[type="checkbox"]');
    const allChecked = Array.from(checkboxes).every(checkbox => checkbox.checked);

    checkboxes.forEach(checkbox => checkbox.checked = !allChecked);
    document.getElementById('toggleAll').textContent = allChecked ? 'Select All' : 'Deselect All';

    displayRankings();
}

function displayRankings() {
    const calculatedScores = data.map(university => {
        const totalScore = categories.reduce((sum, category) => {
            const checkbox = document.getElementById(category);
            if (checkbox?.checked) {
                sum += getScore(university.University, category);
            }
            return sum;
        }, 0);

        return { University: university.University, Continent: university.Continent || 'Unknown', Score: totalScore };
    });

    calculatedScores.sort((a, b) => b.Score - a.Score);

    const table = document.getElementById('rankingTable');
    const tableBody = table.querySelector('tbody');
    tableBody.innerHTML = ''; // Clear existing rows

    calculatedScores.forEach((university, index) => {
        const row = tableBody.insertRow();
        row.innerHTML = `
            <td>${index + 1}</td>
            <td>${university.University}</td>
            <td>${university.Score.toFixed(2)}</td>`;
    });
}

function getScore(universityName, categoryName) {
    const row = data.find(entry => entry.University === universityName);
    if (!row) return 0;

    const score = parseFloat(row[categoryName]);
    return isNaN(score) ? 0 : score;
}

// Initialize the application
initialize();
