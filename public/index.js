let data = [];
let categories = [];
let currentPage = 1;
const rowsPerPage = 25;
let selectedRegion = 'all';
let lastFiltered = [];

const ACTIVE_FIELDS = [
    "Machine Learning",
    "Computer Vision & Image Processing",
];

const DISPLAY_LABELS = {
    'Machine Learning': 'Machine Learning',
    'Computer Vision & Image Processing': 'Computer Vision & Image Processing',
};

const EPS = 1e-9;
const NORMALIZE = 'unit-variance';
let fieldStats = {};

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
    data = await loadCSV('2_f.csv');

    // Extract categories dynamically
    if (data.length > 0) {
        const columns = Object.keys(data[0]);
        categories = columns.slice(2); // Assume categories start at index 2
    }

    computeFieldStats();
    displayFilters();
    setupRegionFilter();
    displayRankings();
}

function computeFieldStats() {
    fieldStats = {};
    const cols = ACTIVE_FIELDS;
    cols.forEach(cat => {
        const vals = data.map(r => parseFloat(r[cat])).filter(v => !isNaN(v) && v > 0);
        const n = vals.length;
        const mean = n ? vals.reduce((a,b)=>a+b,0)/n : 0;
        const varSample = n>1 ? vals.reduce((a,b)=>a+(b-mean)*(b-mean),0)/(n-1) : 0;
        const std = Math.sqrt(varSample);
        fieldStats[cat] = { mean, std };
    });
}

function displayFilters() {
    const table = document.getElementById('filterTable');
    const tableBody = table.querySelector('tbody');
    tableBody.innerHTML = '';

    // Add active fields with checkboxes
    ACTIVE_FIELDS.forEach(category => {
        const label = DISPLAY_LABELS[category] || category;
        const safeId = toId(category);
        const row = tableBody.insertRow();
        row.innerHTML = `
            <td>${label}</td>
            <td>
                <label class="switch">
                    <input
                        id="${safeId}"
                        class="field-checkbox"
                        type="checkbox"
                        data-field="${category}"
                        checked
                        onclick="resetPageAndDisplayRankings()"
                    >
                    <span class="slider round"></span>
                </label>
            </td>`;
    });

    // Add coming soon row
    const comingRow = tableBody.insertRow();
    comingRow.innerHTML = `
        <td>Coming Soon</td>
        <td>
            <label class="switch">
                <input type="checkbox" disabled>
                <span class="slider round"></span>
            </label>
        </td>`;

    updateToggleAllButtonLabel();
}

function toId(name) {
    return 'field-' + name.toLowerCase().replace(/[^a-z0-9]+/g, '-');
}

function updateToggleAllButtonLabel() {
    const boxes = document.querySelectorAll('.field-checkbox');
    const allChecked = Array.from(boxes).every(b => b.checked);
    const btn = document.getElementById('toggleAll');
    if (btn) btn.textContent = allChecked ? 'Deselect All' : 'Select All';
}

function setupRegionFilter() {
    const regionFilter = document.getElementById('regionFilter');
    regionFilter.addEventListener('change', resetPageAndDisplayRankings);
}

function resetPageAndDisplayRankings() {
    currentPage = 1;
    selectedRegion = document.getElementById('regionFilter').value;
    updateToggleAllButtonLabel();
    displayRankings();
}

function getSelectedCategories() {
    return [...document.querySelectorAll('.field-checkbox:checked')]
           .map(el => el.dataset.field);
}

function getRawScore(univ, field) {
    const row = data.find(e => e.University === univ);
    if (!row) return 0;
    const s = parseFloat(row[field]);
    return isNaN(s) ? 0 : s;
}

function getNormalizedScore(univ, field) {
    const s = getRawScore(univ, field);
    if (!(s > 0)) return 0;
    const stats = fieldStats[field] || { std: 1 };
    const std = stats.std > EPS ? stats.std : 1;
    return s / std;
}

function displayRankings() {
    const calculatedScores = [];
    const seenUniversities = new Set();

    data.forEach(university => {
        const selectedCats = getSelectedCategories();

        let totalScore = 0;
        if (selectedCats.length >= 2) {
            // Normalized harmonic mean for multiple fields
            let denom = 0, count = 0;
            selectedCats.forEach(cat => {
                const x = getNormalizedScore(university.University, cat);
                if (x > 0) { 
                    denom += 1 / (x + EPS); 
                    count++;
                }
            });
            totalScore = count > 0 ? (count / denom) : 0;
        } else if (selectedCats.length === 1) {
            // Raw score for single field
            totalScore = getRawScore(university.University, selectedCats[0]);
        } else {
            totalScore = 0;
        }

        // Apply region filtering
        if (selectedRegion !== 'all' && university.Continent) {
            const continentMatch = university.Continent.trim().toLowerCase() === selectedRegion.toLowerCase();
            if (!continentMatch) {
                return;
            }
        }

        if (!seenUniversities.has(university.University) && totalScore > 0) {
            calculatedScores.push({
                University: university.University,
                Continent: university.Continent || 'Unknown',
                Score: totalScore
            });
            seenUniversities.add(university.University);
        }
    });

    calculatedScores.sort((a, b) => b.Score - a.Score);

    lastFiltered = calculatedScores;
    displayPage(currentPage, calculatedScores);
    updatePageIndicator(currentPage, Math.ceil(calculatedScores.length / rowsPerPage));
}

function displayPage(page, data) {
    const table = document.getElementById('rankingTable');
    const tableBody = table.querySelector('tbody');
    tableBody.innerHTML = '';

    const start = (page - 1) * rowsPerPage;
    const end = page * rowsPerPage;

    data.slice(start, end).forEach((university, index) => {
        const row = tableBody.insertRow();
        row.innerHTML = `
            <td>${start + index + 1}</td>
            <td>${university.University}</td>
            <td>${university.Score.toFixed(2)}</td>`;
    });
}

function updatePageIndicator(page, totalPages) {
    document.getElementById('pageIndicator').textContent = `Page ${page} of ${totalPages}`;
}

function prevPage() {
    if (currentPage > 1) {
        currentPage--;
        displayPage(currentPage, lastFiltered);
        updatePageIndicator(currentPage, Math.ceil(lastFiltered.length / rowsPerPage));
    }
}

function nextPage() {
    const totalPages = Math.ceil(lastFiltered.length / rowsPerPage);
    if (currentPage < totalPages) {
        currentPage++;
        displayPage(currentPage, lastFiltered);
        updatePageIndicator(currentPage, totalPages);
    }
}

function toggleAllCheckboxes() {
    const boxes = document.querySelectorAll('.field-checkbox');
    const allChecked = [...boxes].every(b => b.checked);
    boxes.forEach(b => b.checked = !allChecked);
    updateToggleAllButtonLabel();
    resetPageAndDisplayRankings();
}

// Initialize the application
initialize();