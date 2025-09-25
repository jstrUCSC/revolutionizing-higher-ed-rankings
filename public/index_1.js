let data = [];
let categories = [];
let currentPage = 1;
const rowsPerPage = 25; // Set the number of rows per page
let selectedRegion = 'all'; // Default to 'all' regions
let lastFiltered = []; 
let selectedCountry = 'all';

const ACTIVE_FIELDS = [
    "Machine Learning",
    "Computer Vision & Image Processing",
  ];

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
    // data = await loadCSV('university_rankings.csv');
    data = await loadCSV('2_f_2.csv');

    // Extract categories dynamically
    if (data.length > 0) {
        const columns = Object.keys(data[0]);
        categories = columns.slice(2); // Assume categories start at index 2
    }

    displayFilters();       // Fields
    setupRegionFilter();    // Set up the region filter
    setupCountryFilter();   // Country filter
    displayRankings();
}

function displayFilters() {
    const table = document.getElementById('filterTable');
    const tableBody = table.querySelector('tbody');
    tableBody.innerHTML = ''; // Clear existing rows

    // Filter categories to exclude "Continent" and any non-research fields
    // const filteredCategories = categories.filter(category => category !== 'Continent');
    const filteredCategories = categories.filter(category => category !== 'Continent' && category !== 'Country');
    
    filteredCategories.forEach(category => {
        // const row = tableBody.insertRow();
        const isActive = ACTIVE_FIELDS.includes(category);

        const row = tableBody.insertRow();
        row.innerHTML = `
            <td>
                ${category}${isActive ? '': '<span class="coming">(coming soon)</span>'}
            </td>
            <td>
                <label class="switch">
                    <input id="${category}" type="checkbox" ${isActive ? 'checked' : 'disabled'} onclick="resetPageAndDisplayRankings()">
                    <span class="slider round"></span>
                </label>
            </td>`;
    });

    updateToggleAllButtonLabel();

}

function updateToggleAllButtonLabel() {
    const boxes = document.querySelectorAll('#filterTable input[type="checkbox"]:not([disabled])');
    const allChecked = Array.from(boxes).every(b => b.checked);
    const btn = document.getElementById('toggleAll');
    if (btn) btn.textContent = allChecked ? 'Deselect All' : 'Select All';
}

function setupRegionFilter() {
    const regionFilter = document.getElementById('regionFilter');
    regionFilter.addEventListener('change', resetPageAndDisplayRankings); // Add listener for region change
}

function setupCountryFilter() {
    const sel = document.getElementById('countryFilter');
    // 从数据中收集去重后的国家（忽略空/Unknown）
    const countries = Array.from(new Set(
        data.map(r => (r.Country || '').trim()).filter(s => s && s.toLowerCase() !== 'unknown')
    )).sort((a, b) => a.localeCompare(b));

    // 先清空，再插入 "All Countries" + 动态选项
    sel.innerHTML = '<option value="all">All Countries</option>' +
        countries.map(c => `<option value="${c}">${c}</option>`).join('');

    sel.addEventListener('change', () => {
        selectedCountry = sel.value;
        resetPageAndDisplayRankings();
    });
}

function resetPageAndDisplayRankings() {
    currentPage = 1;
    selectedRegion = document.getElementById('regionFilter').value; // Get selected region
    updateToggleAllButtonLabel();
    displayRankings();
}

function displayRankings() {
    const calculatedScores = [];
    const seenUniversities = new Set();

    data.forEach(university => {
        if (selectedRegion !== 'all' && university.Continent) {
            const continentMatch = university.Continent.trim().toLowerCase() === selectedRegion.toLowerCase();
            if (!continentMatch) return;
        }
        if (selectedCountry !== 'all') {
            const uniCountry = (university.Country || 'Unknown').trim();
            if (uniCountry !== selectedCountry) return;
        }

        const totalScore = categories.reduce((sum, category) => {
            const checkbox = document.getElementById(category);
            if (checkbox?.checked) sum += getScore(university.University, category);
            return sum;
        }, 0);

        if (!seenUniversities.has(university.University)) {
            calculatedScores.push({
                University: university.University,
                Continent: university.Continent || 'Unknown',
                Country: university.Country || 'Unknown',
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
    tableBody.innerHTML = ''; // Clear existing rows

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
        displayRankings();
    }
}

function nextPage() {
    const totalPages = Math.ceil((lastFiltered.length || 0) / rowsPerPage);
    if (currentPage < totalPages) {
        currentPage++;
        displayRankings();
    }
}

function getScore(universityName, categoryName) {
    const row = data.find(entry => entry.University === universityName);
    if (!row) return 0;

    const score = parseFloat(row[categoryName]);
    return isNaN(score) ? 0 : score;
}

function toggleAllCheckboxes() {
    const checkboxes = document.querySelectorAll('#filterTable input[type="checkbox"]:not([disabled])');
    const allChecked = Array.from(checkboxes).every(checkbox => checkbox.checked);

    checkboxes.forEach(checkbox => checkbox.checked = !allChecked);
    // document.getElementById('toggleAll').textContent = allChecked ? 'Select All' : 'Deselect All';
    updateToggleAllButtonLabel();
    resetPageAndDisplayRankings(); // Re-display rankings after toggle
}

// Initialize the application at the end of the script
initialize();



